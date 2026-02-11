#!/usr/bin/env python3
"""
Drone Follow — a Hailo pipeline app that follows a person with a drone.

Three modes:
    --dry-run:       Mock detections + drone (Gazebo). No Hailo needed.
    --hailo-dry-run: Real Hailo detection + connect and control simulation (Gazebo).
    Default (live):  Real Hailo detection + real drone.

Dry-run patterns (--mock-pattern):
    static: Person stands still at a specific (x, y) coordinate.
    circle: Person walks in a large 3D circle (changes angle and distance).
    line:   Person walks in a straight line away from the drone.
    sweep:  Person paces back and forth (testing yaw).

Usage:
    python drone_follow.py --dry-run --mock-pattern circle
    python drone_follow.py --hailo-dry-run --input /path/to/video.mp4 --input-codec h264
    python drone_follow.py --input rpi  # live mode with camera + drone

Pipeline options (--input, --input-codec, etc.) are passed through to the tiling pipeline.
"""

import argparse
import asyncio
import math
import os
import signal
import sys
import threading 
import time
import subprocess
import mavsdk
from dataclasses import dataclass
from typing import Optional

# MAVSDK
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    label: str
    confidence: float
    center_x: float      # 0.0 to 1.0
    center_y: float      # 0.0 to 1.0
    bbox_height: float   # 0.0 to 1.0
    timestamp: float

class SharedDetectionState:
    def __init__(self):
        self._lock = threading.Lock()
        self._detection: Optional[Detection] = None
        self._frame_count: int = 0

    def update(self, detection: Optional[Detection]):
        with self._lock:
            self._detection = detection
            self._frame_count += 1

    def get_latest(self):
        with self._lock:
            return self._detection, self._frame_count

# ---------------------------------------------------------------------------
# FOV-aware proportional controller
# ---------------------------------------------------------------------------

@dataclass
class ControllerConfig:
    hfov: float = 66.0
    vfov: float = 41.0
    kp_yaw: float = 2.0
    dead_zone_deg: float = 2.0
    max_yawspeed: float = 90.0
    kp_down: float = 0.08
    max_down_speed: float = 1.5
    target_bbox_height: float = 0.3
    kp_forward: float = 3.0
    dead_zone_height: float = 0.05
    max_forward: float = 2.0
    max_backward: float = 1.0
    search_yawspeed: float = 20.0
    detection_timeout_s: float = 0.5
    control_loop_hz: float = 10.0
    fixed_altitude: bool = True

def compute_velocity_command(detection: Optional[Detection], config: ControllerConfig) -> VelocityBodyYawspeed:
    if detection is None:
        return VelocityBodyYawspeed(0.0, 0.0, 0.0, config.search_yawspeed)

    error_x_deg = (detection.center_x - 0.5) * config.hfov
    error_y_deg = (detection.center_y - 0.5) * config.vfov

    # Yaw (Spin)
    yawspeed = 0.0 if abs(error_x_deg) < config.dead_zone_deg else config.kp_yaw * error_x_deg
    yawspeed = max(-config.max_yawspeed, min(config.max_yawspeed, yawspeed))

    # Altitude (Climb)
    down = 0.0
    if not config.fixed_altitude:
        down = 0.0 if abs(error_y_deg) < config.dead_zone_deg else config.kp_down * error_y_deg
        down = max(-config.max_down_speed, min(config.max_down_speed, down))

    # Forward (Distance)
    height_error = config.target_bbox_height - detection.bbox_height
    forward = 0.0 if abs(height_error) < config.dead_zone_height else config.kp_forward * height_error
    forward = max(-config.max_backward, min(config.max_forward, forward))

    return VelocityBodyYawspeed(forward, 0.0, down, yawspeed)


# ---------------------------------------------------------------------------
# Detached MAVSDK Server (for graceful shutdown)
# ---------------------------------------------------------------------------

class DetachedMavsdkServer:
    """
    Manages a mavsdk_server process that is detached from the current session,
    so it doesn't die on Ctrl+C (SIGINT). This allows the Python script to
    catch SIGINT and perform a graceful landing sequence using the server.
    """
    def __init__(self, connection_url, port=50051):
        self.connection_url = connection_url
        self.port = port
        self.process = None

    def __enter__(self):
        # If already using grpc, no need to start a server
        if self.connection_url.startswith("grpc://"):
            return self.connection_url
            
        # Try to find mavsdk_server binary
        try:
            server_path = os.path.join(os.path.dirname(mavsdk.__file__), 'bin', 'mavsdk_server')
        except Exception:
            server_path = None
            
        if not server_path or not os.path.exists(server_path):
            print(f"[drone] Warning: mavsdk_server not found at {server_path}, using default System() behavior")
            return self.connection_url # Fallback to default behavior

        # Pick a random port to avoid conflicts? 50051 is default.
        # Let's use 50051 for now.
        cmd = [server_path, "-u", self.connection_url, "-p", str(self.port)]
        print(f"[drone] Starting detached mavsdk_server: {' '.join(cmd)}")
        
        # start_new_session=True creates a new process group/session, 
        # isolating it from terminal signals like SIGINT
        self.process = subprocess.Popen(
            cmd, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        return f"grpc://127.0.0.1:{self.port}"

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()


# ---------------------------------------------------------------------------
# Unified Simulation Engine (Dry-Run Only)
# ---------------------------------------------------------------------------

def apply_physics_step(cx, cy, bh, cmd, dt, config):
    """Updates camera-frame position based on DRONE velocity."""
    # Drone Yaws Right (+) -> Target moves Left (-) in frame
    new_cx = cx - (cmd.yawspeed_deg_s * dt / config.hfov)
    # Drone Descends (+) -> Target moves Up (-) in frame
    new_cy = cy - (0.15 * cmd.down_m_s * dt)
    # Drone Approaches (+) -> Target Bbox Height increases (+)
    new_bh = bh + (0.2 * cmd.forward_m_s * dt)
    return new_cx, new_cy, new_bh

async def mock_control_loop(drone, shared_state, config, pattern, initial_x, initial_y, circle_diameter_m: float = 10.0):
    period = 1.0 / config.control_loop_hz
    t0 = time.monotonic()
    # Scale circle motion by diameter (reference 10 m = current default behavior)
    circle_scale = circle_diameter_m / 10.0

    # Internal state: where the "person" is relative to camera
    cx, cy, bh = initial_x, initial_y, config.target_bbox_height * 0.1

    try:
        while True:
            t = time.monotonic() - t0

            # 1. Update shared state
            shared_state.update(Detection("person", 0.99, cx, cy, bh, time.monotonic()))

            # 2. Get commands
            detection, _ = shared_state.get_latest()
            cmd = compute_velocity_command(detection, config)
            await drone.offboard.set_velocity_body(cmd)

            # 3. Apply World Movement (The Person Walking)
            dx, dy, dbh = 0.0, 0.0, 0.0
            if pattern == "circle":
                dx = 0.02 * circle_scale * math.cos(2 * math.pi * t / 10.0)   # Horizontal drift
                dbh = 0.008 * circle_scale * math.sin(2 * math.pi * t / 10.0)  # Distance drift
            elif pattern == "line":
                dbh = -0.005  # Walking straight away
            elif pattern == "sweep":
                dx = 0.03 * math.sin(2 * math.pi * t / 5.0)

            # 4. Reaction Step: Apply Drone Movement + World Drift
            cx, cy, bh = apply_physics_step(cx, cy, bh, cmd, period, config)
            cx += dx
            cy += dy
            bh += dbh

            # Bounds checks
            cx, cy = max(0.01, min(0.99, cx)), max(0.01, min(0.99, cy))
            bh = max(0.02, min(0.95, bh))

            print(f"\r[SIM] Pos:({cx:.2f}, {cy:.2f}) H:{bh:.2f} | CMD Yaw:{cmd.yawspeed_deg_s:+5.1f} Fwd:{cmd.forward_m_s:+4.1f}", end="")
            await asyncio.sleep(period)
    except asyncio.CancelledError:
        try:
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        except Exception:
            pass  # connection may already be gone; main finally will do stop/land
        raise

# ---------------------------------------------------------------------------
# Dry-Run Drone Runner
# ---------------------------------------------------------------------------

async def run_drone(
    args,
    shared_state,
    shutdown: asyncio.Event,
    shutdown_read_fd: Optional[int] = None,
):
    config = ControllerConfig(
        hfov=args.hfov, vfov=args.vfov, kp_yaw=args.yaw_gain,
        kp_down=args.pitch_gain, kp_forward=args.forward_gain,
        target_bbox_height=args.target_bbox_height, fixed_altitude=args.fixed_altitude
    )

    # If we have a pipe from parent, Ctrl+C is handled by parent; we shutdown when pipe is written
    if shutdown_read_fd is not None:
        loop = asyncio.get_running_loop()
        def _on_shutdown_pipe():
            try:
                os.read(shutdown_read_fd, 1)
            except (OSError, BlockingIOError):
                pass
            try:
                loop.remove_reader(shutdown_read_fd)
            except (OSError, ValueError):
                pass
            shutdown.set()
        loop.add_reader(shutdown_read_fd, _on_shutdown_pipe)

    mavsdk_server = DetachedMavsdkServer(args.connection)
    connection_url = mavsdk_server.__enter__()
    try:
        drone = System()
        await drone.connect(system_address=connection_url)

        print("[drone] Connecting and taking off...")
        async for state in drone.core.connection_state():
            if state.is_connected:
                break

        await drone.action.set_takeoff_altitude(args.takeoff_altitude)
        await drone.action.arm()
        await drone.action.takeoff()
        await asyncio.sleep(8)

        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        await drone.offboard.start()
        offboard_started = True

        task = asyncio.create_task(
            mock_control_loop(drone, shared_state, config, args.mock_pattern, args.mock_x, args.mock_y, args.mock_circle_diameter)
        )

        try:
            # Wait for mission duration or Ctrl+C (shutdown event)
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(shutdown.wait()),
                    asyncio.create_task(asyncio.sleep(args.mission_duration)),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
            if shutdown.is_set():
                print("\n[drone] Shutdown requested, landing...")
        except asyncio.CancelledError:
            print("\n[drone] Shutdown requested, landing...")
        finally:
            # 1. Stop control loop (may raise if cancel-handler RPC failed; ignore so we still land)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            # 2. Leave offboard mode and land safely (ignore second Ctrl+C during landing)
            if offboard_started and drone is not None:
                print("[drone] Landing safely - please wait (ignoring further Ctrl+C until done)...")
                try:
                    _ignore_sigint_during_landing(ignore=True)
                    try:
                        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                        await drone.offboard.stop()
                    except OffboardError as e:
                        print(f"[drone] Offboard stop: {e._result.result}")
                    except Exception as e:
                        _print_connection_error("[drone] Offboard stop", e)
                    print("[drone] Landing...")
                    try:
                        await drone.action.land()
                        await asyncio.sleep(8)
                    except Exception as e:
                        _print_connection_error("[drone] Land", e, hint=True)
                finally:
                    _ignore_sigint_during_landing(ignore=False)
        print("[drone] Done.")
    finally:
        mavsdk_server.__exit__(None, None, None)


def _print_connection_error(prefix: str, e: Exception, hint: bool = False) -> None:
    """Print a short message when failure is due to lost connection (e.g. sim closed)."""
    msg = str(e).lower()
    if "unavailable" in msg or "connection refused" in msg or "connection reset" in msg:
        print(f"{prefix}: connection lost (sim or MAVSDK backend closed).")
        if hint:
            print("[drone] Tip: press Ctrl+C once and wait for landing before closing the sim.")
    else:
        print(f"{prefix}: {e}")


def _ignore_sigint_during_landing(ignore: bool) -> None:
    """Ignore or restore SIGINT so a second Ctrl+C does not kill the process during landing."""
    try:
        if ignore:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        else:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
    except (ValueError, OSError):
        pass  # signal only works in main thread; ignore

# ---------------------------------------------------------------------------
# Hailo App Callback (module-level, like tiling.py app_callback)
# ---------------------------------------------------------------------------

def app_callback(element, buffer, user_data):
    """Tiling pipeline callback: pick largest person, update shared state."""
    import hailo
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    persons = [d for d in detections if d.get_label() == "person"]
    frame = user_data.get_count()
    if not persons:
        user_data.shared_state.update(None)
        print(f"[callback] frame={frame} clear (no person)")
        return
    best = max(persons, key=lambda d: d.get_bbox().width() * d.get_bbox().height())
    bbox = best.get_bbox()
    cx = bbox.xmin() + bbox.width() / 2
    cy = bbox.ymin() + bbox.height() / 2
    user_data.shared_state.update(Detection(
        label="person",
        confidence=best.get_confidence(),
        center_x=cx,
        center_y=cy,
        bbox_height=bbox.height(),
        timestamp=time.monotonic(),
    ))
    print(f"[callback] frame={frame} update person conf={best.get_confidence():.2f} center=({cx:.2f},{cy:.2f}) h={bbox.height():.2f}")


# ---------------------------------------------------------------------------
# Live Control Loop (hailo modes)
# ---------------------------------------------------------------------------

async def live_control_loop(drone, shared_state, config, shutdown):
    """Control loop for Hailo modes.

    Reads detections from shared_state, computes velocity commands.
    If drone is None (hailo-dry-run), prints commands instead.
    """
    period = 1.0 / config.control_loop_hz
    try:
        while not shutdown.is_set():
            detection, _ = shared_state.get_latest()
            if detection is not None:
                age = time.monotonic() - detection.timestamp
                if age > config.detection_timeout_s:
                    detection = None
            cmd = compute_velocity_command(detection, config)
            if drone is not None:
                await drone.offboard.set_velocity_body(cmd)
            else:
                tag = "TRACK" if detection is not None else "SEARCH"
                print(f"\r[{tag}] Yaw:{cmd.yawspeed_deg_s:+6.1f}°/s  "
                      f"Fwd:{cmd.forward_m_s:+5.2f}m/s  "
                      f"Down:{cmd.down_m_s:+5.2f}m/s", end="")
            await asyncio.sleep(period)
    except asyncio.CancelledError:
        if drone is not None:
            try:
                await drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            except Exception:
                pass
        raise


async def run_live_drone(args, shared_state, config, shutdown, shutdown_read_fd=None, takeoff_done=None, pipeline_quit_cb=None):
    """Connect to drone and run live control loop with Hailo detections.

    If takeoff_done is a threading.Event, it is set after takeoff and offboard start,
    so the Hailo pipeline can wait before starting.
    If pipeline_quit_cb is set, it is called at shutdown start so the pipeline stops first.
    """
    if shutdown_read_fd is not None:
        loop = asyncio.get_running_loop()
        def _on_shutdown_pipe():
            try:
                os.read(shutdown_read_fd, 1)
            except (OSError, BlockingIOError):
                pass
            try:
                loop.remove_reader(shutdown_read_fd)
            except (OSError, ValueError):
                pass
            shutdown.set()
        loop.add_reader(shutdown_read_fd, _on_shutdown_pipe)

    mavsdk_server = DetachedMavsdkServer(args.connection)
    connection_url = mavsdk_server.__enter__()
    try:
        drone = System()
        await drone.connect(system_address=connection_url)

        print("[drone] Connecting and taking off...")
        async for state in drone.core.connection_state():
            if state.is_connected:
                break

        await drone.action.set_takeoff_altitude(args.takeoff_altitude)
        await drone.action.arm()
        await drone.action.takeoff()
        await asyncio.sleep(15)

        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        await drone.offboard.start()
        offboard_started = True
        await asyncio.sleep(3)

        if takeoff_done is not None:
            takeoff_done.set()

        task = asyncio.create_task(live_control_loop(drone, shared_state, config, shutdown))

        try:
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(shutdown.wait()),
                    asyncio.create_task(asyncio.sleep(args.mission_duration)),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
            if shutdown.is_set():
                print("\n[drone] Shutdown requested, landing...")
        except asyncio.CancelledError:
            print("\n[drone] Shutdown requested, landing...")
        finally:
            # Land first, before cancelling the control task or quitting pipeline, so the
            # land command goes out while the connection is still likely alive.
            if offboard_started and drone is not None:
                print("[drone] Landing safely - please wait (ignoring further Ctrl+C until done)...")
                try:
                    _ignore_sigint_during_landing(ignore=True)
                    try:
                        await drone.offboard.set_velocity_body(
                            VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                        await drone.offboard.stop()
                    except OffboardError as e:
                        _print_connection_error("[drone] Offboard stop", e, hint=False)
                    except Exception as e:
                        _print_connection_error("[drone] Offboard stop", e, hint=False)
                    print("[drone] Landing...")
                    try:
                        await drone.action.land()
                        await asyncio.sleep(8)
                    except Exception as e:
                        _print_connection_error("[drone] Land", e, hint=False)
                finally:
                    _ignore_sigint_during_landing(ignore=False)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            if pipeline_quit_cb is not None:
                try:
                    pipeline_quit_cb()
                except Exception:
                    pass
        print("[drone] Done.")
    finally:
        mavsdk_server.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Hailo App Setup
# ---------------------------------------------------------------------------

def _add_drone_follow_args(parser):
    """Register drone-follow CLI flags on a pipeline parser."""
    group = parser.add_argument_group("drone-follow")
    group.add_argument("--hailo-dry-run", action="store_true",
                       help="Run Hailo pipeline + connect and control simulation (Gazebo)")
    group.add_argument("--dry-run", action="store_true",
                       help="Mock detections + real drone (Gazebo)")
    group.add_argument("--target-bbox-height", type=float, default=0.3)
    group.add_argument("--yaw-gain", type=float, default=2.0)
    group.add_argument("--forward-gain", type=float, default=3.0)
    group.add_argument("--pitch-gain", type=float, default=0.08)
    group.add_argument("--connection", default="udpin://0.0.0.0:14540")
    group.add_argument("--takeoff-altitude", type=float, default=3.0)
    group.add_argument("--mission-duration", type=float, default=300.0)
    group.add_argument("--hfov", type=float, default=66.0)
    group.add_argument("--vfov", type=float, default=41.0)
    group.add_argument("--fixed-altitude", action="store_true")
    group.add_argument("--detection-timeout", type=float, default=0.5,
                       help="Seconds before a stale detection triggers search mode")
    group.add_argument("--control-loop-hz", type=float, default=10.0)


def create_app(shared_state, eos_reached=None):
    """Create the tiling pipeline app with drone-follow callback.

    Follows the hailo-app pattern: build parser, create user_data,
    instantiate GStreamerTilingApp. If eos_reached is a threading.Event,
    EOS will set it instead of calling GStreamer shutdown (so we can land first).
    """
    from hailo_apps.python.pipeline_apps.tiling.tiling_pipeline import (
        GStreamerTilingApp, user_app_callback_class,
    )
    from hailo_apps.python.core.common.core import get_pipeline_parser

    class DroneFollowUserData(user_app_callback_class):
        def __init__(self, shared_state):
            super().__init__()
            self.shared_state = shared_state

    class DroneFollowTilingApp(GStreamerTilingApp):
        """Tiling app that on EOS sets eos_reached instead of calling shutdown."""
        def __init__(self, app_callback, user_data, parser=None, eos_reached=None):
            self._eos_reached = eos_reached
            super().__init__(app_callback, user_data, parser=parser)

        def on_eos(self):
            if self._eos_reached is not None:
                self._eos_reached.set()
            else:
                super().on_eos()

    parser = get_pipeline_parser()
    _add_drone_follow_args(parser)

    user_data = DroneFollowUserData(shared_state)
    return DroneFollowTilingApp(app_callback, user_data, parser=parser, eos_reached=eos_reached)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Pre-parse to determine mode before full arg parsing
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--dry-run", action="store_true")
    pre_parser.add_argument("--hailo-dry-run", action="store_true")
    pre_args, _ = pre_parser.parse_known_args()

    if pre_args.dry_run:
        # === DRY-RUN MODE (mock detections + real drone) ===
        parser = argparse.ArgumentParser()
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--mock-pattern", default="static",
                            choices=["static", "circle", "line", "sweep"])
        parser.add_argument("--mock-x", type=float, default=0.7)
        parser.add_argument("--mock-y", type=float, default=0.5)
        parser.add_argument("--mock-circle-diameter", type=float, default=10.0, metavar="M",
                            help="Circle pattern diameter in meters (default: 10)")
        parser.add_argument("--target-bbox-height", type=float, default=0.3)
        parser.add_argument("--yaw-gain", type=float, default=2.0)
        parser.add_argument("--forward-gain", type=float, default=3.0)
        parser.add_argument("--pitch-gain", type=float, default=0.08)
        parser.add_argument("--connection", default="udpin://0.0.0.0:14540")
        parser.add_argument("--takeoff-altitude", type=float, default=3.0)
        parser.add_argument("--mission-duration", type=float, default=300.0)
        parser.add_argument("--hfov", type=float, default=66.0)
        parser.add_argument("--vfov", type=float, default=41.0)
        parser.add_argument("--fixed-altitude", action="store_true")

        args = parser.parse_args()

        if hasattr(os, "fork") and os.name != "nt":
            r, w = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(w)
                try:
                    os.setsid()
                    shutdown = asyncio.Event()
                    asyncio.run(run_drone(args, SharedDetectionState(), shutdown,
                                         shutdown_read_fd=r))
                finally:
                    os.close(r)
                os._exit(0)
            os.close(r)
            def on_signal(*_args):
                print("\n[drone] Ctrl+C received, shutting down...")
                try:
                    os.write(w, b"x")
                except OSError:
                    pass
                try:
                    os.close(w)
                except OSError:
                    pass
            signal.signal(signal.SIGINT, on_signal)
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, on_signal)
            try:
                os.waitpid(pid, 0)
            except OSError:
                pass
        else:
            shutdown = asyncio.Event()
            def on_signal(*_args):
                if not shutdown.is_set():
                    shutdown.set()
                    print("\n[drone] Ctrl+C received, shutting down...")
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    for sig in (signal.SIGINT, signal.SIGTERM):
                        loop.add_signal_handler(sig, on_signal)
                except NotImplementedError:
                    signal.signal(signal.SIGINT, on_signal)
                    if hasattr(signal, "SIGTERM"):
                        signal.signal(signal.SIGTERM, on_signal)
                loop.run_until_complete(run_drone(args, SharedDetectionState(), shutdown))
            except KeyboardInterrupt:
                if not shutdown.is_set():
                    shutdown.set()
                print("\n[drone] Shutdown.")

    else:
        # === HAILO MODES (structured as a hailo-app) ===
        # Hailo-app pattern: create shared state, build app, run pipeline
        shared_state = SharedDetectionState()
        shutdown = asyncio.Event()
        eos_reached = threading.Event()
        app = create_app(shared_state, eos_reached=eos_reached)
        args = app.options_menu
        config = ControllerConfig(
            hfov=args.hfov, vfov=args.vfov,
            kp_yaw=args.yaw_gain, kp_down=args.pitch_gain,
            kp_forward=args.forward_gain,
            target_bbox_height=args.target_bbox_height,
            fixed_altitude=args.fixed_altitude,
            detection_timeout_s=args.detection_timeout,
            control_loop_hz=args.control_loop_hz,
        )
        takeoff_done = threading.Event()

        def _eos_to_shutdown():
            eos_reached.wait()
            shutdown.set()
        threading.Thread(target=_eos_to_shutdown, daemon=True).start()

        # Run pipeline in thread; start only after takeoff (non-daemon so we can join on exit)
        def run_pipeline():
            takeoff_done.wait()
            try:
                app.run()
            except SystemExit:
                pass
        pipeline_thread = threading.Thread(target=run_pipeline, daemon=False)
        pipeline_thread.start()

        # --- HAILO-DRY-RUN or LIVE: Hailo detections + connect and control drone (sim or real) ---
        def on_signal(*_):
            if not shutdown.is_set():
                shutdown.set()
                print("\n[drone] Ctrl+C received, shutting down...")
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(sig, on_signal)
            except NotImplementedError:
                signal.signal(signal.SIGINT, on_signal)
                if hasattr(signal, "SIGTERM"):
                    signal.signal(signal.SIGTERM, on_signal)
            loop.run_until_complete(
                run_live_drone(args, shared_state, config, shutdown,
                              takeoff_done=takeoff_done, pipeline_quit_cb=app.loop.quit))
        except KeyboardInterrupt:
            if not shutdown.is_set():
                shutdown.set()
            print("\n[drone] Shutdown.")
        finally:
            pipeline_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
