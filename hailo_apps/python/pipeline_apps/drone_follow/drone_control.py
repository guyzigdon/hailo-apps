import asyncio
import math
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import mavsdk
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
from mavsdk.telemetry import FlightMode

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
        self._available_ids: set = set()

    def update(self, detection: Optional[Detection], available_ids: set = None):
        with self._lock:
            self._detection = detection
            self._frame_count += 1
            if available_ids is not None:
                self._available_ids = available_ids

    def get_latest(self):
        with self._lock:
            return self._detection, self._frame_count
    
    def get_available_ids(self):
        """Get the set of currently visible detection IDs."""
        with self._lock:
            return self._available_ids.copy()

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
    max_backward: float = 2.0
    search_yawspeed: float = 20.0
    detection_timeout_s: float = 0.5
    search_timeout_s: float = 60.0
    control_loop_hz: float = 10.0
    fixed_altitude: bool = True
    max_bbox_height_safety: float = 0.8  # Safety limit: if bbox height > 0.8, we are too close
    yaw_only: bool = False

    @classmethod
    def from_args(cls, args):
        return cls(
            hfov=args.hfov,
            vfov=args.vfov,
            kp_yaw=args.yaw_gain,
            kp_down=args.pitch_gain,
            kp_forward=args.forward_gain,
            target_bbox_height=args.target_bbox_height,
            fixed_altitude=args.fixed_altitude,
            detection_timeout_s=getattr(args, 'detection_timeout', 0.5),
            control_loop_hz=getattr(args, 'control_loop_hz', 10.0),
            max_forward=getattr(args, 'max_forward', 1.0),
            max_backward=getattr(args, 'max_backward', 2.0),
            max_bbox_height_safety=getattr(args, 'max_bbox_height_safety', 0.8),
            search_timeout_s=getattr(args, 'search_timeout', 60.0),
        )

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

    # Forward (Distance) with safety check
    if config.yaw_only:
        forward = 0.0
    elif detection.bbox_height > config.max_bbox_height_safety:
        # Safety Check: If bbox is too big, we are dangerously close
        forward = -config.max_backward
    else:
        height_error = config.target_bbox_height - detection.bbox_height
        forward = 0.0 if abs(height_error) < config.dead_zone_height else config.kp_forward * height_error

        # Asymmetric limits: faster backward (negative), slower forward (positive)
        if forward > 0:
            forward = min(config.max_forward, forward)
        else:
            forward = max(-config.max_backward, forward)

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
        print(f"[drone] NOTE: The following ERROR messages from mavsdk_server are expected and harmless:")
        print(f"[drone]   - 'Unknown protocol' and 'Connection failed: Invalid connection URL'")
        print(f"[drone]   These occur during initial startup and can be safely ignored.")
        
        # start_new_session=True creates a new process group/session, 
        # isolating it from terminal signals like SIGINT
        # Redirect both stdout and stderr to suppress mavsdk_server's own logging
        try:
            devnull = open(os.devnull, 'w')
            self.process = subprocess.Popen(
                cmd, 
                stdout=devnull, 
                stderr=devnull,
                start_new_session=True,
                close_fds=True
            )
        except Exception:
            # Fallback if devnull approach fails
            self.process = subprocess.Popen(
                cmd, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        
        # Give server a moment to start before returning
        time.sleep(0.5)
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
    
    last_detection_time = time.monotonic()

    try:
        while True:
            t = time.monotonic() - t0

            # 1. Update shared state
            shared_state.update(Detection("person", 0.99, cx, cy, bh, time.monotonic()))
            # Since we always simulate a detection here in mock mode, we update last_detection_time
            last_detection_time = time.monotonic()

            # 2. Get commands
            detection, _ = shared_state.get_latest()
            
            # (In mock mode we don't really lose detection unless we simulate it, 
            # so the timeout check is less critical here but good for consistency)
            if time.monotonic() - last_detection_time > config.search_timeout_s:
                print(f"\n[SIM] Search timeout ({config.search_timeout_s}s) exceeded. Landing...")
                break

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
    config = ControllerConfig.from_args(args)

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

        manage_takeoff_landing = not getattr(args, 'no_takeoff_landing', False)
        if manage_takeoff_landing:
            print("[drone] Connecting and taking off...")
        else:
            print("[drone] Connecting (drone must already be in OFFBOARD)...")
        async for state in drone.core.connection_state():
            if state.is_connected:
                break

        if manage_takeoff_landing:
            await drone.action.set_takeoff_altitude(args.takeoff_altitude)
            await drone.action.arm()
            await drone.action.takeoff()
            await asyncio.sleep(8)
        else:
            await _require_offboard_mode(drone)

        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        await drone.offboard.start()
        offboard_started = True
        # When not manage_takeoff_landing we do not call offboard.stop() on exit (don't change mode)

        task = asyncio.create_task(
            mock_control_loop(drone, shared_state, config, args.mock_pattern, args.mock_x, args.mock_y, args.mock_circle_diameter)
        )
        watch_task = None
        if not manage_takeoff_landing:
            watch_task = asyncio.create_task(_watch_offboard_mode(drone, shutdown))

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
            if watch_task is not None:
                watch_task.cancel()
                try:
                    await watch_task
                except asyncio.CancelledError:
                    pass
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            if offboard_started and drone is not None:
                if manage_takeoff_landing:
                    try:
                        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                        await drone.offboard.stop()
                    except OffboardError as e:
                        print(f"[drone] Offboard stop: {e._result.result}")
                    except Exception as e:
                        _print_connection_error("[drone] Offboard stop", e)
                if manage_takeoff_landing:
                    print("[drone] Landing safely - please wait (ignoring further Ctrl+C until done)...")
                    try:
                        _ignore_sigint_during_landing(ignore=True)
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


def _exit_if_not_offboard(reason: str) -> None:
    """Exit the process immediately. Use when --no-takeoff-landing and drone must be OFFBOARD."""
    print(f"[drone] {reason}", file=sys.stderr)
    sys.stderr.flush()
    os._exit(1)


async def _require_offboard_mode(drone: System) -> None:
    """Get current flight mode; if not OFFBOARD, kill the app."""
    async for mode in drone.telemetry.flight_mode():
        if mode != FlightMode.OFFBOARD:
            _exit_if_not_offboard(
                f"Drone is not in OFFBOARD mode (current: {mode.name}). Exiting."
            )
        return


async def _watch_offboard_mode(drone: System, shutdown: asyncio.Event) -> None:
    """Background task: if flight mode ever leaves OFFBOARD, kill the app."""
    async for mode in drone.telemetry.flight_mode():
        if shutdown.is_set():
            return
        if mode != FlightMode.OFFBOARD:
            _exit_if_not_offboard(
                f"Drone left OFFBOARD mode (current: {mode.name}). Exiting."
            )


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
# Live Control Loop (hailo modes)
# ---------------------------------------------------------------------------

async def live_control_loop(drone, shared_state, config, shutdown):
    """Control loop for Hailo modes.

    Reads detections from shared_state, computes velocity commands.
    If drone is None (hailo-dry-run), prints commands instead.
    """
    period = 1.0 / config.control_loop_hz
    last_detection_time = time.monotonic()
    
    try:
        while not shutdown.is_set():
            detection, _ = shared_state.get_latest()
            
            if detection is not None:
                age = time.monotonic() - detection.timestamp
                if age > config.detection_timeout_s:
                    detection = None
                else:
                    last_detection_time = time.monotonic()
            
            # Check search timeout
            if time.monotonic() - last_detection_time > config.search_timeout_s:
                print(f"\n\n[drone] Search timeout ({config.search_timeout_s}s) exceeded - no person found.")
                print("[drone] Initiating safety landing...")
                shutdown.set()
                break

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


async def run_live_drone(args, shared_state, shutdown, shutdown_read_fd=None, takeoff_done=None, pipeline_quit_cb=None, config=None):
    """Connect to drone and run live control loop with Hailo detections.

    If takeoff_done is a threading.Event, it is set after takeoff and offboard start,
    so the Hailo pipeline can wait before starting.
    If pipeline_quit_cb is set, it is called at shutdown start so the pipeline stops first.
    If config is provided, use it directly (allows live mutation from web UI).
    """
    if config is None:
        config = ControllerConfig.from_args(args)
    
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

        manage_takeoff_landing = not getattr(args, 'no_takeoff_landing', False)
        if manage_takeoff_landing:
            print("[drone] Connecting and taking off...")
        else:
            print("[drone] Connecting (drone must already be in OFFBOARD)...")
        async for state in drone.core.connection_state():
            if state.is_connected:
                break

        if manage_takeoff_landing:
            await drone.action.set_takeoff_altitude(args.takeoff_altitude)
            await drone.action.arm()
            await drone.action.takeoff()
            await asyncio.sleep(15)
        else:
            await _require_offboard_mode(drone)

        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        await drone.offboard.start()
        offboard_started = True
        if manage_takeoff_landing:
            await asyncio.sleep(3)
        # When not manage_takeoff_landing we do not call offboard.stop() on exit (don't change mode)

        if takeoff_done is not None:
            takeoff_done.set()

        task = asyncio.create_task(live_control_loop(drone, shared_state, config, shutdown))
        watch_task = None
        if not manage_takeoff_landing:
            watch_task = asyncio.create_task(_watch_offboard_mode(drone, shutdown))

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
            if watch_task is not None:
                watch_task.cancel()
                try:
                    await watch_task
                except asyncio.CancelledError:
                    pass
            if offboard_started and drone is not None and manage_takeoff_landing:
                try:
                    await drone.offboard.set_velocity_body(
                        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                    await drone.offboard.stop()
                except OffboardError as e:
                    _print_connection_error("[drone] Offboard stop", e, hint=False)
                except Exception as e:
                    _print_connection_error("[drone] Offboard stop", e, hint=False)
                print("[drone] Landing safely - please wait (ignoring further Ctrl+C until done)...")
                try:
                    _ignore_sigint_during_landing(ignore=True)
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
