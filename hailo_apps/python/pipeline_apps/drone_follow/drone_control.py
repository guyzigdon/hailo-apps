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
from urllib.parse import urlparse

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
    kp_yaw: float = 5
    dead_zone_deg: float = 2.0
    max_yawspeed: float = 90.0
    kp_down: float = 0.08
    max_down_speed: float = 1.5
    target_bbox_height: float = 0.3
    kp_forward: float = 6.0
    kp_backward: float = 7.5
    dead_zone_height_percent: float = 5.0  # dead zone as % of target_bbox_height (default 5%)
    max_forward: float = 2.0
    max_backward: float = 3.0
    detection_timeout_s: float = 0.5
    search_enter_delay_s: float = 2.0
    search_timeout_s: float = 60.0
    control_loop_hz: float = 10.0
    fixed_altitude: bool = True
    max_bbox_height_safety: float = 0.8  # Safety limit: if bbox height > 0.8, we are too close
    yaw_only: bool = False
    reference_altitude_m: float = 3.0  # target_bbox_height is defined at this altitude; scales by (ref_alt/current_alt)
    # Bottom-of-frame backward: bbox bottom edge beyond this triggers backward
    bottom_y_threshold: float = 0.85
    # Search mode
    search_yawspeed_slow: float = 10.0  # yaw speed during search (slower than tracking)
    search_vel_damp: float = 0.3        # dampening factor for forward/backward speed during search
    min_search_forward: float = 0.2     # minimum forward speed in search when last bbox was too small

    takeoff_altitude: float = 3.0

    @classmethod
    def from_args(cls, args):
        # Single source of defaults: dataclass values.
        defaults = cls()

        def _arg(*names, default):
            for name in names:
                value = getattr(args, name, None)
                if value is not None:
                    return value
            return default

        # yaw_only: only True when user explicitly passed --yaw-only.
        yaw_only = _arg("yaw_only", default=defaults.yaw_only)
        if not isinstance(yaw_only, bool):
            yaw_only = bool(yaw_only)

        ref_alt = _arg("reference_altitude", "reference_altitude_m", default=defaults.reference_altitude_m)
        ref_alt = ref_alt if ref_alt and ref_alt > 0 else defaults.reference_altitude_m

        return cls(
            hfov=_arg("hfov", default=defaults.hfov),
            vfov=_arg("vfov", default=defaults.vfov),
            kp_yaw=_arg("kp_yaw", "yaw_gain", default=defaults.kp_yaw),
            kp_down=_arg("kp_down", "pitch_gain", default=defaults.kp_down),
            kp_forward=float(_arg("kp_forward", "forward_gain", default=defaults.kp_forward)),
            kp_backward=_arg("kp_backward", "backward_gain", default=defaults.kp_backward),
            target_bbox_height=_arg("target_bbox_height", default=defaults.target_bbox_height),
            dead_zone_height_percent=_arg("dead_zone_height_percent", default=defaults.dead_zone_height_percent),
            reference_altitude_m=ref_alt,
            fixed_altitude=_arg("fixed_altitude", default=defaults.fixed_altitude),
            yaw_only=yaw_only,
            detection_timeout_s=_arg("detection_timeout", "detection_timeout_s", default=defaults.detection_timeout_s),
            search_enter_delay_s=_arg("search_enter_delay", "search_enter_delay_s", default=defaults.search_enter_delay_s),
            control_loop_hz=_arg("control_loop_hz", default=defaults.control_loop_hz),
            max_forward=_arg("max_forward", default=defaults.max_forward),
            max_backward=_arg("max_backward", default=defaults.max_backward),
            max_bbox_height_safety=_arg("max_bbox_height_safety", default=defaults.max_bbox_height_safety),
            search_timeout_s=_arg("search_timeout", "search_timeout_s", default=defaults.search_timeout_s),
            search_vel_damp=_arg("search_vel_damp", default=defaults.search_vel_damp),
            takeoff_altitude=_arg("takeoff_altitude", default=defaults.takeoff_altitude),
        )

def _calculate_forward_speed(
    detection: Detection,
    config: ControllerConfig,
    target_bh: float,
    explain: Optional[dict] = None,
) -> float:
    """Calculate forward/backward speed based on bbox height and bottom-of-frame position."""
    if config.yaw_only or config.kp_forward == 0:
        if explain is not None:
            explain.update(mode="disabled", target_bh=target_bh,
                           bbox_h=detection.bbox_height, final_forward=0.0)
        return 0.0

    if detection.bbox_height > config.max_bbox_height_safety:
        if explain is not None:
            explain.update(mode="safety_retreat", target_bh=target_bh,
                           bbox_h=detection.bbox_height, final_forward=-config.max_backward)
        return -config.max_backward

    height_delta = target_bh - detection.bbox_height
    dead_zone_height = (config.dead_zone_height_percent / 100.0) * target_bh
    reason = "dead_zone"

    if abs(height_delta) < dead_zone_height:
        forward = 0.0
    elif height_delta > 0:
        forward = config.kp_forward * math.sqrt(height_delta)
        reason = "too_small"
    else:
        forward = -config.kp_backward * math.sqrt(-height_delta)
        reason = "too_big"
    raw_forward = forward

    # Bottom-of-frame backward: bbox bottom edge past threshold means drone is above and too close
    bottom_backward = 0.0
    max_y = detection.center_y + detection.bbox_height / 2
    if max_y > config.bottom_y_threshold:
        y_excess = max_y - config.bottom_y_threshold
        bottom_backward = config.kp_backward * math.sqrt(y_excess)
        forward = min(forward, -bottom_backward)

    # Clamp
    forward = max(-config.max_backward, min(config.max_forward, forward))

    if explain is not None:
        explain.update(mode="normal", reason=reason, target_bh=target_bh,
                       bbox_h=detection.bbox_height, height_delta=height_delta,
                       dead_zone=dead_zone_height, raw_forward=raw_forward,
                       bottom_backward=bottom_backward, final_forward=forward)

    return forward


def compute_velocity_command(
    detection: Optional[Detection],
    config: ControllerConfig,
    target_bbox_height_override: Optional[float] = None,
    search_direction: float = 1.0,
    last_detection: Optional[Detection] = None,
    search_active: bool = True,
    hold_velocity: Optional[VelocityBodyYawspeed] = None,
) -> VelocityBodyYawspeed:
    target_bh = target_bbox_height_override if target_bbox_height_override is not None else config.target_bbox_height

    # --- Search mode: no current detection ---
    if detection is None:
        if not search_active:
            return hold_velocity if hold_velocity is not None else VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
        # Spin toward last seen direction with damped forward correction
        search_forward = 0.0
        if last_detection is not None:
            raw = _calculate_forward_speed(last_detection, config, target_bh)
            search_forward = raw * config.search_vel_damp
            dead_zone = (config.dead_zone_height_percent / 100.0) * target_bh
            if raw > 0 and (target_bh - last_detection.bbox_height) > dead_zone:
                search_forward = max(search_forward, config.min_search_forward)
            search_forward = min(config.max_forward, search_forward)
        return VelocityBodyYawspeed(search_forward, 0.0, 0.0, search_direction * config.search_yawspeed_slow)

    # --- Tracking mode ---
    error_x_deg = (detection.center_x - 0.5) * config.hfov
    error_y_deg = (detection.center_y - 0.5) * config.vfov

    # Yaw: signed square-root response
    if abs(error_x_deg) < config.dead_zone_deg:
        yawspeed = 0.0
    else:
        yawspeed = math.copysign(config.kp_yaw * math.sqrt(abs(error_x_deg)), error_x_deg)
    yawspeed = max(-config.max_yawspeed, min(config.max_yawspeed, yawspeed))

    # Altitude
    down = 0.0
    if not config.fixed_altitude:
        down = 0.0 if abs(error_y_deg) < config.dead_zone_deg else config.kp_down * error_y_deg
        down = max(-config.max_down_speed, min(config.max_down_speed, down))

    forward = _calculate_forward_speed(detection, config, target_bh)

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

    def _grpc_address_from_connection(self):
        """Derive gRPC address from connection URL (host from connection, port from self.port)."""
        try:
            parsed = urlparse(self.connection_url)
            host = (parsed.hostname or "127.0.0.1").strip() or "127.0.0.1"
            if host == "0.0.0.0":
                host = "127.0.0.1"
            return f"grpc://{host}:{self.port}"
        except Exception:
            return f"grpc://127.0.0.1:{self.port}"

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

        cmd = [server_path, "-u", self.connection_url, "-p", str(self.port)]
        print(f"[drone] Starting detached mavsdk_server: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        
        # Give server a moment to start before returning
        time.sleep(0.5)
        return self._grpc_address_from_connection()

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
    circle_scale = circle_diameter_m / 10.0
    cx, cy, bh = initial_x, initial_y, config.target_bbox_height * 0.1

    try:
        while True:
            t = time.monotonic() - t0

            shared_state.update(Detection("person", 0.99, cx, cy, bh, time.monotonic()))
            detection, _ = shared_state.get_latest()

            cmd = compute_velocity_command(detection, config)
            await drone.offboard.set_velocity_body(cmd)

            # Apply world movement (the person walking)
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

        # PX4 requires setpoints to be streamed before offboard.start() (NO_SETPOINT_SET otherwise).
        setpoint_period_s = 0.05
        setpoint_duration_s = 1.5
        deadline = asyncio.get_event_loop().time() + setpoint_duration_s
        while asyncio.get_event_loop().time() < deadline and not shutdown.is_set():
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            await asyncio.sleep(setpoint_period_s)
        if shutdown.is_set():
            return
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
            if offboard_started and drone is not None and manage_takeoff_landing:
                try:
                    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                    await drone.offboard.stop()
                except OffboardError as e:
                    print(f"[drone] Offboard stop: {e._result.result}")
                except Exception as e:
                    _print_connection_error("[drone] Offboard stop", e)
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

def _effective_target_bbox_height(
    reference_altitude_m: float,
    target_bbox_height: float,
    current_altitude_m: float,
    min_altitude_m: float = 0.5,
    max_target: float = 0.9,
) -> float:
    """Target bbox height from altitude: at reference_altitude we want target_bbox_height; scales inversely with altitude."""
    alt = max(current_altitude_m, min_altitude_m)
    effective = (reference_altitude_m * target_bbox_height) / alt
    return min(effective, max_target)


async def _telemetry_altitude_task(drone, altitude_cache: dict, shutdown: asyncio.Event) -> None:
    """Background task: stream position and store relative altitude (m) in altitude_cache['m']."""
    try:
        async for position in drone.telemetry.position():
            if shutdown.is_set():
                return
            altitude_cache["m"] = position.relative_altitude_m
    except Exception:
        pass


async def live_control_loop(drone, shared_state, config, shutdown, altitude_cache: Optional[dict] = None, ui_state=None):
    """Control loop for Hailo modes.

    Reads detections from shared_state, computes velocity commands.
    If drone is None (hailo-dry-run), prints commands instead.
    When config.reference_altitude_m is set and altitude_cache is provided, target bbox height is scaled by altitude.
    If ui_state is provided, logs are also pushed to the web UI.
    """
    def _log(msg):
        if ui_state is not None:
            ui_state.push_log(msg)
        else:
            print(msg, flush=True)

    period = 1.0 / max(0.1, min(config.control_loop_hz, 5.0))
    last_detection_time = time.monotonic()
    search_direction = 1.0
    last_valid_detection: Optional[Detection] = None
    _prev_takeoff_alt = config.takeoff_altitude
    _goto_altitude = None
    _prev_cmd: Optional[VelocityBodyYawspeed] = None

    # Constants
    _GOTO_KP = 0.5
    _GOTO_MAX_SPEED = 1.5
    _GOTO_TOLERANCE = 0.3
    _LOG_INTERVAL = 1.0
    _FWD_LOG_INTERVAL = 0.5

    # Throttle timers
    _last_log_time = 0.0
    _last_fwd_log_time = 0.0

    try:
        while not shutdown.is_set():
            detection, _ = shared_state.get_latest()

            if detection is not None:
                age = time.monotonic() - detection.timestamp
                if age > config.detection_timeout_s:
                    detection = None
                else:
                    last_detection_time = time.monotonic()
                    last_valid_detection = detection
                    # Only update direction if detection is confident (away from center noise)
                    if abs(detection.center_x - 0.5) > 0.05:
                        search_direction = 1.0 if detection.center_x > 0.5 else -1.0

            # Check search timeout
            time_since_detection = time.monotonic() - last_detection_time
            if time_since_detection > config.search_timeout_s:
                _log(f"[drone] Search timeout ({config.search_timeout_s}s) exceeded - no person found. Landing...")
                shutdown.set()
                break

            # Detect takeoff_altitude changes and start goto
            if config.takeoff_altitude != _prev_takeoff_alt:
                _goto_altitude = config.takeoff_altitude
                _log(f"[drone] Altitude changed: going to {_goto_altitude:.1f}m")
                _prev_takeoff_alt = config.takeoff_altitude

            target_override = None
            if config.reference_altitude_m is not None and altitude_cache and altitude_cache.get("m") is not None:
                target_override = _effective_target_bbox_height(
                    config.reference_altitude_m,
                    config.target_bbox_height,
                    altitude_cache["m"],
                )
            cmd = compute_velocity_command(
                detection, config,
                target_bbox_height_override=target_override,
                search_direction=search_direction,
                last_detection=last_valid_detection,
                search_active=(time_since_detection >= config.search_enter_delay_s),
                hold_velocity=_prev_cmd,
            )

            # Override vertical velocity when going to a new altitude
            if _goto_altitude is not None and altitude_cache.get("m") is not None:
                alt_error = _goto_altitude - altitude_cache["m"]
                if abs(alt_error) < _GOTO_TOLERANCE:
                    _log(f"[drone] Reached target altitude {_goto_altitude:.1f}m")
                    _goto_altitude = None
                else:
                    down_speed = max(-_GOTO_MAX_SPEED, min(_GOTO_MAX_SPEED, -_GOTO_KP * alt_error))
                    cmd = VelocityBodyYawspeed(cmd.forward_m_s, cmd.right_m_s, down_speed, cmd.yawspeed_deg_s)

            now = time.monotonic()

            # Explain forward-velocity calculation in logs (throttled)
            if now - _last_fwd_log_time >= _FWD_LOG_INTERVAL and detection is not None:
                e = {}
                _calculate_forward_speed(detection, config,
                    target_override if target_override is not None else config.target_bbox_height,
                    explain=e)
                _log(f"[FWD] {e.get('reason', e.get('mode', '-'))} "
                     f"target={e.get('target_bh', 0):.2f} bbox={e.get('bbox_h', 0):.2f} "
                     f"err={e.get('height_delta', 0):+.3f} raw={e.get('raw_forward', 0):+.2f} "
                     f"bottom={e.get('bottom_backward', 0):+.2f} final={e.get('final_forward', 0):+.2f}")
                _last_fwd_log_time = now

            if drone is not None:
                await drone.offboard.set_velocity_body(cmd)
            else:
                tag = "TRACK" if detection is not None else "SEARCH"
                print(f"\r[{tag}] Yaw:{cmd.yawspeed_deg_s:+6.1f}\u00b0/s  "
                      f"Fwd:{cmd.forward_m_s:+5.2f}m/s  "
                      f"Down:{cmd.down_m_s:+5.2f}m/s", end="")
            _prev_cmd = cmd

            # Periodic status log to UI
            if now - _last_log_time >= _LOG_INTERVAL:
                _last_log_time = now
                alt_str = f" alt={altitude_cache['m']:.1f}m" if altitude_cache and altitude_cache.get("m") is not None else ""
                eff_str = f" eff_target={target_override:.2f}" if target_override is not None else ""
                if detection is not None:
                    _log(f"[TRACK] Yaw:{cmd.yawspeed_deg_s:+5.1f} Fwd:{cmd.forward_m_s:+5.2f} Down:{cmd.down_m_s:+5.2f}"
                         f" pos=({detection.center_x:.2f},{detection.center_y:.2f}) bbox_h={detection.bbox_height:.2f}"
                         f"{eff_str}{alt_str}")
                elif time_since_detection < config.search_enter_delay_s:
                    _log(f"[SEARCH-WAIT] entering search in {config.search_enter_delay_s - time_since_detection:.1f}s{alt_str}")
                else:
                    search_dir = "right" if cmd.yawspeed_deg_s > 0 else "left"
                    _log(f"[SEARCH] Spinning {search_dir} at {abs(cmd.yawspeed_deg_s):.1f} deg/s{alt_str}")

            await asyncio.sleep(period)
    except asyncio.CancelledError:
        if drone is not None:
            try:
                await drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            except Exception:
                pass
        raise


async def run_live_drone(args, shared_state, shutdown, shutdown_read_fd=None, takeoff_done=None, pipeline_quit_cb=None, config=None, ui_state=None):
    """Connect to drone and run live control loop with Hailo detections.

    If takeoff_done is a threading.Event, it is set after takeoff and offboard start,
    so the Hailo pipeline can wait before starting.
    If pipeline_quit_cb is set, it is called at shutdown start so the pipeline stops first.
    If config is provided, use it directly (allows live mutation from web UI).
    If ui_state is provided, logs are pushed to the web UI.
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

        # PX4 requires setpoints to be streamed before offboard.start() (NO_SETPOINT_SET otherwise).
        # Stream zero setpoint at ~20 Hz.
        setpoint_period_s = 0.05
        
        # Initial stream of setpoints
        for _ in range(int(2.0 / setpoint_period_s)):
            if shutdown.is_set():
                return
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            await asyncio.sleep(setpoint_period_s)

        # Try to start offboard mode, with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await drone.offboard.start()
                break
            except OffboardError as e:
                if attempt == max_retries - 1:
                    raise
                print(f"[drone] Failed to start offboard ({e}), retrying...")
                # Send more setpoints before retrying
                for _ in range(int(1.0 / setpoint_period_s)):
                    if shutdown.is_set():
                        return
                    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                    await asyncio.sleep(setpoint_period_s)
        offboard_started = True
        if manage_takeoff_landing:
            await asyncio.sleep(3)
        # When not manage_takeoff_landing we do not call offboard.stop() on exit (don't change mode)

        if takeoff_done is not None:
            takeoff_done.set()

        altitude_cache: dict = {}
        alt_task = asyncio.create_task(_telemetry_altitude_task(drone, altitude_cache, shutdown))

        task = asyncio.create_task(live_control_loop(drone, shared_state, config, shutdown, altitude_cache, ui_state=ui_state))
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
            alt_task.cancel()
            try:
                await alt_task
            except asyncio.CancelledError:
                pass
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
