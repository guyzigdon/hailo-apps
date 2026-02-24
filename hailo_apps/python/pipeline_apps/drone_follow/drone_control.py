import asyncio
import logging
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

LOGGER = logging.getLogger("drone_follow.control")

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
    target_distance_m: Optional[float] = 8.0   # desired horizontal distance; overrides target_bbox_height when set
    person_height_m: float = 1.7               # assumed person height for distance calculation
    kp_forward: float = 3.0
    kp_backward: float = 5.0
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
    bottom_y_threshold: float = 0.7
    # Search mode
    search_yawspeed_slow: float = 10.0  # yaw speed during search (slower than tracking)
    search_vel_damp: float = 0.3        # dampening factor for forward/backward speed during search
    min_search_forward: float = 0.2     # minimum forward speed in search when last bbox was too small
    # Yaw smoothing
    smooth_yaw: bool = True             # enable low-pass smoothing on yaw command
    yaw_alpha: float = 0.3              # yaw EMA factor (0=very smooth, 1=no smoothing)
    # Forward smoothing: estimate person velocity and smooth commands
    smooth_forward: bool = True         # enable forward velocity smoothing
    forward_alpha: float = 0.1          # EMA smoothing factor (0=ignore new, 1=no smoothing)
    kd_forward: float = 2.0            # derivative gain: anticipate person movement

    takeoff_altitude: float = 3.0
    log_verbosity: str = "normal"  # quiet | normal | debug

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Raise ValueError if the configuration is internally inconsistent."""
        if self.target_distance_m is not None and not self.fixed_altitude:
            raise ValueError(
                "target_distance_m requires fixed_altitude=True; "
                "with variable altitude, use target_bbox_height instead"
            )

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

        # --target-distance and --target-bbox-height are mutually exclusive.
        # Argparse defaults are None so we can detect user-explicit values.
        _user_distance = getattr(args, "target_distance", None)
        _user_bbox = getattr(args, "target_bbox_height", None)
        if _user_distance is not None and _user_bbox is not None:
            raise ValueError(
                "--target-distance and --target-bbox-height are mutually exclusive"
            )
        # If user explicitly chose bbox-height mode, disable distance mode.
        if _user_bbox is not None and _user_distance is None:
            target_distance_val = None
        else:
            target_distance_val = _arg("target_distance", "target_distance_m",
                                       default=defaults.target_distance_m)

        return cls(
            hfov=_arg("hfov", default=defaults.hfov),
            vfov=_arg("vfov", default=defaults.vfov),
            kp_yaw=_arg("kp_yaw", "yaw_gain", default=defaults.kp_yaw),
            kp_down=_arg("kp_down", "pitch_gain", default=defaults.kp_down),
            kp_forward=float(_arg("kp_forward", "forward_gain", default=defaults.kp_forward)),
            kp_backward=_arg("kp_backward", "backward_gain", default=defaults.kp_backward),
            target_bbox_height=_arg("target_bbox_height", default=defaults.target_bbox_height),
            target_distance_m=target_distance_val,
            person_height_m=_arg("person_height", "person_height_m", default=defaults.person_height_m),
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
            smooth_yaw=_arg("smooth_yaw", default=defaults.smooth_yaw),
            yaw_alpha=_arg("yaw_alpha", default=defaults.yaw_alpha),
            smooth_forward=_arg("smooth_forward", default=defaults.smooth_forward),
            forward_alpha=_arg("forward_alpha", default=defaults.forward_alpha),
            kd_forward=_arg("kd_forward", default=defaults.kd_forward),
            takeoff_altitude=_arg("takeoff_altitude", default=defaults.takeoff_altitude),
            log_verbosity=_arg("log_verbosity", default=defaults.log_verbosity),
        )

# ---------------------------------------------------------------------------
# Velocity Command API – clamps maximums & low-pass filters yaw
# ---------------------------------------------------------------------------

class VelocityCommandAPI:
    """Wrapper around drone.offboard.set_velocity_body that enforces max
    velocity limits and applies an exponential low-pass filter on the yaw axis.

    Usage:
        api = VelocityCommandAPI(drone, config)
        await api.send(cmd)          # clamped + filtered
        await api.send_zero()        # immediate zero (bypasses filter)
    """

    def __init__(self, drone, config: ControllerConfig, yaw_alpha: Optional[float] = None):
        """
        Args:
            drone: MAVSDK System (or None for print-only mode).
            config: ControllerConfig used to read max_* limits.
            yaw_alpha: Low-pass filter coefficient for yaw (0..1).
                       Smaller = more smoothing, larger = faster response.
        """
        self._drone = drone
        self._config = config
        self._yaw_alpha = config.yaw_alpha if yaw_alpha is None else yaw_alpha
        self._filtered_yaw: float = 0.0

    async def send(self, cmd: VelocityBodyYawspeed) -> VelocityBodyYawspeed:
        """Clamp velocity components, apply yaw low-pass filter, and send.

        Returns the command that was actually sent (after clamping/filtering).
        """
        # Clamp each axis to configured maximums
        forward = max(-self._config.max_backward, min(self._config.max_forward, cmd.forward_m_s))
        right = max(-1.0, min(1.0, cmd.right_m_s))  # lateral not heavily used; keep bounded
        down = max(-self._config.max_down_speed, min(self._config.max_down_speed, cmd.down_m_s))
        yaw_raw = max(-self._config.max_yawspeed, min(self._config.max_yawspeed, cmd.yawspeed_deg_s))

        if self._config.smooth_yaw:
            # Low-pass filter on yaw: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
            self._filtered_yaw = (self._yaw_alpha * yaw_raw
                                  + (1.0 - self._yaw_alpha) * self._filtered_yaw)
            yaw_out = self._filtered_yaw
        else:
            self._filtered_yaw = yaw_raw
            yaw_out = yaw_raw

        clamped = VelocityBodyYawspeed(forward, right, down, yaw_out)

        if self._drone is not None:
            await self._drone.offboard.set_velocity_body(clamped)

        return clamped

    async def send_zero(self) -> None:
        """Send an immediate zero-velocity command and reset the yaw filter."""
        self._filtered_yaw = 0.0
        zero = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
        if self._drone is not None:
            await self._drone.offboard.set_velocity_body(zero)

    async def send_raw(self, cmd: VelocityBodyYawspeed) -> None:
        """Send a command without clamping or filtering (for pre-offboard setpoints)."""
        if self._drone is not None:
            await self._drone.offboard.set_velocity_body(cmd)

    def reset_filter(self) -> None:
        """Reset the yaw low-pass filter state."""
        self._filtered_yaw = 0.0


def _distance_to_bbox_height(
    altitude_m: float,
    horizontal_distance_m: float,
    vfov_deg: float,
    person_height_m: float = 1.7,
) -> float:
    """Convert desired horizontal distance to expected normalized bbox height (0-1).

    Uses perspective projection: at a given altitude and horizontal distance,
    compute what fraction of the vertical FOV an average person occupies.
    """
    slant_range = math.sqrt(horizontal_distance_m ** 2 + altitude_m ** 2)
    angular_height = 2.0 * math.atan(person_height_m / (2.0 * slant_range))
    vfov_rad = math.radians(vfov_deg)
    return angular_height / vfov_rad


def _calculate_forward_speed(
    detection: Detection,
    config: ControllerConfig,
    target_bh: float,
) -> float:
    """Calculate forward/backward speed based on bbox height and bottom-of-frame position."""
    if config.yaw_only or config.kp_forward == 0:
        return 0.0

    if detection.bbox_height > config.max_bbox_height_safety:
        return -config.max_backward

    height_delta = target_bh - detection.bbox_height
    dead_zone_height = (config.dead_zone_height_percent / 100.0) * target_bh

    if abs(height_delta) < dead_zone_height:
        forward = 0.0
    elif height_delta > 0:
        forward = config.kp_forward * math.sqrt(height_delta)
    else:
        forward = -config.kp_backward * math.sqrt(-height_delta)

    # Bottom-of-frame backward: bbox bottom edge past threshold means drone is above and too close
    max_y = detection.center_y + detection.bbox_height / 2
    if max_y > config.bottom_y_threshold:
        y_excess = max_y - config.bottom_y_threshold
        bottom_backward = config.kp_backward * math.sqrt(y_excess)
        forward = min(forward, -bottom_backward)

    return forward


class ForwardSmoother:
    """Estimates person approach/recede velocity and smooths forward commands.

    Tracks bbox_height over time to compute d(bbox_height)/dt, then uses that
    as a derivative feed-forward term. Also applies EMA to the final forward
    velocity to avoid big jumps.
    """

    def __init__(self):
        self._smoothed_forward: float = 0.0
        self._prev_bbox_h: Optional[float] = None
        self._prev_time: Optional[float] = None
        self._bbox_h_rate: float = 0.0  # EMA of d(bbox_height)/dt
        self._rate_alpha: float = 0.3   # smoothing for rate estimation

    def update(self, detection: Optional[Detection], raw_forward: float,
               config: ControllerConfig) -> float:
        """Return smoothed forward velocity."""
        now = time.monotonic()

        # Update bbox height rate estimate
        if detection is not None and self._prev_bbox_h is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 0.01:
                instant_rate = (detection.bbox_height - self._prev_bbox_h) / dt
                self._bbox_h_rate = (self._rate_alpha * instant_rate
                                     + (1.0 - self._rate_alpha) * self._bbox_h_rate)
        if detection is not None:
            self._prev_bbox_h = detection.bbox_height
            self._prev_time = now
        else:
            self._bbox_h_rate *= 0.9

        # Derivative feed-forward: positive rate means person is getting closer (bbox growing)
        # -> we should move backward (negative forward). Negative rate -> move forward.
        derivative_term = -config.kd_forward * self._bbox_h_rate

        target_forward = raw_forward + derivative_term

        # Clamp before smoothing
        target_forward = max(-config.max_backward, min(config.max_forward, target_forward))

        # EMA smoothing
        alpha = config.forward_alpha
        self._smoothed_forward = alpha * target_forward + (1.0 - alpha) * self._smoothed_forward

        return self._smoothed_forward

    def reset(self):
        self._smoothed_forward = 0.0
        self._prev_bbox_h = None
        self._prev_time = None
        self._bbox_h_rate = 0.0


def compute_velocity_command(
    detection: Optional[Detection],
    config: ControllerConfig,
    target_bbox_height_override: Optional[float] = None,
    last_detection: Optional[Detection] = None,
    search_active: bool = True,
    hold_velocity: Optional[VelocityBodyYawspeed] = None,
) -> VelocityBodyYawspeed:
    target_bh = target_bbox_height_override if target_bbox_height_override is not None else config.target_bbox_height

    # --- Search mode: no current detection ---
    if detection is None:
        if not search_active:
            return hold_velocity if hold_velocity is not None else VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
        # Derive search direction from last seen position.
        search_direction = 1.0
        if last_detection is not None:
            search_direction = 1.0 if last_detection.center_x > 0.5 else -1.0
        # Spin toward last seen direction with damped forward correction
        search_forward = 0.0
        if last_detection is not None:
            raw = _calculate_forward_speed(last_detection, config, target_bh)
            search_forward = raw * config.search_vel_damp
            search_forward = max(search_forward, 0)
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
    if not config.fixed_altitude and not config.yaw_only:
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
            LOGGER.warning("[drone] mavsdk_server not found at %s, using default System() behavior", server_path)
            return self.connection_url # Fallback to default behavior

        cmd = [server_path, "-u", self.connection_url, "-p", str(self.port)]
        LOGGER.info("[drone] Starting detached mavsdk_server: %s", " ".join(cmd))

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


def _exit_if_not_offboard(reason: str) -> None:
    """Exit the process immediately. Use when --no-takeoff-landing and drone must be OFFBOARD."""
    LOGGER.error("[drone] %s", reason)
    sys.stderr.flush()
    os._exit(1)


async def _wait_for_offboard_mode(drone: System, shutdown: asyncio.Event) -> None:
    """Block until the drone enters OFFBOARD mode, streaming zero setpoints as keep-alive.

    In --no-takeoff-landing mode the user switches to OFFBOARD externally (e.g. via
    a GCS).  We stream zero-velocity setpoints so PX4 accepts the transition, and
    wait patiently instead of killing the process.
    """
    zero = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
    setpoint_period = 0.05

    async def _stream_setpoints():
        while not shutdown.is_set():
            try:
                await drone.offboard.set_velocity_body(zero)
            except Exception:
                pass
            await asyncio.sleep(setpoint_period)

    async def _watch_for_offboard():
        async for mode in drone.telemetry.flight_mode():
            if shutdown.is_set():
                return
            if mode == FlightMode.OFFBOARD:
                LOGGER.info("[drone] OFFBOARD mode detected.")
                return
            LOGGER.info("[drone] Current mode: %s -- waiting for OFFBOARD...", mode.name)

    setpoint_task = asyncio.create_task(_stream_setpoints())
    watch_task = asyncio.create_task(_watch_for_offboard())
    shutdown_task = asyncio.create_task(shutdown.wait())
    try:
        LOGGER.info("[drone] Waiting for OFFBOARD mode (switch via GCS)...")
        done, pending = await asyncio.wait(
            [watch_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            await _cancel_task(t)
    finally:
        await _cancel_task(setpoint_task)


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
        LOGGER.warning("%s: connection lost (sim or MAVSDK backend closed).", prefix)
        if hint:
            LOGGER.warning("[drone] Tip: press Ctrl+C once and wait for landing before closing the sim.")
    else:
        LOGGER.warning("%s: %s", prefix, e)


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
    config: ControllerConfig,
    current_altitude_m: float,
    min_altitude_m: float = 0.5,
    max_target: float = 0.9,
) -> float:
    """Compute effective target bbox height for the current altitude.

    If target_distance_m is set, use perspective geometry to derive bbox height
    from altitude + horizontal distance. Otherwise, scale target_bbox_height
    inversely with altitude relative to reference_altitude_m.
    """
    alt = max(current_altitude_m, min_altitude_m)
    if config.target_distance_m is not None and config.target_distance_m > 0:
        return min(_distance_to_bbox_height(
            alt, config.target_distance_m, config.vfov, config.person_height_m,
        ), max_target)
    effective = (config.reference_altitude_m * config.target_bbox_height) / alt
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
    When config.reference_altitude_m is set and altitude_cache is provided, target bbox height is scaled by altitude.
    If ui_state is provided, logs are also pushed to the web UI.
    """
    vel_api = VelocityCommandAPI(drone, config)

    def _log(msg: str, level: int = logging.INFO):
        if not LOGGER.isEnabledFor(level):
            return
        LOGGER.log(level, msg)
        if ui_state is not None:
            ui_state.push_log(msg)

    period = 1.0 / max(0.1, config.control_loop_hz)
    last_detection_time = time.monotonic()
    last_valid_detection: Optional[Detection] = None
    _prev_takeoff_alt = config.takeoff_altitude
    _goto_altitude = None
    _prev_cmd: Optional[VelocityBodyYawspeed] = None
    _fwd_smoother = ForwardSmoother()

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
            now = time.monotonic()
            detection, _ = shared_state.get_latest()

            if detection is not None:
                age = now - detection.timestamp
                if age > config.detection_timeout_s:
                    detection = None
                else:
                    last_detection_time = now
                    last_valid_detection = detection

            # Check search timeout
            time_since_detection = now - last_detection_time
            if time_since_detection > config.search_timeout_s:
                _log(f"[drone] Search timeout ({config.search_timeout_s}s) exceeded - no person found. Landing...", level=logging.WARNING)
                shutdown.set()
                break

            # Detect takeoff_altitude changes and start goto
            if config.takeoff_altitude != _prev_takeoff_alt:
                _goto_altitude = config.takeoff_altitude
                _log(f"[drone] Altitude changed: going to {_goto_altitude:.1f}m", level=logging.INFO)
                _prev_takeoff_alt = config.takeoff_altitude

            target_override = None
            if altitude_cache and altitude_cache.get("m") is not None:
                target_override = _effective_target_bbox_height(config, altitude_cache["m"])
            cmd = compute_velocity_command(
                detection, config,
                target_bbox_height_override=target_override,
                last_detection=last_valid_detection,
                search_active=(time_since_detection >= config.search_enter_delay_s),
                hold_velocity=_prev_cmd,
            )

            if config.smooth_forward and not config.yaw_only:
                smoothed_fwd = _fwd_smoother.update(detection, cmd.forward_m_s, config)
                cmd = VelocityBodyYawspeed(smoothed_fwd, cmd.right_m_s, cmd.down_m_s, cmd.yawspeed_deg_s)

            # Override vertical velocity when going to a new altitude
            if _goto_altitude is not None and altitude_cache.get("m") is not None:
                alt_error = _goto_altitude - altitude_cache["m"]
                if abs(alt_error) < _GOTO_TOLERANCE:
                    _log(f"[drone] Reached target altitude {_goto_altitude:.1f}m", level=logging.INFO)
                    _goto_altitude = None
                else:
                    down_speed = max(-_GOTO_MAX_SPEED, min(_GOTO_MAX_SPEED, -_GOTO_KP * alt_error))
                    cmd = VelocityBodyYawspeed(cmd.forward_m_s, cmd.right_m_s, down_speed, cmd.yawspeed_deg_s)

            # Forward-velocity log (throttled)
            if now - _last_fwd_log_time >= _FWD_LOG_INTERVAL and detection is not None:
                target_bh = target_override if target_override is not None else config.target_bbox_height
                _log(f"[FWD] target={target_bh:.2f} bbox={detection.bbox_height:.2f} "
                     f"final={cmd.forward_m_s:+.2f}", level=logging.DEBUG)
                _last_fwd_log_time = now

            cmd = await vel_api.send(cmd)
            if drone is None:
                tag = "TRACK" if detection is not None else "SEARCH"
                _log(f"[{tag}] Yaw:{cmd.yawspeed_deg_s:+6.1f}\u00b0/s  "
                     f"Fwd:{cmd.forward_m_s:+5.2f}m/s  "
                     f"Down:{cmd.down_m_s:+5.2f}m/s", level=logging.INFO)
            if ui_state is not None:
                mode = "TRACK" if detection is not None else ("SEARCH" if time_since_detection >= config.search_enter_delay_s else "SEARCH-WAIT")
                ui_state.update_velocity(cmd.forward_m_s, cmd.down_m_s, cmd.yawspeed_deg_s, mode)
            _prev_cmd = cmd

            # Periodic status log to UI
            if now - _last_log_time >= _LOG_INTERVAL:
                _last_log_time = now
                alt_str = f" alt={altitude_cache['m']:.1f}m" if altitude_cache and altitude_cache.get("m") is not None else ""
                eff_str = f" eff_target={target_override:.2f}" if target_override is not None else ""
                if detection is not None:
                    _log(f"[TRACK] Yaw:{cmd.yawspeed_deg_s:+5.1f} Fwd:{cmd.forward_m_s:+5.2f} Down:{cmd.down_m_s:+5.2f}"
                         f" pos=({detection.center_x:.2f},{detection.center_y:.2f}) bbox_h={detection.bbox_height:.2f}"
                         f"{eff_str}{alt_str}", level=logging.INFO)
                elif time_since_detection < config.search_enter_delay_s:
                    _log(f"[SEARCH-WAIT] entering search in {config.search_enter_delay_s - time_since_detection:.1f}s{alt_str}", level=logging.INFO)
                else:
                    search_dir = "right" if cmd.yawspeed_deg_s > 0 else "left"
                    _log(f"[SEARCH] Spinning {search_dir} at {abs(cmd.yawspeed_deg_s):.1f} deg/s{alt_str}", level=logging.INFO)

            await asyncio.sleep(period)
    except asyncio.CancelledError:
        try:
            await vel_api.send_zero()
        except Exception:
            pass
        raise


async def _start_offboard(drone, vel_api: VelocityCommandAPI, shutdown: asyncio.Event) -> None:
    """Stream zero setpoints then start offboard mode with retries.

    PX4 requires setpoints to be streamed before offboard.start()
    (NO_SETPOINT_SET otherwise). Streams at ~20 Hz for 2 s, then
    retries offboard.start() up to 3 times.
    """
    zero = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
    setpoint_period_s = 0.05

    for _ in range(int(2.0 / setpoint_period_s)):
        if shutdown.is_set():
            return
        await vel_api.send_raw(zero)
        await asyncio.sleep(setpoint_period_s)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            await drone.offboard.start()
            return
        except OffboardError as e:
            if attempt == max_retries - 1:
                raise
            LOGGER.warning("[drone] Failed to start offboard (%s), retrying...", e)
            for _ in range(int(1.0 / setpoint_period_s)):
                if shutdown.is_set():
                    return
                await vel_api.send_raw(zero)
                await asyncio.sleep(setpoint_period_s)


async def _land_safely(drone, vel_api: VelocityCommandAPI) -> None:
    """Stop offboard mode and land, ignoring SIGINT during the sequence."""
    try:
        await vel_api.send_zero()
        await drone.offboard.stop()
    except Exception as e:
        _print_connection_error("[drone] Offboard stop", e)

    LOGGER.warning("[drone] Landing safely - please wait (ignoring further Ctrl+C until done)...")
    try:
        _ignore_sigint_during_landing(ignore=True)
        LOGGER.info("[drone] Landing...")
        try:
            await drone.action.land()
            await asyncio.sleep(8)
        except Exception as e:
            _print_connection_error("[drone] Land", e)
    finally:
        _ignore_sigint_during_landing(ignore=False)


async def _cancel_task(task: asyncio.Task) -> None:
    """Cancel an asyncio task and suppress CancelledError."""
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


async def run_live_drone(args, shared_state, shutdown, shutdown_read_fd=None,
                         takeoff_done=None, pipeline_quit_cb=None, config=None, ui_state=None):
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

    manage_takeoff_landing = not getattr(args, 'no_takeoff_landing', False)

    with DetachedMavsdkServer(args.connection) as connection_url:
        drone = System()
        await drone.connect(system_address=connection_url)

        if manage_takeoff_landing:
            LOGGER.info("[drone] Connecting and taking off...")
        else:
            LOGGER.info("[drone] Connecting (drone must already be in OFFBOARD)...")
        async for state in drone.core.connection_state():
            if state.is_connected:
                break

        if manage_takeoff_landing:
            await drone.action.set_takeoff_altitude(args.takeoff_altitude)
            await drone.action.arm()
            await drone.action.takeoff()
            await asyncio.sleep(15)
        else:
            await _wait_for_offboard_mode(drone, shutdown)
            if shutdown.is_set():
                return

        vel_api = VelocityCommandAPI(drone, config)
        await _start_offboard(drone, vel_api, shutdown)
        if shutdown.is_set():
            return

        if manage_takeoff_landing:
            await asyncio.sleep(3)

        if takeoff_done is not None:
            takeoff_done.set()

        altitude_cache: dict = {}
        alt_task = asyncio.create_task(_telemetry_altitude_task(drone, altitude_cache, shutdown))
        control_task = asyncio.create_task(
            live_control_loop(drone, shared_state, config, shutdown, altitude_cache, ui_state=ui_state))
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
                await _cancel_task(t)
            if shutdown.is_set():
                if manage_takeoff_landing:
                    LOGGER.warning("[drone] Shutdown requested, landing...")
                else:
                    LOGGER.warning("[drone] Shutdown requested, stopping control loop...")
        except asyncio.CancelledError:
            if manage_takeoff_landing:
                LOGGER.warning("[drone] Shutdown requested, landing...")
            else:
                LOGGER.warning("[drone] Shutdown requested, stopping control loop...")
        finally:
            await _cancel_task(alt_task)
            if watch_task is not None:
                await _cancel_task(watch_task)
            if manage_takeoff_landing:
                await _land_safely(drone, vel_api)
            await _cancel_task(control_task)
            if pipeline_quit_cb is not None:
                try:
                    pipeline_quit_cb()
                except Exception:
                    pass
        LOGGER.info("[drone] Done.")
