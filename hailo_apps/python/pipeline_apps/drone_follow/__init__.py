"""Drone Follow — visual-servoing pipeline app for Hailo AI processors.

Uses the tiling pipeline for person detection and feeds detections into
a proportional controller that drives a drone via MAVSDK.
"""

from .drone_control import (
    Detection,
    SharedDetectionState,
    ControllerConfig,
    compute_velocity_command,
)

# Keep package import lightweight for tests/environments that don't have
# optional runtime deps (e.g. scipy/byte_tracker stack).
try:
    from .drone_follow import (
        app_callback,
        create_app,
    )
except Exception:  # pragma: no cover - optional runtime dependencies
    app_callback = None
    create_app = None

try:
    from .web_server import SharedUIState, WebServer
except Exception:  # pragma: no cover - optional runtime dependencies
    SharedUIState = None
    WebServer = None

__all__ = [
    "Detection",
    "SharedDetectionState",
    "ControllerConfig",
    "compute_velocity_command",
    "app_callback",
    "create_app",
    "SharedUIState",
    "WebServer",
]
