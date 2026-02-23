"""Drone Follow — visual-servoing pipeline app for Hailo AI processors.

Uses the tiling pipeline for person detection and feeds detections into
a proportional controller that drives a drone via MAVSDK.
"""

from .drone_control import (
    Detection,
    SharedDetectionState,
    ControllerConfig,
    compute_velocity_command,
    apply_physics_step,
)
from .drone_follow import (
    app_callback,
    create_app,
)
from .web_server import SharedUIState, WebServer

__all__ = [
    "Detection",
    "SharedDetectionState",
    "ControllerConfig",
    "compute_velocity_command",
    "apply_physics_step",
    "app_callback",
    "create_app",
    "SharedUIState",
    "WebServer",
]
