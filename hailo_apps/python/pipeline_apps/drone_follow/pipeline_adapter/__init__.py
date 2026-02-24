"""pipeline_adapter — Hailo/GStreamer pipeline adapters.

All Hailo and GStreamer imports are confined to this package.
Other modules receive detections as pure Detection objects via callbacks.
ByteTracker (multi-object tracker) also lives here.
"""

from .byte_tracker import ByteTracker

try:
    from .hailo_tiling import app_callback, create_app, add_pipeline_args
except Exception:  # pragma: no cover - optional runtime dependencies
    app_callback = None
    create_app = None
    add_pipeline_args = None

__all__ = [
    "ByteTracker",
    "app_callback",
    "create_app",
    "add_pipeline_args",
]
