"""
Web server for drone follow UI.

Serves a React UI with live MJPEG video stream and interactive bounding boxes.
The React app polls for detection metadata and allows click-to-follow.

Architecture:
    SharedUIState receives data from two GStreamer sources:
        1. app_callback (identity element) -> detection metadata
        2. appsink callback (JPEG branch) -> encoded JPEG frames

    WebServer (stdlib ThreadingHTTPServer) serves:
        GET  /api/video        -> MJPEG stream
        GET  /api/detections   -> JSON detection list
        POST /api/follow/<id>  -> set follow target
        POST /api/follow/clear -> clear target
        GET  /api/status       -> current status
        GET  /*                -> React static build (SPA fallback)
"""

import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional


class SharedUIState:
    """Thread-safe state shared between GStreamer callbacks and the web server."""

    def __init__(self):
        self._lock = threading.Lock()
        self._detections: list = []
        self._following_id: Optional[int] = None
        self._frame_jpeg: Optional[bytes] = None
        self._frame_event = threading.Event()
        self._logs: deque = deque(maxlen=200)
        self._log_counter: int = 0

    def update_detections(self, detections: list, following_id: Optional[int] = None):
        """Called from app_callback with detection metadata."""
        with self._lock:
            self._detections = detections
            self._following_id = following_id

    def update_frame(self, jpeg_bytes: bytes):
        """Called from appsink callback with pre-encoded JPEG bytes."""
        with self._lock:
            self._frame_jpeg = jpeg_bytes
        # Wake up all MJPEG waiters
        self._frame_event.set()
        self._frame_event.clear()

    def get_detections(self) -> dict:
        """Return current detections and following state."""
        with self._lock:
            return {
                "detections": list(self._detections),
                "following_id": self._following_id,
            }

    def push_log(self, message: str):
        """Append a log message (thread-safe). Also prints to console."""
        with self._lock:
            self._log_counter += 1
            self._logs.append({
                "id": self._log_counter,
                "ts": time.time(),
                "msg": message,
            })
        print(message, flush=True)

    def get_logs(self, since_id: int = 0) -> list:
        """Return log entries with id > since_id."""
        with self._lock:
            return [entry for entry in self._logs if entry["id"] > since_id]

    def wait_frame(self, timeout: float = 1.0) -> Optional[bytes]:
        """Block until a new frame is available (for MJPEG streaming)."""
        self._frame_event.wait(timeout=timeout)
        with self._lock:
            return self._frame_jpeg


class _WebHandler(BaseHTTPRequestHandler):
    """HTTP handler for the drone-follow UI."""

    # Class-level references set by WebServer before serving
    ui_state: SharedUIState = None
    target_state = None  # FollowTargetState
    shared_state = None  # SharedDetectionState
    controller_config = None  # ControllerConfig
    static_dir: str = None

    def log_message(self, format, *args):
        pass  # Suppress default stderr logging

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    # ------------------------------------------------------------------
    # GET routes
    # ------------------------------------------------------------------

    def do_GET(self):
        if self.path == "/api/video":
            self._handle_mjpeg()
        elif self.path == "/api/detections":
            self._handle_detections()
        elif self.path == "/api/status":
            self._handle_status()
        elif self.path == "/api/config":
            self._handle_get_config()
        elif self.path.startswith("/api/logs"):
            self._handle_logs()
        else:
            self._handle_static()

    def _handle_mjpeg(self):
        """Stream MJPEG: multipart/x-mixed-replace with JPEG frames."""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self._cors_headers()
        self.end_headers()

        try:
            while True:
                jpeg = self.ui_state.wait_frame(timeout=2.0)
                if jpeg is None:
                    continue
                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                    b"\r\n"
                )
                self.wfile.write(header)
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _handle_detections(self):
        data = self.ui_state.get_detections()
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _handle_status(self):
        status = {}
        if self.target_state is not None:
            status = self.target_state.get_status()
        if self.shared_state is not None:
            status["available_ids"] = list(self.shared_state.get_available_ids())
        body = json.dumps(status).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    # Exposed controller config fields and their types
    _CONFIG_FIELDS = {
        "kp_yaw": float,
        "kp_forward": float,
        "max_forward": float,
        "max_backward": float,
        "yaw_only": bool,
        "fixed_altitude": bool,
        "target_bbox_height": float,
        "dead_zone_height_percent": float,
        "takeoff_altitude": float,
    }

    def _handle_get_config(self):
        cfg = self.controller_config
        if cfg is None:
            self.send_error(404, "No controller config available")
            return
        data = {k: getattr(cfg, k) for k in self._CONFIG_FIELDS}
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _handle_logs(self):
        """Return log entries newer than ?since_id=N."""
        since_id = 0
        if "?" in self.path:
            query = self.path.split("?", 1)[1]
            for part in query.split("&"):
                if part.startswith("since_id="):
                    try:
                        since_id = int(part.split("=", 1)[1])
                    except ValueError:
                        pass
        logs = self.ui_state.get_logs(since_id)
        body = json.dumps({"logs": logs}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _handle_post_config(self):
        cfg = self.controller_config
        if cfg is None:
            self.send_error(404, "No controller config available")
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return
        for key, value in payload.items():
            if key not in self._CONFIG_FIELDS:
                continue
            expected = self._CONFIG_FIELDS[key]
            try:
                setattr(cfg, key, expected(value))
            except (TypeError, ValueError):
                continue
        data = {k: getattr(cfg, k) for k in self._CONFIG_FIELDS}
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _handle_static(self):
        """Serve React static build with SPA fallback to index.html."""
        if self.static_dir is None or not os.path.isdir(self.static_dir):
            self.send_error(404, "UI not built. Run: cd ui && npm install && npm run build")
            return

        # Map URL path to file path
        path = self.path.lstrip("/")
        if not path:
            path = "index.html"

        file_path = os.path.join(self.static_dir, path)

        # SPA fallback: if file doesn't exist, serve index.html
        if not os.path.isfile(file_path):
            file_path = os.path.join(self.static_dir, "index.html")

        if not os.path.isfile(file_path):
            self.send_error(404, "UI not built. Run: cd ui && npm install && npm run build")
            return

        # Determine content type
        content_type = self._guess_content_type(file_path)
        with open(file_path, "rb") as f:
            body = f.read()

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _guess_content_type(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        return {
            ".html": "text/html",
            ".js": "application/javascript",
            ".css": "text/css",
            ".json": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
            ".woff": "font/woff",
            ".woff2": "font/woff2",
        }.get(ext, "application/octet-stream")

    # ------------------------------------------------------------------
    # POST routes
    # ------------------------------------------------------------------

    def do_POST(self):
        if self.path == "/api/config":
            self._handle_post_config()
        elif self.path == "/api/follow/clear":
            self._handle_follow_clear()
        elif self.path.startswith("/api/follow/"):
            self._handle_follow()
        else:
            self.send_error(404, "Not Found")

    def _handle_follow(self):
        try:
            id_str = self.path.split("/api/follow/")[1]
            detection_id = int(id_str)
        except (ValueError, IndexError):
            self.send_error(400, "Invalid detection ID")
            return

        if self.shared_state is not None:
            available = self.shared_state.get_available_ids()
            if detection_id not in available:
                body = json.dumps({
                    "status": "error",
                    "message": f"ID {detection_id} not in frame",
                    "available_ids": list(available),
                }).encode()
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self._cors_headers()
                self.end_headers()
                self.wfile.write(body)
                return

        if self.target_state is not None:
            self.target_state.set_target(detection_id)

        body = json.dumps({"status": "success", "following_id": detection_id}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)
        print(f"[ui-server] Now following ID: {detection_id}")

    def _handle_follow_clear(self):
        if self.target_state is not None:
            self.target_state.set_target(None)

        body = json.dumps({
            "status": "success",
            "following_id": None,
            "message": "Cleared target, following largest person",
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)
        print("[ui-server] Cleared target, following largest person")


class WebServer:
    """Web server for drone-follow UI. Runs in a daemon thread."""

    def __init__(self, ui_state, target_state=None, shared_state=None,
                 controller_config=None, host="0.0.0.0", port=5001, static_dir=None):
        self.ui_state = ui_state
        self.target_state = target_state
        self.shared_state = shared_state
        self.controller_config = controller_config
        self.host = host
        self.port = port
        self.static_dir = static_dir
        self.server = None
        self.thread = None

    def start(self):
        _WebHandler.ui_state = self.ui_state
        _WebHandler.target_state = self.target_state
        _WebHandler.shared_state = self.shared_state
        _WebHandler.controller_config = self.controller_config
        _WebHandler.static_dir = self.static_dir

        self.server = ThreadingHTTPServer((self.host, self.port), _WebHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f"[ui-server] Started on http://{self.host}:{self.port}")

    def stop(self):
        if self.server:
            self.server.shutdown()
            print("[ui-server] Stopped")
