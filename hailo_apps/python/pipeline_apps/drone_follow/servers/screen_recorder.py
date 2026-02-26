"""Screen recorder for full desktop capture (X11 via ffmpeg, Wayland via wf-recorder)."""

import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime

LOGGER = logging.getLogger("drone_follow.screen_recorder")


class ScreenRecorder:
    """Thread-safe screen recorder (X11 via ffmpeg x11grab, Wayland via wf-recorder)."""

    def __init__(self, output_dir: str = "recordings"):
        self._output_dir = output_dir
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._file_path: str | None = None
        self._start_time: float | None = None
        self._error: str | None = None
        self._is_wayland: bool = False

    def _build_command(self, file_path: str) -> list[str] | None:
        """Build the recording command based on session type. Returns None on error."""
        session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
        self._is_wayland = session_type == "wayland"

        if self._is_wayland:
            if not shutil.which("wf-recorder"):
                self._error = (
                    "wf-recorder not found. Install it with: "
                    "sudo apt install wf-recorder"
                )
                return None
            return [
                "wf-recorder",
                "-f", file_path,
                "-c", "libx264",
                "-p", "preset=ultrafast",
                "-p", "crf=23",
            ]
        else:
            if not shutil.which("ffmpeg"):
                self._error = "ffmpeg not found in PATH"
                return None
            display = os.environ.get("DISPLAY", ":0")
            return [
                "ffmpeg", "-y",
                "-f", "x11grab",
                "-framerate", "30",
                "-i", display,
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                file_path,
            ]

    def start(self) -> dict:
        """Start recording the screen. Returns status dict."""
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return {"recording": True, "error": "Already recording"}

            os.makedirs(self._output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self._file_path = os.path.join(self._output_dir, f"rec_{timestamp}.mp4")
            self._error = None

            cmd = self._build_command(self._file_path)
            if cmd is None:
                return {"recording": False, "error": self._error}

            try:
                self._process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except OSError as e:
                self._error = f"Failed to start recorder: {e}"
                self._process = None
                return {"recording": False, "error": self._error}

            self._start_time = time.time()

        # Wait briefly and check for immediate failure (outside lock)
        time.sleep(0.5)
        with self._lock:
            if self._process is not None and self._process.poll() is not None:
                stderr = self._process.stderr.read().decode(errors="replace") if self._process.stderr else ""
                self._error = f"Recorder exited immediately: {stderr[-200:]}"
                self._process = None
                self._start_time = None
                return {"recording": False, "error": self._error}

            LOGGER.info("Recording started: %s", self._file_path)
            return {"recording": True, "file": self._file_path}

    def stop(self) -> dict:
        """Stop the current recording. Returns status dict with file info."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._process = None
                self._start_time = None
                return {"recording": False, "error": "Not recording"}

            proc = self._process
            file_path = self._file_path
            self._process = None
            self._start_time = None

        # Send SIGINT (same as Ctrl-C) so ffmpeg finalizes the file
        try:
            proc.send_signal(signal.SIGINT)
        except OSError:
            pass

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            LOGGER.warning("Recorder did not stop gracefully, sending SIGKILL")
            proc.kill()
            proc.wait(timeout=2)

        size = 0
        if file_path and os.path.isfile(file_path):
            size = os.path.getsize(file_path)

        LOGGER.info("Recording stopped: %s (%d bytes)", file_path, size)
        return {"recording": False, "file": file_path, "size": size}

    def status(self) -> dict:
        """Return current recording status."""
        with self._lock:
            if self._process is None:
                return {
                    "recording": False,
                    "file": self._file_path,
                    "elapsed_seconds": 0,
                    "error": self._error,
                    "available": True,
                }

            # Check if ffmpeg died mid-recording
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode(errors="replace") if self._process.stderr else ""
                self._error = f"Recorder exited unexpectedly: {stderr[-200:]}"
                self._process = None
                elapsed = time.time() - self._start_time if self._start_time else 0
                self._start_time = None
                return {
                    "recording": False,
                    "file": self._file_path,
                    "elapsed_seconds": round(elapsed, 1),
                    "error": self._error,
                    "available": True,
                }

            elapsed = time.time() - self._start_time if self._start_time else 0
            return {
                "recording": True,
                "file": self._file_path,
                "elapsed_seconds": round(elapsed, 1),
                "error": None,
                "available": True,
            }
