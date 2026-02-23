#!/usr/bin/env python3
"""
HTTP server for drone follow application.

Provides a REST API to control which tracked person the drone should follow.
The server is always available and provides status information.
Target selection by ID requires tracking to be enabled (--enable-tracking flag).

Usage:
    The server starts automatically in all modes.
    
    POST /follow/<detection_id>
        Start following the person with the specified tracking ID.
        Requires --enable-tracking to be enabled.
        Returns: {"status": "success", "following_id": <id>}
    
    GET /status
        Get current tracking status.
        Returns: {"following_id": <id or null>, "last_seen": <timestamp or null>}

Example:
    curl -X POST http://localhost:8080/follow/42
    curl http://localhost:8080/status
"""

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from typing import Optional


class FollowTargetState:
    """Thread-safe state for which detection ID to follow."""
    def __init__(self):
        self._lock = threading.Lock()
        self._target_id: Optional[int] = None
        self._last_seen: Optional[float] = None

    def set_target(self, detection_id: Optional[int]):
        """Set the target detection ID to follow."""
        with self._lock:
            self._target_id = detection_id
            if detection_id is not None:
                self._last_seen = time.monotonic()

    def get_target(self) -> Optional[int]:
        """Get the current target detection ID."""
        with self._lock:
            return self._target_id

    def update_last_seen(self):
        """Update the last seen timestamp for the current target."""
        with self._lock:
            if self._target_id is not None:
                self._last_seen = time.monotonic()

    def get_status(self):
        """Get current status as a dict."""
        with self._lock:
            return {
                "following_id": self._target_id,
                "last_seen": self._last_seen
            }


class FollowServerHandler(BaseHTTPRequestHandler):
    """HTTP request handler for follow server."""
    
    # Class variables to hold shared state
    target_state: FollowTargetState = None
    shared_state: 'SharedDetectionState' = None

    def log_message(self, format, *args):
        """Override to customize logging."""
        print(f"[follow-server] {format % args}")

    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/follow/clear" or self.path == "/follow/":
            # Clear target, return to following biggest person
            self.target_state.set_target(None)
            
            response = {
                "status": "success",
                "following_id": None,
                "message": "Cleared target, now following largest person"
            }
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
            print(f"[follow-server] Cleared target, now following largest person")
        elif self.path.startswith("/follow/"):
            try:
                detection_id_str = self.path.split("/follow/")[1]
                detection_id = int(detection_id_str)
                
                # Check if the detection ID is currently in the frame
                if self.shared_state is not None:
                    available_ids = self.shared_state.get_available_ids()
                    if detection_id not in available_ids:
                        self.send_response(404)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        error_response = {
                            "status": "error",
                            "message": f"Detection ID {detection_id} not found in current frame",
                            "available_ids": list(available_ids)
                        }
                        self.wfile.write(json.dumps(error_response).encode())
                        print(f"[follow-server] Detection ID {detection_id} not found. Available: {available_ids}")
                        return
                
                self.target_state.set_target(detection_id)
                
                response = {
                    "status": "success",
                    "following_id": detection_id
                }
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
                print(f"[follow-server] Now following detection ID: {detection_id}")
            except (ValueError, IndexError) as e:
                self.send_error(400, f"Invalid detection ID: {e}")
        elif self.path == "/follow/clear" or self.path == "/follow/":
            # Clear target, return to following biggest person
            self.target_state.set_target(None)
            
            response = {
                "status": "success",
                "following_id": None,
                "message": "Cleared target, now following largest person"
            }
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
            print(f"[follow-server] Cleared target, now following largest person")
        else:
            self.send_error(404, "Not Found")

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/status":
            status = self.target_state.get_status()
            
            # Add available IDs if shared_state is available
            if self.shared_state is not None:
                status["available_ids"] = list(self.shared_state.get_available_ids())
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status).encode())
        else:
            self.send_error(404, "Not Found")


class FollowServer:
    """HTTP server for follow target selection."""
    
    def __init__(self, target_state: FollowTargetState, shared_state: 'SharedDetectionState' = None, host: str = "0.0.0.0", port: int = 8080):
        self.target_state = target_state
        self.shared_state = shared_state
        self.host = host
        self.port = port
        self.server = None
        self.thread = None

    def start(self):
        """Start the HTTP server in a background thread."""
        # Set the class variables so handlers can access them
        FollowServerHandler.target_state = self.target_state
        FollowServerHandler.shared_state = self.shared_state
        
        self.server = HTTPServer((self.host, self.port), FollowServerHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f"[follow-server] Started on http://{self.host}:{self.port}")
        print(f"[follow-server] POST /follow/<id> to select a person to follow")
        print(f"[follow-server] POST /follow/clear to clear target (follow biggest)")
        print(f"[follow-server] GET /status to check current target")

    def stop(self):
        """Stop the HTTP server."""
        if self.server:
            self.server.shutdown()
            print("[follow-server] Stopped")
