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
import os
import signal
import threading 
import time

from drone_control import (
    Detection, SharedDetectionState, run_drone, run_live_drone
)
from follow_server import FollowServer, FollowTargetState

# ---------------------------------------------------------------------------
# Hailo App Callback
# ---------------------------------------------------------------------------

def app_callback(element, buffer, user_data):
    """Tiling pipeline callback: pick largest person (or specific tracked person), update shared state."""
    import hailo
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    persons = [d for d in detections if d.get_label() == "person"]
    frame = user_data.get_count()
    
    if not persons:
        user_data.shared_state.update(None, available_ids=set())
        # Clear follow state when no person in frame
        if hasattr(user_data, 'target_state') and user_data.target_state is not None:
            user_data.target_state.set_target(None)
        print(f"\r[SEARCH MODE] No person detected in frame - follow state cleared", end="", flush=True)
        return
    
    # Build map of available IDs (if tracking is enabled)
    available_ids = set()
    person_by_id = {}
    for person in persons:
        track = person.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()
            available_ids.add(track_id)
            person_by_id[track_id] = person
    
    # If tracking is enabled and a target is set, follow that specific person
    target_id = None
    if hasattr(user_data, 'target_state') and user_data.target_state is not None:
        target_id = user_data.target_state.get_target()
    
    best = None
    follow_mode = ""
    if target_id is not None:
        # Look for the person with the matching track ID
        best = person_by_id.get(target_id)
        
        if best is not None:
            user_data.target_state.update_last_seen()
            follow_mode = f"ID {target_id}"
        else:
            # Target not found, clear detection and clear follow state
            user_data.shared_state.update(None, available_ids=available_ids)
            user_data.target_state.set_target(None)
            print(f"\r[SEARCH MODE] Target ID {target_id} not in frame. Available: {sorted(available_ids) if available_ids else 'none'} - follow state cleared", end="", flush=True)
            return
    else:
        # No specific target, pick the largest person
        best = max(persons, key=lambda d: d.get_bbox().width() * d.get_bbox().height())
        
        # Check if this person has a tracking ID
        best_track = best.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(best_track) == 1:
            follow_mode = f"largest (ID {best_track[0].get_id()})"
        else:
            follow_mode = "largest (no tracking)"
    
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
    ), available_ids=available_ids)
    
    # Log following status
    available_str = f"Available: {sorted(available_ids)}" if available_ids else ""
    print(f"\r[FOLLOWING {follow_mode}] conf={best.get_confidence():.2f} center=({cx:.2f},{cy:.2f}) h={bbox.height():.2f} {available_str}".ljust(120), end="", flush=True)


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
    group.add_argument("--follow-server-port", type=int, default=8080,
                       help="HTTP server port for target selection (only with --enable-tracking)")
    
    # Dry run specific
    group.add_argument("--mock-pattern", default="static",
                        choices=["static", "circle", "line", "sweep"])
    group.add_argument("--mock-x", type=float, default=0.7)
    group.add_argument("--mock-y", type=float, default=0.5)
    group.add_argument("--mock-circle-diameter", type=float, default=10.0, metavar="M",
                        help="Circle pattern diameter in meters (default: 10)")


def create_app(shared_state, target_state=None, eos_reached=None):
    """Create the tiling pipeline app with drone-follow callback.

    Follows the hailo-app pattern: build parser, create user_data,
    instantiate GStreamerTilingApp. If eos_reached is a threading.Event,
    EOS will set it instead of calling GStreamer shutdown (so we can land first).
    
    Args:
        shared_state: SharedDetectionState for passing detections to control loop
        target_state: FollowTargetState for tracking-based target selection (optional)
        eos_reached: threading.Event to signal EOS instead of shutdown (optional)
    """
    from hailo_apps.python.pipeline_apps.tiling.tiling_pipeline import (
        GStreamerTilingApp, user_app_callback_class,
    )
    from hailo_apps.python.core.common.core import get_pipeline_parser

    class DroneFollowUserData(user_app_callback_class):
        def __init__(self, shared_state, target_state=None):
            super().__init__()
            self.shared_state = shared_state
            self.target_state = target_state

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

    user_data = DroneFollowUserData(shared_state, target_state)
    return DroneFollowTilingApp(app_callback, user_data, parser=parser, eos_reached=eos_reached)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Pre-parse to determine mode before full arg parsing
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--dry-run", action="store_true")
    pre_args, _ = pre_parser.parse_known_args()

    if pre_args.dry_run:
        # === DRY-RUN MODE (mock detections + real drone) ===
        parser = argparse.ArgumentParser()
        _add_drone_follow_args(parser)
        args = parser.parse_args()
        
        # Start follow server (won't do much in mock mode but available for consistency)
        target_state = FollowTargetState()
        shared_state_for_dry_run = SharedDetectionState()
        follow_server = FollowServer(target_state, shared_state_for_dry_run, port=args.follow_server_port)
        follow_server.start()

        try:
            if hasattr(os, "fork") and os.name != "nt":
                r, w = os.pipe()
                pid = os.fork()
                if pid == 0:
                    os.close(w)
                    try:
                        os.setsid()
                        shutdown = asyncio.Event()
                        asyncio.run(run_drone(args, shared_state_for_dry_run, shutdown,
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
                    loop.run_until_complete(run_drone(args, shared_state_for_dry_run, shutdown))
                except KeyboardInterrupt:
                    if not shutdown.is_set():
                        shutdown.set()
                    print("\n[drone] Shutdown.")
        finally:
            follow_server.stop()

    else:
        # === HAILO MODES (structured as a hailo-app) ===
        # Hailo-app pattern: create shared state, build app, run pipeline
        shared_state = SharedDetectionState()
        shutdown = asyncio.Event()
        eos_reached = threading.Event()
        
        # Create target state for follow server
        target_state = FollowTargetState()
        
        app = create_app(shared_state, target_state=target_state, eos_reached=eos_reached)
        args = app.options_menu
        
        # Start follow server (always available)
        follow_server = FollowServer(target_state, shared_state, port=args.follow_server_port)
        follow_server.start()
        
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
                run_live_drone(args, shared_state, shutdown,
                              takeoff_done=takeoff_done, pipeline_quit_cb=app.loop.quit))
        except KeyboardInterrupt:
            if not shutdown.is_set():
                shutdown.set()
            print("\n[drone] Shutdown.")
        finally:
            follow_server.stop()
            pipeline_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
