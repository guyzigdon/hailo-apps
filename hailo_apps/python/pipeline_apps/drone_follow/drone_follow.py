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
        # Update UI state (empty detections)
        ui_state = getattr(user_data, 'ui_state', None)
        if ui_state is not None:
            following_id = user_data.target_state.get_target() if getattr(user_data, 'target_state', None) else None
            ui_state.update_detections([], following_id)
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
            # Still update UI with all visible persons even though target is lost
            ui_state = getattr(user_data, 'ui_state', None)
            if ui_state is not None:
                all_dets = []
                for person in persons:
                    pbbox = person.get_bbox()
                    det_info = {
                        "label": "person",
                        "confidence": round(person.get_confidence(), 3),
                        "bbox": {
                            "x": round(pbbox.xmin(), 4),
                            "y": round(pbbox.ymin(), 4),
                            "w": round(pbbox.width(), 4),
                            "h": round(pbbox.height(), 4),
                        },
                    }
                    ptrack = person.get_objects_typed(hailo.HAILO_UNIQUE_ID)
                    if len(ptrack) == 1:
                        det_info["id"] = ptrack[0].get_id()
                    all_dets.append(det_info)
                ui_state.update_detections(all_dets, None)
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
    
    # Update UI state with all person detections
    ui_state = getattr(user_data, 'ui_state', None)
    if ui_state is not None:
        all_dets = []
        for person in persons:
            pbbox = person.get_bbox()
            det_info = {
                "label": "person",
                "confidence": round(person.get_confidence(), 3),
                "bbox": {
                    "x": round(pbbox.xmin(), 4),
                    "y": round(pbbox.ymin(), 4),
                    "w": round(pbbox.width(), 4),
                    "h": round(pbbox.height(), 4),
                },
            }
            ptrack = person.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(ptrack) == 1:
                det_info["id"] = ptrack[0].get_id()
            all_dets.append(det_info)
        following_id = user_data.target_state.get_target() if getattr(user_data, 'target_state', None) else None
        ui_state.update_detections(all_dets, following_id)

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

    # UI options
    group.add_argument("--ui", action="store_true",
                       help="Enable web UI with live video and clickable bounding boxes")
    group.add_argument("--ui-port", type=int, default=5001,
                       help="Web UI server port (default: 5001)")
    group.add_argument("--ui-fps", type=int, default=10,
                       help="MJPEG stream frame rate (default: 10)")


def create_app(shared_state, target_state=None, eos_reached=None, ui_state=None, ui_fps=10):
    """Create the tiling pipeline app with drone-follow callback.

    Follows the hailo-app pattern: build parser, create user_data,
    instantiate GStreamerTilingApp. If eos_reached is a threading.Event,
    EOS will set it instead of calling GStreamer shutdown (so we can land first).

    Args:
        shared_state: SharedDetectionState for passing detections to control loop
        target_state: FollowTargetState for tracking-based target selection (optional)
        eos_reached: threading.Event to signal EOS instead of shutdown (optional)
        ui_state: SharedUIState for web UI (optional)
        ui_fps: MJPEG stream frame rate (default: 10)
    """
    from hailo_apps.python.pipeline_apps.tiling.tiling_pipeline import (
        GStreamerTilingApp, user_app_callback_class,
    )
    from hailo_apps.python.core.common.core import get_pipeline_parser
    from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
        QUEUE, DISPLAY_PIPELINE,
    )

    class DroneFollowUserData(user_app_callback_class):
        def __init__(self, shared_state, target_state=None, ui_state=None):
            super().__init__()
            self.shared_state = shared_state
            self.target_state = target_state
            self.ui_state = ui_state

    class DroneFollowTilingApp(GStreamerTilingApp):
        """Tiling app with EOS handling and optional MJPEG appsink for web UI."""
        def __init__(self, app_callback, user_data, parser=None, eos_reached=None,
                     ui_enabled=False, ui_state=None, ui_fps=30):
            self._eos_reached = eos_reached
            self._ui_enabled = ui_enabled
            self._ui_state = ui_state
            self._ui_fps = ui_fps
            super().__init__(app_callback, user_data, parser=parser)
            # Connect appsink after pipeline is created by super().__init__
            if self._ui_enabled:
                self._connect_mjpeg_sink()

        def _connect_mjpeg_sink(self):
            """Connect the MJPEG appsink's new-sample signal."""
            import gi
            gi.require_version("Gst", "1.0")
            mjpeg_sink = self.pipeline.get_by_name("mjpeg_sink")
            if mjpeg_sink:
                mjpeg_sink.connect("new-sample", self._on_mjpeg_sample)

        def _on_mjpeg_sample(self, appsink):
            """appsink callback: extract pre-encoded JPEG bytes."""
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
            sample = appsink.emit("pull-sample")
            if sample:
                buf = sample.get_buffer()
                success, map_info = buf.map(Gst.MapFlags.READ)
                if success:
                    self._ui_state.update_frame(bytes(map_info.data))
                    buf.unmap(map_info)
            return Gst.FlowReturn.OK

        def on_eos(self):
            if self._eos_reached is not None:
                self._eos_reached.set()
            else:
                super().on_eos()

        def _on_pipeline_rebuilt(self):
            super()._on_pipeline_rebuilt()
            if self._ui_enabled:
                self._connect_mjpeg_sink()

        def get_pipeline_string(self):
            if not self._ui_enabled:
                return super().get_pipeline_string()

            # Build pipeline with tee: one branch for display, one for MJPEG appsink
            from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
                SOURCE_PIPELINE, INFERENCE_PIPELINE, USER_CALLBACK_PIPELINE,
                TILE_CROPPER_PIPELINE, TRACKER_PIPELINE,
            )

            source_pipeline = SOURCE_PIPELINE(
                video_source=self.video_source,
                video_width=self.video_width,
                video_height=self.video_height,
                frame_rate=self.frame_rate,
                sync=self.sync,
                input_codec=self.input_codec,
                mirror_image=not self.no_mirror,
            )

            nms_score_thresh = self.nms_score_threshold if self.nms_score_threshold is not None else 0.001
            detection_pipeline = INFERENCE_PIPELINE(
                hef_path=self.hef_path,
                post_process_so=self.post_process_so,
                post_function_name=self.post_function,
                batch_size=self.batch_size,
                config_json=self.labels_json,
                additional_params=f"nms-score-threshold={nms_score_thresh}",
            )

            tiling_mode = 1 if self.use_multi_scale else 0
            scale_level = self.scale_level if self.use_multi_scale else 0
            tile_cropper_pipeline = TILE_CROPPER_PIPELINE(
                detection_pipeline,
                name='tile_cropper_wrapper',
                internal_offset=True,
                scale_level=scale_level,
                tiling_mode=tiling_mode,
                tiles_along_x_axis=self.tiles_x,
                tiles_along_y_axis=self.tiles_y,
                overlap_x_axis=self.overlap_x,
                overlap_y_axis=self.overlap_y,
                iou_threshold=self.iou_threshold,
                border_threshold=self.border_threshold,
            )

            tracker_pipeline = ""
            if self.enable_tracking:
                tracker_pipeline = TRACKER_PIPELINE(
                    class_id=self.tracking_class_id,
                    name='hailo_tracker',
                )

            user_callback_pipeline = USER_CALLBACK_PIPELINE()

            # Display branch (with overlay)
            if self.no_display:
                display_branch = f"fakesink sync={self.sync}"
            else:
                display_branch = DISPLAY_PIPELINE(
                    video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps,
                )

            # MJPEG branch (raw video, no overlay — React draws bboxes)
            mjpeg_branch = (
                f"videoconvert n-threads=2 ! "
                f"videorate max-rate={self._ui_fps} ! "
                f"video/x-raw,framerate={self._ui_fps}/1 ! "
                f"jpegenc quality=70 ! "
                f"appsink name=mjpeg_sink sync=false drop=true emit-signals=true"
            )

            # Tee splits into display + MJPEG
            output_pipeline = (
                f"tee name=ui_tee "
                f"ui_tee. ! {QUEUE(name='display_branch_q')} ! {display_branch} "
                f"ui_tee. ! {QUEUE(name='mjpeg_branch_q')} ! {mjpeg_branch}"
            )

            pipeline_parts = [source_pipeline, tile_cropper_pipeline]
            if tracker_pipeline:
                pipeline_parts.append(tracker_pipeline)
            pipeline_parts.extend([user_callback_pipeline, output_pipeline])

            return ' ! '.join(pipeline_parts)

    parser = get_pipeline_parser()
    _add_drone_follow_args(parser)

    user_data = DroneFollowUserData(shared_state, target_state, ui_state=ui_state)
    return DroneFollowTilingApp(
        app_callback, user_data, parser=parser, eos_reached=eos_reached,
        ui_enabled=(ui_state is not None), ui_state=ui_state, ui_fps=ui_fps,
    )


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

        # Pre-parse --ui flag to set up web UI before create_app parses all args
        ui_pre = argparse.ArgumentParser(add_help=False)
        ui_pre.add_argument("--ui", action="store_true")
        ui_pre.add_argument("--ui-port", type=int, default=5001)
        ui_pre.add_argument("--ui-fps", type=int, default=10)
        ui_pre.add_argument("--enable-tracking", action="store_true")
        ui_pre_args, _ = ui_pre.parse_known_args()

        ui_state = None
        web_server = None
        if ui_pre_args.ui:
            from web_server import WebServer, SharedUIState
            ui_state = SharedUIState()
            # Auto-enable tracking for UI (stable IDs needed for click-to-follow)
            if not ui_pre_args.enable_tracking:
                import sys as _sys
                _sys.argv.append("--enable-tracking")
                print("[ui] Auto-enabling tracking for UI mode")

        app = create_app(shared_state, target_state=target_state, eos_reached=eos_reached,
                         ui_state=ui_state, ui_fps=ui_pre_args.ui_fps)
        args = app.options_menu

        # Start follow server (always available)
        follow_server = FollowServer(target_state, shared_state, port=args.follow_server_port)
        follow_server.start()

        # Start web UI server
        if ui_state is not None:
            static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "build")
            web_server = WebServer(ui_state, target_state, shared_state,
                                   port=args.ui_port, static_dir=static_dir)
            web_server.start()

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
            if web_server is not None:
                web_server.stop()
            follow_server.stop()
            pipeline_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
