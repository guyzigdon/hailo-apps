#!/usr/bin/env python3
"""
Drone Follow — a Hailo pipeline app that follows a person with a drone.

Usage:
    python drone_follow.py --input rpi  # live mode with camera + drone

Pipeline options (--input, --input-codec, etc.) are passed through to the tiling pipeline.
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import threading
import time

import hailo
import numpy as np

try:
    from drone_control import (
        ControllerConfig, Detection, SharedDetectionState, run_live_drone
    )
    from follow_server import FollowServer, FollowTargetState
except ImportError:
    from .drone_control import (
        ControllerConfig, Detection, SharedDetectionState, run_live_drone
    )
    from .follow_server import FollowServer, FollowTargetState

# ---------------------------------------------------------------------------
# ByteTracker import
# ---------------------------------------------------------------------------

try:
    from byte_tracker import ByteTracker
except ImportError:
    from .byte_tracker import ByteTracker

LOGGER = logging.getLogger("drone_follow.app")

_EMPTY_DET_ARRAY = np.empty((0, 5), dtype=np.float32)


def _configure_logging(verbosity: str) -> None:
    level = {
        "quiet": logging.WARNING,
        "normal": logging.INFO,
        "debug": logging.DEBUG,
    }.get(verbosity, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger().setLevel(level)


# ---------------------------------------------------------------------------
# Hailo App Callback
# ---------------------------------------------------------------------------

def _maybe_clear_target_after_lost(user_data):
    """Clear the follow target only if it has been missing longer than tracking_lost_timeout_s."""
    target_state = user_data.target_state
    if target_state is None:
        return
    target_id = target_state.get_target()
    if target_id is None:
        return
    last_seen = target_state.get_last_seen()
    if last_seen is None:
        target_state.set_target(None)
        return
    if time.monotonic() - last_seen >= user_data.tracking_lost_timeout_s:
        target_state.set_target(None)


def _build_det_info(person, track_id=None):
    """Build a UI detection dict from a Hailo detection object."""
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
    if track_id is not None:
        det_info["id"] = track_id
    return det_info


def _update_ui(ui_state, persons, person_to_id, following_id):
    """Push detection metadata to the web UI if enabled."""
    if ui_state is None:
        return
    all_dets = [_build_det_info(p, person_to_id.get(id(p))) for p in persons]
    ui_state.update_detections(all_dets, following_id)


def _run_tracker(byte_tracker, persons, hailo):
    """Run ByteTracker (or HailoTracker fallback) and return (available_ids, person_by_id, person_to_id).

    person_by_id:  {track_id -> person detection}
    person_to_id:  {id(person) -> track_id}  (reverse lookup)
    """
    available_ids = set()
    person_by_id = {}

    if byte_tracker is not None:
        SCALE = 1000.0
        det_array = np.empty((len(persons), 5), dtype=np.float32)
        for i, person in enumerate(persons):
            bbox = person.get_bbox()
            det_array[i, 0] = bbox.xmin() * SCALE
            det_array[i, 1] = bbox.ymin() * SCALE
            det_array[i, 2] = (bbox.xmin() + bbox.width()) * SCALE
            det_array[i, 3] = (bbox.ymin() + bbox.height()) * SCALE
            det_array[i, 4] = person.get_confidence()

        all_tracks = byte_tracker.update(det_array)

        for t in all_tracks:
            if t.is_activated and 0 <= t.input_index < len(persons):
                available_ids.add(t.track_id)
                person_by_id[t.track_id] = persons[t.input_index]
            elif t.is_activated:
                available_ids.add(t.track_id)
    else:
        for person in persons:
            track = person.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()
                available_ids.add(track_id)
                person_by_id[track_id] = person

    person_to_id = {id(p): tid for tid, p in person_by_id.items()}
    return available_ids, person_by_id, person_to_id


def app_callback(element, buffer, user_data):
    """Tiling pipeline callback: pick largest person (or specific tracked person), update shared state.

    When ByteTracker is enabled (user_data.byte_tracker is not None):
    1. Convert detections to Nx5 array, run tracker.update() synchronously
    2. Each returned track has input_index pointing to the matched detection
    3. Build person_by_id directly -- no cross-frame IoU re-matching needed
    """
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    persons = [d for d in detections if d.get_label() == "person"]

    target_state = user_data.target_state
    ui_state = user_data.ui_state

    if not persons:
        if user_data.byte_tracker is not None:
            user_data.byte_tracker.update(_EMPTY_DET_ARRAY)
        user_data.shared_state.update(None, available_ids=set())
        if target_state is not None:
            _maybe_clear_target_after_lost(user_data)
        _update_ui(ui_state, [], {}, target_state.get_target() if target_state else None)
        if target_state is None or target_state.get_target() is None:
            LOGGER.info("[SEARCH MODE] No person detected in frame - follow state cleared")
        return

    available_ids, person_by_id, person_to_id = _run_tracker(
        user_data.byte_tracker, persons, hailo)

    # --- Target selection ---
    target_id = target_state.get_target() if target_state is not None else None

    best = None
    follow_mode = ""
    if target_id is not None:
        best = person_by_id.get(target_id)

        if best is not None:
            target_state.update_last_seen()
            follow_mode = f"ID {target_id}"
        else:
            user_data.shared_state.update(None, available_ids=available_ids)
            _maybe_clear_target_after_lost(user_data)
            _update_ui(ui_state, persons, person_to_id, target_state.get_target())
            if target_state.get_target() is None:
                LOGGER.info("[SEARCH MODE] Target ID %s not in frame. Available: %s - follow state cleared",
                            target_id, sorted(available_ids) if available_ids else "none")
            return
    else:
        best = max(persons, key=lambda d: d.get_bbox().width() * d.get_bbox().height())
        best_tid = person_to_id.get(id(best))
        if best_tid is not None and target_state is not None:
            target_state.set_target(best_tid)
            follow_mode = f"locked ID {best_tid}"
        elif best_tid is not None:
            follow_mode = f"largest (ID {best_tid})"
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

    _update_ui(ui_state, persons, person_to_id,
               target_state.get_target() if target_state else None)

    available_str = f"Available: {sorted(available_ids)}" if available_ids else ""
    LOGGER.info("[FOLLOWING %s] conf=%.2f center=(%.2f,%.2f) h=%.2f %s",
                follow_mode, best.get_confidence(), cx, cy, bbox.height(), available_str)


# ---------------------------------------------------------------------------
# Hailo App Setup
# ---------------------------------------------------------------------------

def _add_drone_follow_args(parser):
    """Register drone-follow CLI flags on a pipeline parser."""
    defaults = ControllerConfig()
    group = parser.add_argument_group("drone-follow")

    # Framing and target geometry
    group.add_argument("--hfov", type=float, default=defaults.hfov)
    group.add_argument("--vfov", type=float, default=defaults.vfov)
    group.add_argument("--target-bbox-height", type=float, default=None,
                       help=f"Target bbox height (0-1). Mutually exclusive with --target-distance. "
                            f"(default when no --target-distance: {defaults.target_bbox_height})")
    group.add_argument("--target-distance", type=float, default=None, metavar="M",
                       help="Desired horizontal distance to person in metres. Requires --fixed-altitude. "
                            "Mutually exclusive with --target-bbox-height. "
                            f"(default: {defaults.target_distance_m})")
    group.add_argument("--person-height", type=float, default=defaults.person_height_m, metavar="M",
                       help=f"Assumed person height for distance calculation (default: {defaults.person_height_m}m)")

    # Controller gains and loop behavior
    group.add_argument("--control-loop-hz", type=float, default=defaults.control_loop_hz)
    group.add_argument("--dead-zone-height-percent", type=float, default=defaults.dead_zone_height_percent,
                       help="Forward dead zone as %% of target bbox height (default: 5)")
    group.add_argument("--yaw-gain", dest="kp_yaw", type=float, default=defaults.kp_yaw)
    group.add_argument("--forward-gain", dest="kp_forward", type=float, default=defaults.kp_forward)
    group.add_argument("--backward-gain", dest="kp_backward", type=float, default=defaults.kp_backward,
                       help="Gain for backward movement when too close (default: 5.0)")
    group.add_argument("--pitch-gain", dest="kp_down", type=float, default=defaults.kp_down)

    # Flight mode and mission lifecycle
    group.add_argument("--fixed-altitude", action="store_true")
    group.add_argument("--yaw-only", action="store_true",
                       help="Yaw only mode: no forward/backward or altitude movement")
    group.add_argument("--no-takeoff-landing", action="store_true",
                       help="Do not take off or land; assume drone is already in offboard mode")
    group.add_argument("--takeoff-altitude", type=float, default=defaults.takeoff_altitude)
    group.add_argument("--mission-duration", type=float, default=300.0)

    # Search/follow behavior
    group.add_argument("--search-enter-delay", type=float, default=defaults.search_enter_delay_s,
                       help="Seconds without detection before active search starts (default: 2.0)")
    group.add_argument("--search-vel-damp", type=float, default=defaults.search_vel_damp,
                       help="Dampening factor for forward/backward speed during search based on last detection (default: 0.3)")
    group.add_argument("--search-timeout", type=float, default=defaults.search_timeout_s,
                       help="Seconds before landing if no person is found (default: 60.0)")
    group.add_argument("--tracking-lost-timeout", type=float, default=2.0,
                       help="Seconds to keep following a track ID after target leaves frame (looser tracking)")

    # Smoothing
    group.add_argument("--smooth-forward", action=argparse.BooleanOptionalAction, default=defaults.smooth_forward,
                       help=f"Enable/disable forward velocity smoothing (default: {defaults.smooth_forward})")
    group.add_argument("--forward-alpha", type=float, default=defaults.forward_alpha,
                       help=f"EMA smoothing factor for forward velocity (0=sluggish, 1=no smoothing, default: {defaults.forward_alpha})")
    group.add_argument("--log-verbosity", choices=["quiet", "normal", "debug"], default=defaults.log_verbosity,
                       help="Console log verbosity (default: normal)")

    # Safety limits
    group.add_argument("--max-forward", type=float, default=defaults.max_forward,
                       help=f"Max forward speed in m/s (default: {defaults.max_forward})")
    group.add_argument("--max-backward", type=float, default=defaults.max_backward,
                       help=f"Max backward speed in m/s (default: {defaults.max_backward})")
    group.add_argument("--max-bbox-height-safety", type=float, default=defaults.max_bbox_height_safety,
                       help="Safety limit: stop/retreat if bbox height > limit (0.0-1.0) (default: 0.8)")

    # Connectivity
    group.add_argument("--connection", default="udpin://0.0.0.0:14540",
                       help="MAVLink connection string (default: udpin://0.0.0.0:14540)")
    group.add_argument("--serial", nargs="?", const="/dev/ttyACM0", default=None,
                       metavar="DEVICE",
                       help="Connect to CubeOrange via serial cable instead of UDP. "
                            "Optionally specify device path (default: /dev/ttyACM0)")
    group.add_argument("--serial-baud", type=int, default=57600,
                       help="Baud rate for serial connection (default: 57600)")

    # Servers/UI
    group.add_argument("--follow-server-port", type=int, default=8080,
                       help="HTTP server port for target selection (only with --enable-tracking)")
    group.add_argument("--ui", action="store_true",
                       help="Enable web UI with live video and clickable bounding boxes")
    group.add_argument("--ui-port", type=int, default=5001,
                       help="Web UI server port (default: 5001)")
    group.add_argument("--ui-fps", type=int, default=10,
                       help="MJPEG stream frame rate (default: 10)")

def _resolve_serial_connection(args):
    """If --serial is given, override --connection with a serial:// URI."""
    if getattr(args, "serial", None) is not None:
        baud = getattr(args, "serial_baud", 57600)
        args.connection = f"serial://{args.serial}:{baud}"
        LOGGER.info("[drone] Serial mode: connection = %s", args.connection)


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
        def __init__(self, shared_state, target_state=None, ui_state=None,
                     tracking_lost_timeout_s=2.0, byte_tracker=None):
            super().__init__()
            self.shared_state = shared_state
            self.target_state = target_state
            self.ui_state = ui_state
            self.tracking_lost_timeout_s = tracking_lost_timeout_s
            self.byte_tracker = byte_tracker

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
            from gi.repository import Gst
            self._Gst = Gst
            mjpeg_sink = self.pipeline.get_by_name("mjpeg_sink")
            if mjpeg_sink:
                mjpeg_sink.connect("new-sample", self._on_mjpeg_sample)

        def _on_mjpeg_sample(self, appsink):
            """appsink callback: extract pre-encoded JPEG bytes."""
            Gst = self._Gst
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
            # Override parent to never insert hailotracker GStreamer element;
            # ByteTracker runs in Python instead.
            saved = self.enable_tracking
            self.enable_tracking = False
            if not self._ui_enabled:
                result = super().get_pipeline_string()
                self.enable_tracking = saved
                return result
            self.enable_tracking = saved

            # Build pipeline with tee: one branch for display, one for MJPEG appsink
            from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
                SOURCE_PIPELINE, INFERENCE_PIPELINE, USER_CALLBACK_PIPELINE,
                TILE_CROPPER_PIPELINE,
            )

            source_pipeline = SOURCE_PIPELINE(
                video_source=self.video_source,
                video_width=self.video_width,
                video_height=self.video_height,
                frame_rate=self.frame_rate,
                sync=self.sync,
                input_codec=self.input_codec,
                horizontal_mirror=not self.no_horizontal_mirror,
                vertical_mirror=self.vertical_mirror,
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

            user_callback_pipeline = USER_CALLBACK_PIPELINE()

            # Display branch (with overlay). Only disable when user passed --no-display.
            no_display = getattr(self.options_menu, 'no_display', False)
            if no_display:
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
            pipeline_parts.extend([user_callback_pipeline, output_pipeline])

            return ' ! '.join(pipeline_parts)

    parser = get_pipeline_parser()
    _add_drone_follow_args(parser)

    # Pre-parse to check if tracking is enabled (so we can create the tracker before full parse)
    _track_pre = argparse.ArgumentParser(add_help=False)
    _track_pre.add_argument("--enable-tracking", action="store_true")
    _track_pre_args, _ = _track_pre.parse_known_args()

    tracker = None
    if _track_pre_args.enable_tracking:
        tracker = ByteTracker(
            track_thresh=0.4, track_buffer=90, match_thresh=0.5, frame_rate=30,
        )
        LOGGER.info("[tracking] ByteTracker running synchronously in callback")

    user_data = DroneFollowUserData(
        shared_state, target_state, ui_state=ui_state, byte_tracker=tracker,
    )
    app = DroneFollowTilingApp(
        app_callback, user_data, parser=parser, eos_reached=eos_reached,
        ui_enabled=(ui_state is not None), ui_state=ui_state, ui_fps=ui_fps,
    )
    user_data.tracking_lost_timeout_s = getattr(app.options_menu, 'tracking_lost_timeout', 2.0)
    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
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
        try:
            from web_server import WebServer, SharedUIState
        except ImportError:
            from .web_server import WebServer, SharedUIState
        ui_state = SharedUIState()
        # Check that the UI has been built
        _ui_build_index = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ui", "build", "index.html")
        if not os.path.isfile(_ui_build_index):
            LOGGER.error("Web UI has not been built yet.")
            LOGGER.error("  cd hailo_apps/python/pipeline_apps/drone_follow/ui")
            LOGGER.error("  npm install")
            LOGGER.error("  npm run build")
            raise SystemExit(1)
        # Auto-enable tracking for UI (stable IDs needed for click-to-follow)
        if not ui_pre_args.enable_tracking:
            import sys as _sys
            _sys.argv.append("--enable-tracking")
            LOGGER.info("[ui] Auto-enabling tracking for UI mode")

    app = create_app(shared_state, target_state=target_state, eos_reached=eos_reached,
                     ui_state=ui_state, ui_fps=ui_pre_args.ui_fps)
    args = app.options_menu
    _configure_logging(getattr(args, "log_verbosity", "normal"))
    _resolve_serial_connection(args)

    # Create controller config once so it can be shared (and mutated via web UI)
    controller_config = ControllerConfig.from_args(args)

    # Start follow server (always available)
    follow_server = FollowServer(target_state, shared_state, port=args.follow_server_port)
    follow_server.start()

    # Start web UI server
    if ui_state is not None:
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "build")
        web_server = WebServer(ui_state, target_state, shared_state,
                               controller_config=controller_config,
                               port=args.ui_port, static_dir=static_dir)
        web_server.start()

    takeoff_done = threading.Event()

    def _eos_to_shutdown():
        eos_reached.wait()
        shutdown.set()
    threading.Thread(target=_eos_to_shutdown, daemon=True).start()

    def run_pipeline():
        takeoff_done.wait()
        if shutdown.is_set():
            return
        try:
            app.run()
        except SystemExit:
            pass
    pipeline_thread = threading.Thread(target=run_pipeline, daemon=False)
    pipeline_thread.start()

    def on_signal(*_):
        if not shutdown.is_set():
            shutdown.set()
            takeoff_done.set()
            LOGGER.warning("[drone] Ctrl+C received, shutting down...")
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
                          takeoff_done=takeoff_done, pipeline_quit_cb=app.loop.quit,
                          config=controller_config, ui_state=ui_state))
    except KeyboardInterrupt:
        if not shutdown.is_set():
            shutdown.set()
        takeoff_done.set()
        LOGGER.warning("[drone] Shutdown.")
    finally:
        if web_server is not None:
            web_server.stop()
        follow_server.stop()
        pipeline_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
