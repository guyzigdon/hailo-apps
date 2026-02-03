# region imports
# Standard library imports
import setproctitle
from pathlib import Path
from typing import Optional, Any
import hailo
import json
import tempfile
import os

# Local application-specific imports
from hailo_apps.python.core.common.core import get_pipeline_parser, handle_list_models_flag
from hailo_apps.python.core.common.defines import TILING_APP_TITLE, TILING_PIPELINE
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import SOURCE_PIPELINE, INFERENCE_PIPELINE, USER_CALLBACK_PIPELINE, DISPLAY_PIPELINE, TILE_CROPPER_PIPELINE
from hailo_apps.python.core.gstreamer.gstreamer_app import GStreamerApp, app_callback_class, dummy_callback
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.pipeline_apps.tiling.configuration import TilingConfiguration
from hailo_apps.python.core.common.hef_utils import get_hef_labels_json

hailo_logger = get_logger(__name__)
# endregion imports

# -----------------------------------------------------------------------------------------------
# Main Tiling Application Class
# -----------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------

# This class inherits from the hailo_rpi_common.GStreamerApp class
class GStreamerTilingApp(GStreamerApp):
    nms_score_threshold: float = None

    
    def __init__(self, app_callback: Any, user_data: Any, parser: Optional[Any] = None) -> None:
        if parser is None:
            parser = get_pipeline_parser()

        # Add tiling-specific arguments
        self._add_tiling_arguments(parser)
        
        # Handle --list-models flag before full initialization
        handle_list_models_flag(parser, TILING_PIPELINE)

        super().__init__(parser, user_data)
    
        # Initialize tiling configuration
        self.config = TilingConfiguration(
            self.options_menu,
            self.video_width,
            self.video_height,
            self.arch
        )

        # Copy configuration attributes to self for compatibility
        self._copy_config_attributes()
        user_data.detection_log_file = f"results/{Path(self.hef_path).stem}_{self.frame_rate}fps_{Path(self.video_source).stem}_t{self.config.tiles_x}x{self.config.tiles_y}_detections.log"
        # User-defined label JSON file
        self.fps_log_file = f"results/{Path(self.hef_path).stem}_tiling_{self.config.tiles_x}x{self.config.tiles_y}_{self.frame_rate}fps_{Path(self.video_source).stem}.log"
        self.labels_json = self.options_menu.labels_json
        if self.labels_json is None: # if no labels JSON file is provided, try auto-detect it from the HEF file
            self.labels_json = get_hef_labels_json(self.hef_path)
            if self.labels_json is not None:
                hailo_logger.info("Auto detected Labels JSON: %s", self.labels_json)

        # Force yellow color for detections
        if self.labels_json and os.path.exists(self.labels_json):
            try:
                with open(self.labels_json, 'r') as f:
                    labels = json.load(f)
                
                # Check if it's the expected format (list of dicts)
                if isinstance(labels, list):
                    modified = False
                    for label in labels:
                        if 'color' in label:
                            # Set color to Yellow [255, 255, 0]
                            label['color'] = [255, 255, 0]
                            modified = True
                    
                    if modified:
                        # Create temp file
                        fd, temp_path = tempfile.mkstemp(suffix='.json', prefix='hailo_labels_yellow_')
                        with os.fdopen(fd, 'w') as f:
                            json.dump(labels, f, indent=4)
                        
                        hailo_logger.info(f"Created temporary labels JSON with yellow colors: {temp_path}")
                        self.labels_json = temp_path
            except Exception as e:
                hailo_logger.warning(f"Failed to patch labels JSON for yellow color: {e}")

        self.app_callback = app_callback
        setproctitle.setproctitle(TILING_APP_TITLE)

        # Print configuration summary
        self._print_configuration()

        self.create_pipeline()

    def _add_tiling_arguments(self, parser: Any) -> None:
        """Add tiling-specific command line arguments."""
        # Labels JSON option
        parser.add_argument(
            "--labels-json",
            default=None,
            help="Path to custom labels JSON file",
        )

        # Tiling options (auto mode by default, manual if tiles-x/y specified)
        parser.add_argument("--tiles-x", type=int, default=None,
                          help="Number of tiles horizontally (triggers manual mode)")
        parser.add_argument("--tiles-y", type=int, default=None,
                          help="Number of tiles vertically (triggers manual mode)")
        parser.add_argument("--min-overlap", type=float, default=0.1,
                          help="Minimum overlap ratio (0.0-0.5). Default: 0.1 (10%% of tile size). "
                               "Should be ≥ (smallest_object_size / model_input_dimension)")

        # Multi-scale options
        parser.add_argument("--multi-scale", action="store_true",
                          help="Enable multi-scale tiling with predefined grids")
        parser.add_argument("--scale-levels", type=int, default=1, choices=[1, 2, 3],
                          help="Scale levels for multi-scale mode: 1={1x1}, 2={1x1+2x2}, 3={1x1+2x2+3x3}. Default: 1")

        # Detection options
        parser.add_argument("--iou-threshold", type=float, default=0.3,
                          help="NMS IOU threshold (default: 0.3)")
        parser.add_argument("--nms-score-threshold", type=float, default=None,
                          help="NMS Score threshold (default: 0.001 if not specified)")
        parser.add_argument("--border-threshold", type=float, default=0.15,
                          help="Border threshold for multi-scale mode (default: 0.15)")
        
        # Codec options
        parser.add_argument("--input-codec", type=str, default="auto", choices=['auto', 'h264', 'h265', 'hevc'],
                          help="Force specific codec for hardware decoding (e.g. h265 for RPi hardware). Default: auto")
        
        # Display options
        parser.add_argument("--no-display", action="store_true", help="Disable display window (headless mode)")

    def _copy_config_attributes(self) -> None:
        """Copy configuration attributes to self for compatibility."""
        # Video source
        self.video_source = self.config.video_source

        # Model configuration
        self.hef_path = self.config.hef_path
        self.model_type = self.config.model_type
        self.model_input_width = self.config.model_input_width
        self.model_input_height = self.config.model_input_height
        self.post_function = self.config.post_function
        self.post_process_so = self.config.post_process_so

        # Tiling configuration
        self.tiles_x = self.config.tiles_x
        self.tiles_y = self.config.tiles_y
        self.overlap_x = self.config.overlap_x
        self.overlap_y = self.config.overlap_y
        self.tile_size_x = self.config.tile_size_x
        self.tile_size_y = self.config.tile_size_y
        self.tiling_mode = self.config.tiling_mode
        self.used_larger_tiles = getattr(self.config, 'used_larger_tiles', False)
        self.min_overlap = self.config.min_overlap

        # Multi-scale configuration
        self.use_multi_scale = self.config.use_multi_scale
        self.scale_level = self.config.scale_level
        self.batch_size = self.config.batch_size

        # Detection configuration
        self.iou_threshold = self.config.iou_threshold
        self.nms_score_threshold = self.config.nms_score_threshold
        self.border_threshold = self.config.border_threshold
        self.input_codec = getattr(self.options_menu, 'input_codec', 'auto')
        self.no_display = getattr(self.options_menu, 'no_display', False)

        # Frame rate adjustment for hailo8l with mobilenet
        if self.model_type == "mobilenet" and self.arch != 'hailo8' and self.batch_size == 15:
            self.frame_rate = 19  # changing frame rate to support hailo8l lower performance.


    def _print_configuration(self) -> None:
        """
        Print a user-friendly configuration summary to the console.
        """
        print("\n" + "="*70)
        print("TILING CONFIGURATION")
        print("="*70)

        # Input information
        print(f"Input Resolution:     {self.video_width}x{self.video_height}")
        print(f"Model:                {Path(self.hef_path).name} ({self.model_type.upper()}, {self.model_input_width}x{self.model_input_height})")

        # Tiling mode and configuration (always show custom tiles)
        print(f"\nTiling Mode:          {self.tiling_mode.upper()}")
        print(f"Custom Tile Grid:     {self.tiles_x}x{self.tiles_y} = {self.tiles_x * self.tiles_y} tiles")

        if hasattr(self, 'used_larger_tiles') and self.used_larger_tiles:
            print(f"Tile Size:            {int(self.tile_size_x)}x{int(self.tile_size_y)} pixels (enlarged to meet min overlap)")
        else:
            print(f"Tile Size:            {self.model_input_width}x{self.model_input_height} pixels")
        # Calculate overlap in pixels using actual tile sizes
        overlap_pixels_x = int(self.overlap_x * self.tile_size_x)
        overlap_pixels_y = int(self.overlap_y * self.tile_size_y)
        print(f"Overlap:              X: {self.overlap_x*100:.1f}% (~{overlap_pixels_x}px), Y: {self.overlap_y*100:.1f}% (~{overlap_pixels_y}px)")

        # Multi-scale additional information
        if self.use_multi_scale:
            print(f"\nMulti-Scale:          ENABLED (scale-level={self.scale_level})")
            if self.scale_level == 1:
                print(f"  Additional Grids:   1x1 = 1 tile")
                predefined = 1
            elif self.scale_level == 2:
                print(f"  Additional Grids:   1x1 + 2x2 = 5 tiles")
                predefined = 5
            else:  # scale_level == 3
                print(f"  Additional Grids:   1x1 + 2x2 + 3x3 = 14 tiles")
                predefined = 14
            custom = self.tiles_x * self.tiles_y
            print(f"  Total Tiles:        {custom} (custom) + {predefined} (predefined) = {self.batch_size}")
        else:
            print(f"\nMulti-Scale:          DISABLED")
            print(f"  Total Tiles:        {self.batch_size}")

        # Detection parameters
        print(f"\nDetection Parameters:")
        print(f"  Batch Size:         {self.batch_size}")
        print(f"  IOU Threshold:      {self.iou_threshold}")
        print(f"  NMS Score Threshold: {self.nms_score_threshold}")
        if self.use_multi_scale:
            print(f"  Border Threshold:   {self.border_threshold}")

        # Overlap information
        # Use average for min overlap pixels display
        avg_model_size = (self.model_input_width + self.model_input_height) / 2
        min_overlap_pixels = int(self.min_overlap * avg_model_size)

        if hasattr(self, 'used_larger_tiles') and self.used_larger_tiles:
            print(f"\nNote:                 Tile sizes enlarged to {int(self.tile_size_x)}x{int(self.tile_size_y)} to meet minimum overlap requirement")

        if overlap_pixels_x < min_overlap_pixels or overlap_pixels_y < min_overlap_pixels:
            print(f"  ⚠️  Warning:         Overlap below minimum ({min_overlap_pixels}px)")
        elif overlap_pixels_x < 50 or overlap_pixels_y < 50:
            print(f"  ⚠️  Warning:         Very small overlap may miss objects on boundaries")

        print("="*70 + "\n")

    def get_pipeline_string(self) -> str:
        """
        Build the GStreamer pipeline string with configured tiling parameters.

        Returns:
            str: Complete GStreamer pipeline string
        """
        source_pipeline = SOURCE_PIPELINE(
            video_source=self.video_source,
            video_width=self.video_width,
            video_height=self.video_height,
            frame_rate=self.frame_rate,
            sync=self.sync,
            input_codec=self.input_codec,
        )

        # Default to 0.001 if not specified (original behavior)
        nms_score_thresh = self.nms_score_threshold if self.nms_score_threshold is not None else 0.001
        
        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function,
            batch_size=self.batch_size,
            config_json=self.labels_json,
            additional_params=f"nms-score-threshold={nms_score_thresh}"
        )

        # Configure tile cropper with calculated parameters
        # tiling_mode: 0 = single-scale, 1 = multi-scale
        tiling_mode = 1 if self.use_multi_scale else 0

        # Set scale_level based on mode
        # Single-scale: scale_level not used (pass 0 to skip in pipeline string)
        # Multi-scale: scale_level 1={1x1}, 2={1x1,2x2}, 3={1x1,2x2,3x3}
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
            border_threshold=self.border_threshold
        )

        user_callback_pipeline = USER_CALLBACK_PIPELINE()

        if self.no_display:
            # Use fakesink for headless execution
            # sync=true to respect framerate, sync=false for max speed
            # User might want max speed if analyzing file, but let's stick to 'self.sync' which depends on input type
            output_pipeline = f"fakesink sync={self.sync}"
        else:
            output_pipeline = DISPLAY_PIPELINE(
                video_sink=self.video_sink,
                sync=self.sync,
                show_fps=self.show_fps
            )

        pipeline_string = (
            f'{source_pipeline} ! '
            f'{tile_cropper_pipeline} ! '
            f'{user_callback_pipeline} ! '
            f'{output_pipeline}'
        )
        

        hailo_logger.debug(f"Pipeline string: {pipeline_string}")
        return pipeline_string

# User-defined class to be used in the callback function: Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    detection_log_file: str = ""

    def __init__(self):
        super().__init__()



# User-defined callback function: This is the callback function that will be called when data is available from the pipeline
def app_callback(element, buffer, user_data):
    # Note: Frame counting is handled automatically by the framework wrapper
    # Log absolute timestamp if available
    pts = buffer.pts
    timestamp_str = ""
    if pts is not None:
        seconds = pts / 1e9
        timestamp_str = f" | Timestamp: {seconds:.3f}s"
    
    string_to_print = f"Frame count: {user_data.get_count()}{timestamp_str}\n"
    
    if buffer is None:
        hailo_logger.warning("Received None buffer at frame=%s", user_data.get_count())
        return
    
    log_filename = user_data.detection_log_file
    with open(log_filename, "a") as log_file:
        for detection in hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_DETECTION):
            string_to_print += (f"Detection: {detection.get_label()} Confidence: {detection.get_confidence():.2f}\n")
            bbox = detection.get_bbox()
            string_to_print += (f"\tBbox: x_min={bbox.xmin()}, y_min={bbox.ymin()}, w={bbox.width()}, h={bbox.height()}\n")
        log_file.write(string_to_print + "\n")

    print(string_to_print)
    return


def main() -> None:
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    app = GStreamerTilingApp(app_callback, user_data)
    app.run()

if __name__ == "__main__":
    print("Starting Hailo Tiling App...")
    main()
