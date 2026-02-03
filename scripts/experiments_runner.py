import itertools
import subprocess
import argparse
from pathlib import Path
import os
import signal
import time
import re
import sys

import cv2
import re

def parse_detections_log(log_path):
    """Parse detections log file into a dictionary with both frame_idx and timestamp"""
    detections_by_timestamp = {}  # timestamp -> detections
    detections_by_frame = {}  # frame_idx -> detections
    current_frame = None
    current_timestamp = None
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            line = line.strip()
            # Frame header: "Frame count: 1 | Timestamp: ..."
            if line.startswith("Frame count:"):
                try:
                    # Extract frame count
                    parts = line.split("|")
                    current_frame = int(parts[0].split(":")[1].strip())
                    
                    # Extract timestamp if present
                    if len(parts) > 1 and "Timestamp:" in parts[1]:
                        ts_match = re.search(r'Timestamp:\s*([\d.]+)s', parts[1])
                        if ts_match:
                            current_timestamp = float(ts_match.group(1))
                    
                    detections_by_frame[current_frame] = []
                    if current_timestamp is not None:
                        detections_by_timestamp[current_timestamp] = []
                except:
                    current_frame = None
                    current_timestamp = None
            
            # Detection line: "Detection: car Confidence: 0.88"
            elif line.startswith("Detection:") and current_frame is not None:
                try:
                    parts = line.split("Confidence:")
                    label = parts[0].replace("Detection:", "").strip()
                    conf = float(parts[1].strip())
                    
                    # Bbox line usually follows: "Bbox: x_min=..., y_min=..."
                    if i + 1 < len(lines):
                        bbox_line = lines[i+1].strip()
                        if bbox_line.startswith("Bbox:"):
                            bp = bbox_line.replace("Bbox:", "").split(",")
                            bd = {}
                            for p in bp:
                                k, v = p.split("=")
                                bd[k.strip()] = float(v.strip())
                            
                            det = {
                                'label': label,
                                'score': conf,
                                'bbox': [bd.get('x_min', 0), bd.get('y_min', 0), bd.get('w', 0), bd.get('h', 0)],
                                'frame_count': current_frame,
                                'timestamp': current_timestamp
                            }
                            
                            # Store relative coordinates
                            detections_by_frame[current_frame].append(det)
                            if current_timestamp is not None:
                                detections_by_timestamp[current_timestamp].append(det)
                except Exception as e:
                    print(f"Error parsing detection line: {e}")
                    pass
    except Exception as e:
        print(f"Error reading log file {log_path}: {e}")
        
    return detections_by_frame, detections_by_timestamp

def generate_annotated_video(input_video_path, log_path, output_video_path):
    print(f"Generating annotated video: {output_video_path}...")
    
    # Parse detections
    detections_by_frame, detections_by_timestamp = parse_detections_log(log_path)
    if not detections_by_frame and not detections_by_timestamp:
        print("No detections found in log file.")
        return

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening input video: {input_video_path}")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {width}x{height} @ {fps}fps")
    print(f"Parsed {len(detections_by_frame)} unique frames with detections")
    
    # Output writer (using mp4v/h264)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    video_frame_idx = 0  # 0-based index for video frames
    
    # Build timestamp lookup for matching
    timestamps = sorted(detections_by_timestamp.keys()) if detections_by_timestamp else []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate this frame's timestamp (0-based)
        video_timestamp = video_frame_idx / fps
        
        # Try timestamp-based matching first
        matched_detections = []
        if timestamps:
            # Find closest timestamp
            closest_ts = min(timestamps, key=lambda t: abs(t - video_timestamp))
            time_diff = abs(closest_ts - video_timestamp)
            
            # Use 1/fps tolerance (one frame duration)
            if time_diff < (1.0 / fps):
                matched_detections = detections_by_timestamp.get(closest_ts, [])
                match_method = f"timestamp (diff={time_diff*1000:.1f}ms)"
            else:
                match_method = f"timestamp (no match, diff={time_diff*1000:.1f}ms)"
        else:
            # Fallback to frame count matching (1-based)
            log_frame_idx = video_frame_idx + 1
            matched_detections = detections_by_frame.get(log_frame_idx, [])
            match_method = f"frame_idx={log_frame_idx}"
        
        # Print frame info for every frame (first 20 frames)
        if video_frame_idx < 20:
            print(f"Frame {video_frame_idx}: ts={video_timestamp:.3f}s, dets={len(matched_detections)}, method={match_method}")
        
        # Draw detections
        for det in matched_detections:
            bbox = det['bbox']
            # Convert relative to absolute
            x = int(bbox[0] * width)
            y = int(bbox[1] * height)
            w = int(bbox[2] * width)
            h = int(bbox[3] * height)
            
            # Draw box (Yellow: BGR 0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            # Draw label
            label = f"{det['label']} {det['score']:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        out.write(frame)
        video_frame_idx += 1
        
        if video_frame_idx % 100 == 0:
            print(f"Processed {video_frame_idx} frames...", end='\r')

    cap.release()
    out.release()
    print(f"\nAnnotated video saved: {output_video_path}")
    print(f"Total frames processed: {video_frame_idx}")

def is_wayland():
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"

def get_display_info(force_virtual=False, width=1920, height=1080):
    display = os.environ.get('DISPLAY', ':0')
    
    # Try actual display first unless forced (but we are removing forcing logic per user request)
    # The user asked to "get rid of the virtual screen logic. display only on actual display"
    # So we assume physical display is what we want.
    
    try:
        # Just check physical display
        xr = subprocess.run(['xrandr', '--query'], capture_output=True, text=True, check=True, env={**os.environ, 'DISPLAY': display})
        for line in xr.stdout.split('\n'):
            if ' connected' in line:
                match = re.search(r'(\d+)x(\d+)\+(\d+)\+(\d+)', line)
                if match:
                    w, h, x, y = map(int, match.groups())
                    return {'display': display, 'width': w, 'height': h, 'x': x, 'y': y, 'virtual': False}
    except Exception:
        pass

    # If no physical display found, we could fallback or just return default
    # User said "display only on actual display", implies we shouldn't start Xvfb.
    # But if no display exists, GStreamer might fail. We return defaults for :0.
    return {'display': display, 'width': width, 'height': height, 'x': 0, 'y': 0, 'virtual': False}

def start_screen_recording(output_file, display_info):
    if is_wayland():
        return None, None

    screen_file = str(output_file) + '.screen.mkv'

    # Always record full screen
    cap_w = display_info['width']
    cap_h = display_info['height']
    offset_x = display_info['x']
    offset_y = display_info['y']

    # Ensure even dimensions for ffmpeg
    cap_w = cap_w - (cap_w % 2)
    cap_h = cap_h - (cap_h % 2)

    ffmpeg_cmd = [
        'ffmpeg',
        '-f', 'x11grab',
        '-video_size', f'{cap_w}x{cap_h}',
        '-framerate', '30',
        '-i', f"{display_info['display']}.0+{offset_x},{offset_y}",
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '28',
        '-y',
        screen_file
    ]

    print(f"Starting screen recording: {' '.join(ffmpeg_cmd)}")

    try:
        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, 'DISPLAY': display_info['display']}
        )
        time.sleep(2)
        if proc.poll() is not None:
            print("⚠️ ffmpeg exited immediately — screen recording disabled.")
            return None, None
        return proc, screen_file
    except Exception as e:
        print(f"⚠️ Screen recording failed: {e}")
        return None, None

def stop_screen_recording(proc, screen_file):
    if not proc:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
        if screen_file and os.path.exists(screen_file):
            size = os.path.getsize(screen_file)
            print(f"Screen recording saved: {screen_file} ({size} bytes)")
    except Exception:
        proc.kill()

def get_native_fps(hef_path, batch_size):
    print(f"Measuring native FPS for {hef_path}...")
    cmd = ["hailortcli", "run2", "set-net", hef_path, "--batch-size", str(batch_size)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        if not lines:
            return None
        last_line = lines[-1].strip()
        tokens = last_line.split()
        if not tokens:
            return None
        return float(tokens[-1])
    except Exception as e:
        print(f"Error measuring FPS: {e}")
        return None

def run_experiments(models, tiles, input_paths, extra_args=None,
                    remote=False, remote_user=None, remote_host=None, remote_path=None,
                    detection_args=None, recording_mode='none'):

    total = len(models) * len(tiles) * len(input_paths) * 1
    count = 0

    # display_info = get_display_info(force_virtual=record_screen) # removed force logic
    display_info = get_display_info(force_virtual=False)
    
    print(f"\nSession type: {os.environ.get('XDG_SESSION_TYPE', 'unknown')}")
    print(f"Display: {display_info['display']} {display_info['width']}x{display_info['height']}")
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    for model, (tx, ty) in itertools.product(models, tiles):
        # Resolve model path if possible, but keep original string if it's just a name
        # User requested "accept only path to models", so we assume 'model' is a path.
        model_path = Path(model)
        if not model_path.exists():
             print(f"Warning: Model path {model} does not exist. Attempting to use as-is.")
        
        native_fps = get_native_fps(model, tx * ty)
        if native_fps is None:
            print(f"Skipping {model} due to FPS check failure")
            count += 2
            continue

        base_fps = int(round(native_fps) / (tx * ty))
        base_fps = min(30, max(1, base_fps))  # Ensure FPS is between 1 and 30
        
        fps_values = sorted(list(set([base_fps])), reverse=True)

        for fps in fps_values:
            for input_path in input_paths:
                count += 1
                model_name = model_path.stem
                output_filename = f"{model_name}_{fps}fps_{Path(input_path).stem}_t{tx}x{ty}.mkv"
                output_file = str(results_dir / output_filename)
                log_file = f"{output_file}.log"

                cmd = [
                    "python", "tiling_pipeline.py",
                    "--input", input_path,
                    "--tiles-x", str(tx),
                    "--tiles-y", str(ty),
                    "--frame-rate", str(fps),
                    "--hef-path", str(model),  # Use full path
                    "--show-fps"
                ]
                if detection_args:
                    if 'iou_threshold' in detection_args:
                        cmd.extend(["--iou-threshold", str(detection_args['iou_threshold'])])
                    if detection_args.get('nms_score_threshold') is not None:
                        cmd.extend(["--nms-score-threshold", str(detection_args['nms_score_threshold'])])
                    if 'min_overlap' in detection_args:
                        cmd.extend(["--min-overlap", str(detection_args['min_overlap'])])

                if extra_args:
                    cmd.extend(extra_args)

                if recording_mode == "headless":
                    cmd.append("--no-display")

                print(f"\n=== Experiment {count}/{total} ===")
                print(" ".join(cmd))

                with open(log_file, "w") as lf:
                    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)

                time.sleep(8)

                ffmpeg_proc = None
                screen_file = None
                if recording_mode == "screen-record":
                    ffmpeg_proc, screen_file = start_screen_recording(output_file, display_info)

                result = proc.wait()

                stop_screen_recording(ffmpeg_proc, screen_file)

                if result == 0:
                    print("Experiment completed successfully.")
                    
                    # Post-process video generation for headless mode
                    if recording_mode == "headless":
                        annotated_video_path = str(output_file).replace(".mkv", "_annotated.mp4")
                        # Detection log path
                        det_log = str(output_file).replace(".mkv", "_detections.log")
                        if os.path.exists(det_log):
                            generate_annotated_video(input_path, det_log, annotated_video_path)
                        else:
                            print(f"Warning: Detection log not found at {det_log}, skipping annotation video.")
                else:
                    print(f"Experiment failed — see {log_file}")

                if remote:
                    assert remote_user and remote_host and remote_path
                    subprocess.run(
                        ["scp", output_file, f"{remote_user}@{remote_host}:{remote_path}/"],
                        check=True
                    )
                    os.remove(output_file)
                    if screen_file and os.path.exists(screen_file):
                        subprocess.run(
                            ["scp", screen_file, f"{remote_user}@{remote_host}:{remote_path}/"],
                            check=True
                        )
                        os.remove(screen_file)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', required=True, help="Paths to HEF models")
    p.add_argument('--tiles', nargs='+', required=True)
    p.add_argument('--input', nargs='+', required=True, help="Path to input video(s)")
    p.add_argument('--extra-args', nargs=argparse.REMAINDER)
    p.add_argument('--remote', action='store_true')
    p.add_argument('--remote-user')
    p.add_argument('--remote-host')
    p.add_argument('--remote-path')
    
    # Recording
    p.add_argument('--record-screen', action='store_true', help="DEPRECATED: Use --recording-mode screen-record")
    p.add_argument('--recording-mode', type=str, default='none', choices=['none', 'headless', 'screen-record'],
                  help="Recording mode: none (display only), screen-record (ffmpeg x11grab), headless (no display, post-process video)")

    # Detection params passed to tiling_pipeline
    p.add_argument('--iou-threshold', type=float, default=0.3, help="IOU threshold")
    p.add_argument('--nms-score-threshold', type=float, default=None, help="NMS Score threshold")
    p.add_argument('--min-overlap', type=float, default=0.1, help="Minimum overlap ratio")
    
    return p.parse_args()

def main():
    args = parse_args()
    tiles = [tuple(map(int, t.split('x'))) for t in args.tiles]
    
    detection_args = {
        'iou_threshold': args.iou_threshold,
        'nms_score_threshold': args.nms_score_threshold,
        'min_overlap': args.min_overlap
    }

    # Map legacy record_screen to recording_mode if necessary
    recording_mode = args.recording_mode
    if args.record_screen and recording_mode == 'none':
        recording_mode = 'screen-record'

    run_experiments(
        args.models,
        tiles,
        args.input,
        args.extra_args,
        args.remote,
        args.remote_user,
        args.remote_host,
        args.remote_path,
        detection_args,
        recording_mode
    )

if __name__ == "__main__":
    main()