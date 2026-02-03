import itertools
import subprocess
import argparse
from pathlib import Path
import os
import signal
import time
import re
import sys

def is_wayland():
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"

def get_display_info(force_virtual=False, width=1920, height=1080):
    display = os.environ.get('DISPLAY', ':0')
    
    if not force_virtual:
        try:
            # Check for existing X11 session
            check = subprocess.run(['xdpyinfo'], capture_output=True, text=True, env={**os.environ, 'DISPLAY': display})
            if check.returncode == 0:
                xr = subprocess.run(['xrandr', '--query'], capture_output=True, text=True, check=True)
                for line in xr.stdout.split('\n'):
                    if ' connected' in line:
                        match = re.search(r'(\d+)x(\d+)\+(\d+)\+(\d+)', line)
                        if match:
                            w, h, x, y = map(int, match.groups())
                            return {'display': display, 'width': w, 'height': h, 'x': x, 'y': y, 'virtual': False}
        except Exception:
            pass

    # Setup Xvfb
    xvfb_display = ":99"
    lock = f"/tmp/.X{xvfb_display.strip(':')}-lock"
    if os.path.exists(lock):
        try:
            with open(lock, 'r') as f:
                os.kill(int(f.read().strip()), signal.SIGTERM)
            os.remove(lock)
        except: pass

    print(f"Starting Xvfb: {width}x{height}")
    subprocess.Popen(['Xvfb', xvfb_display, '-screen', '0', f'{width}x{height}x24'])
    os.environ['DISPLAY'] = xvfb_display
    time.sleep(2) 
    return {'display': xvfb_display, 'width': width, 'height': height, 'x': 0, 'y': 0, 'virtual': True}

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
        '-framerate', '60',
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

def run_experiments(models, tiles, input_path, extra_args=None,
                    remote=False, remote_user=None, remote_host=None, remote_path=None,
                    detection_args=None, record_screen=False):

    total = len(models) * len(tiles) * 2
    count = 0

    display_info = get_display_info(force_virtual=record_screen)
    
    print(f"\nSession type: {os.environ.get('XDG_SESSION_TYPE', 'unknown')}")
    print(f"Display: {display_info['display']} {display_info['width']}x{display_info['height']}")
    if is_wayland():
        print("Screen recording: OFF (Wayland)")
    elif record_screen:
        print("Screen recording: ON (X11)")
    else:
        print("Screen recording: OFF (User disabled)")
    print()

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
        base_fps = min(5, max(1, base_fps))  # Ensure FPS is between 1 and 30
        
        fps_values = sorted(list(set([base_fps, max(1, int(0.9 * base_fps))])), reverse=True)

        for fps in fps_values:
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

            if extra_args:
                cmd.extend(extra_args)

            print(f"\n=== Experiment {count}/{total} ===")
            print(" ".join(cmd))

            with open(log_file, "w") as lf:
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)

            time.sleep(8)

            ffmpeg_proc = None
            screen_file = None
            if record_screen:
                ffmpeg_proc, screen_file = start_screen_recording(output_file, display_info)

            result = proc.wait()

            stop_screen_recording(ffmpeg_proc, screen_file)

            if result == 0:
                print("Experiment completed successfully.")
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
    p.add_argument('--input', required=True)
    p.add_argument('--extra-args', nargs=argparse.REMAINDER)
    p.add_argument('--remote', action='store_true')
    p.add_argument('--remote-user')
    p.add_argument('--remote-host')
    p.add_argument('--remote-path')
    
    # Recording
    p.add_argument('--record-screen', action='store_true', help="Enable screen recording")

    # Detection params passed to tiling_pipeline
    p.add_argument('--iou-threshold', type=float, default=0.3, help="IOU threshold")
    p.add_argument('--nms-score-threshold', type=float, default=None, help="NMS Score threshold")
    
    return p.parse_args()

def main():
    args = parse_args()
    tiles = [tuple(map(int, t.split('x'))) for t in args.tiles]
    
    detection_args = {
        'iou_threshold': args.iou_threshold,
        'nms_score_threshold': args.nms_score_threshold
    }

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
        args.record_screen
    )

if __name__ == "__main__":
    main()