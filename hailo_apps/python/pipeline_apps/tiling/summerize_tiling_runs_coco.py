#!/home/hailo/Desktop/hailo-apps/venv/bin/python3
import os
import glob
import re
import json
import argparse
import numpy as np
import sys
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# --- Configuration ---
LOG_DIR = "/home/hailo/Desktop/hailo-apps/hailo_apps/python/pipeline_apps/tiling/results"
LOG_PATTERN = "*_tiling_*x*_*fps_skateboard-chase*.log"
ANNOTATION_DIR = "/home/hailo/Desktop/hailo-apps/hailo_apps/python/pipeline_apps/detection/VisDrone2019-VID-val/annotations"

VIDEO_SEQUENCE_MAP = {"dataset_video": "uav0000086_00000_v"}
VIDEO_RESOLUTION = (1344, 756) 
SOURCE_FPS = 30.0

# Jitter removed
# JITTER_OFFSETS = [0]

# Evaluation Categories (VisDrone)
EVAL_CATEGORIES = {
    1: "pedestrian", 2: "people", 3: "bicycle", 4: "car",
    5: "van", 6: "truck", 7: "tricycle", 8: "awning-tricycle",
    9: "bus", 10: "motor"
}

# Model Label -> Evaluation ID (VisDrone IDs)
# Mapping standard COCO labels to VisDrone classes
LABEL_TO_ID = {
    "person": 1,      # Maps to pedestrian
    "bicycle": 3,
    "car": 4,
    "motorcycle": 10, # Maps to motor
    "bus": 9,
    "truck": 6,
    # Add more mappings if model outputs other labels
    "van": 5,
    "tricycle": 7,
    "awning-tricycle": 8,
    "motor": 10,
    "pedestrian": 1,
    "people": 2
}

# VisDrone GT ID -> Evaluation ID
# Identity mapping to keep all classes separate for evaluation
VISDRONE_TO_CUSTOM = {k: k for k in EVAL_CATEGORIES}

# Classes to ignore during evaluation (crowd/ignore regions)
# 0: ignored regions, 11: others
IGNORE_VISDRONE_CLASSES = {0, 11}

# Grouping for final table reporting
REPORT_GROUPS = {
    "Person": [1, 2],       # pedestrian, people
    "Vehicle": [4, 5, 6, 9], # car, van, truck, bus
    "Cycle": [3, 10, 7, 8], # bicycle, motor, tricycle, awning-tricycle
    "Overall": list(EVAL_CATEGORIES.keys())
}

GT_CACHE = {}

def parse_filename(filename):
    name = filename[:-4]
    parts = name.split('_tiling_')
    if len(parts) < 2: return None
    model_name = parts[0]
    rest = '_tiling_'.join(parts[1:])
    match = re.match(r"(\d+)x(\d+)_(\d+)fps_(.*)", rest)
    if not match: return None
    tx, ty, fps, video_source = match.groups()
    if video_source.endswith('_detections'): return None
    return {'model_name': model_name, 'tiling': f"{tx}x{ty}", 'input_video': video_source, 'frame_rate': int(fps), 'filename': filename}

def analyze_log_file(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines: return 0.0
        vals = [float(l.split(',')[2]) for l in reversed(lines) if len(l.split(',')) >= 3][:10]
        return sum(vals) / len(vals) if vals else 0.0
    except: return 0.0

def load_ground_truth(video_source):
    if video_source in GT_CACHE: return GT_CACHE[video_source]
    
    # Try to find sequence name in map, otherwise assume video_source is the sequence name
    seq_name = VIDEO_SEQUENCE_MAP.get(video_source, video_source)
    
    annotation_file = os.path.join(ANNOTATION_DIR, f"{seq_name}.txt")
    if not os.path.exists(annotation_file): return None, None
    gt_entries, all_frames = [], set()
    with open(annotation_file, 'r') as f:
        for line in f:
            p = [x.strip() for x in line.split(',')]
            if len(p) < 8: continue
            f_idx = int(p[0])
            all_frames.add(f_idx)
            gt_entries.append({'frame': f_idx, 'bbox': [int(p[2]), int(p[3]), int(p[4]), int(p[5])], 'vis_cat': int(p[7])})
    GT_CACHE[video_source] = (gt_entries, all_frames)
    return gt_entries, all_frames

def create_coco_gt_dict(gt_entries, valid_frames):
    coco_gt = {
        "images": [{"id": f, "width": VIDEO_RESOLUTION[0], "height": VIDEO_RESOLUTION[1]} for f in sorted(list(valid_frames))],
        "annotations": [],
        "categories": [{"id": k, "name": v} for k, v in EVAL_CATEGORIES.items()]
    }
    ann_id = 1
    for entry in gt_entries:
        if entry['frame'] not in valid_frames: continue
        vis_cat, bbox = entry['vis_cat'], entry['bbox']
        if vis_cat in VISDRONE_TO_CUSTOM:
            coco_gt["annotations"].append({"id": ann_id, "image_id": entry['frame'], "category_id": VISDRONE_TO_CUSTOM[vis_cat], "bbox": bbox, "area": bbox[2]*bbox[3], "iscrowd": 0})
            ann_id += 1
        elif vis_cat in IGNORE_VISDRONE_CLASSES:
            for cat_id in EVAL_CATEGORIES.keys():
                coco_gt["annotations"].append({"id": ann_id, "image_id": entry['frame'], "category_id": cat_id, "bbox": bbox, "area": bbox[2]*bbox[3], "iscrowd": 1})
                ann_id += 1
    return coco_gt

def parse_detections_raw(filepath):
    """Extract detections and their raw timestamps first."""
    raw_data = []
    curr_ts = None
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "Timestamp:" in line:
                match = re.search(r'Timestamp:\s*([\d.]+)s', line)
                if match: curr_ts = float(match.group(1))
            elif line.startswith("Detection:") and curr_ts is not None:
                parts = line.split("Confidence:")
                label = parts[0].replace("Detection:", "").strip()
                if label not in LABEL_TO_ID: continue
                conf = float(parts[1].strip())
                if i + 1 < len(lines) and lines[i+1].strip().startswith("Bbox:"):
                    bp = lines[i+1].strip().replace("Bbox:", "").split(",")
                    bd = {p.split("=")[0].strip(): float(p.split("=")[1].strip()) for p in bp}
                    w, h = VIDEO_RESOLUTION
                    raw_data.append({"ts": curr_ts, "cat": LABEL_TO_ID[label], "bbox": [bd['x_min']*w, bd['y_min']*h, bd['w']*w, bd['h']*h], "score": conf})
    except: pass
    return raw_data

def run_coco_eval(detections, gt_dict):
    try:
        sys.stdout = open(os.devnull, 'w')
        cocoGt = COCO()
        cocoGt.dataset = gt_dict
        cocoGt.createIndex()
        cocoDt = cocoGt.loadRes(detections)
        eval = COCOeval(cocoGt, cocoDt, 'bbox')
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        
        # Calculate per-category results
        cat_results = {}
        for cat_id in EVAL_CATEGORIES:
            # Stats layout: [AP50:95, AP50, AP75, AP_S, AP_M, AP_L, AR1, AR10, AR100, AR_S, AR_M, AR_L]
            # To get per-category, we could rely on accumulate() but extracting is complex.
            # Easier to run evaluate() for single category if needed, but slow.
            # Alternatively, access eval.eval['precision'] which is [TxRxKxAxM]
            # T=10 (IoU thr), R=101 (rec thr), K=num_cats, A=4 (area rng), M=3 (max dets)
            # We want mean over T (for AP) or specific T (for AP50), mean over R, specific K, A=0 (all), M=2 (100 dets)
            
            # Index of category in eval.params.catIds
            if cat_id in eval.params.catIds:
                k = eval.params.catIds.index(cat_id)
                # AP@50 (idx 0 for T=0.50 ? No, params.iouThrs starts at 0.5, step 0.05. So index 0 is 0.50)
                # Precision shape: [10, 101, K, 4, 3]
                
                # mAP@50 (T=0)
                p_50 = eval.eval['precision'][0, :, k, 0, 2]
                ap_50 = np.mean(p_50[p_50 > -1]) if np.any(p_50 > -1) else 0.0
                
                # mAP@50:95 (mean over T=0..9)
                p_all = eval.eval['precision'][:, :, k, 0, 2]
                ap_all = np.mean(p_all[p_all > -1]) if np.any(p_all > -1) else 0.0
                
                cat_results[cat_id] = {'m50': round(ap_all, 4), 'm5095': round(ap_all, 4)} # Wait, ap_all is 50:95
                cat_results[cat_id] = {'m50': round(ap_50, 4), 'm5095': round(ap_all, 4)}
            else:
                cat_results[cat_id] = {'m50': 0.0, 'm5095': 0.0}

        sys.stdout = sys.__stdout__
        return round(eval.stats[1], 4), round(eval.stats[0], 4), cat_results
    except Exception as e:
        sys.stdout = sys.__stdout__
        # print(f"Eval error: {e}")
        return 0.0, 0.0, {}

def evaluate_detections(log_file, video_source):
    gt_entries, all_gt_frames = load_ground_truth(video_source)
    if not gt_entries: return "N/A", "N/A", {}
    
    raw_dets = parse_detections_raw(log_file)
    if not raw_dets: return 0.0, 0.0, {}
    
    frame_ts = {f: (f - 1) / SOURCE_FPS for f in all_gt_frames}
    
    coco_dets = []
    for d in raw_dets:
        # Match using timestamp directly (no jitter offset)
        if not frame_ts: continue
        matched_frame = min(frame_ts.keys(), key=lambda k: abs(frame_ts[k] - d['ts']))
        coco_dets.append({"image_id": matched_frame, "category_id": d['cat'], "bbox": d['bbox'], "score": d['score']})
    
    det_frames = set(d['image_id'] for d in coco_dets)
    valid_frames = det_frames.intersection(all_gt_frames)
    if not valid_frames: return 0.0, 0.0, {}
    
    filtered_dets = [d for d in coco_dets if d['image_id'] in valid_frames]
    gt_dict = create_coco_gt_dict(gt_entries, valid_frames)
    
    return run_coco_eval(filtered_dets, gt_dict)

def parse_args():
    parser = argparse.ArgumentParser(description="Summarize tiling runs with COCO evaluation")
    parser.add_argument("--log-dir", default="/home/hailo/Desktop/hailo-apps/hailo_apps/python/pipeline_apps/tiling/results", help="Directory containing log files")
    parser.add_argument("--annotation-dir", default="/home/hailo/Desktop/hailo-apps/hailo_apps/python/pipeline_apps/detection/VisDrone2019-VID-val/annotations", help="Directory containing annotations")
    parser.add_argument("--resolution", default="1344x756", help="Video resolution WxH")
    parser.add_argument("--source-fps", type=float, default=30.0, help="Source video FPS")
    parser.add_argument("--log-pattern", default="*_tiling_*x*_*fps_*.log", help="Pattern to match log files")
    return parser.parse_args()

def main():
    args = parse_args()
    
    global LOG_DIR, ANNOTATION_DIR, VIDEO_RESOLUTION, SOURCE_FPS, LOG_PATTERN
    LOG_DIR = args.log_dir
    ANNOTATION_DIR = args.annotation_dir
    w, h = map(int, args.resolution.split('x'))
    VIDEO_RESOLUTION = (w, h)
    SOURCE_FPS = args.source_fps
    LOG_PATTERN = args.log_pattern

    log_files = glob.glob(os.path.join(LOG_DIR, LOG_PATTERN))
    results = []
    total = len(log_files)
    print(f"Testing {total} logs...")
    
    for idx, log_file in enumerate(log_files, 1):
        filename = os.path.basename(log_file)
        if filename.endswith('_detections.log'): continue
        meta = parse_filename(filename)
        if not meta: continue

        print(f"  [{idx}/{total}] {filename} ... ", end="", flush=True)
        fps = analyze_log_file(log_file)
        print(f"fps={fps:.1f} ", end="", flush=True)
        passed = fps >= (meta['frame_rate'] * 0.95)
        m50, m5095 = 0.0, 0.0
        cat_stats = {}

        if passed:
            det_log = log_file.replace(".log", "_detections.log")
            if os.path.exists(det_log):
                print("mAP ... ", end="", flush=True)
                m50, m5095, cat_stats = evaluate_detections(det_log, meta['input_video'])
                print(f"mAP@50={m50:.3f} ", end="", flush=True)
        print("done.", flush=True)

        results.append({
            **meta, 
            'fps': round(fps, 2), 
            'status': "PASS" if passed else "FAIL", 
            'm50': m50, 
            'm5095': m5095,
            'cat_stats': cat_stats
        })
        
    results.sort(key=lambda x: (x['model_name'], x['tiling'], x['frame_rate']))
    
    # Calculate group stats for each result
    for r in results:
        groups = {}
        if not r['cat_stats']:
            r['groups'] = {g: 0.0 for g in REPORT_GROUPS}
            continue
            
        for g_name, cat_ids in REPORT_GROUPS.items():
            vals = [r['cat_stats'].get(cid, {'m50': 0})['m50'] for cid in cat_ids if cid in r['cat_stats']]
            # Average mAP@50 for the group (macro average)
            groups[g_name] = sum(vals) / len(vals) if vals else 0.0
        r['groups'] = groups

    # Dynamic columns based on REPORT_GROUPS
    group_cols = sorted(REPORT_GROUPS.keys())
    # Prefer specific order: Person, Vehicle, Cycle, ...
    preferred_order = ["Person", "Vehicle", "Cycle"]
    group_cols = [g for g in preferred_order if g in group_cols] + [g for g in group_cols if g not in preferred_order]
    if "Overall" in group_cols: group_cols.remove("Overall") # Overall is roughly m50

    head = ["Model", "Tiling", "FPS", "Status", "mAP@50"] + group_cols
    
    header_fmt = "{:<20} {:<8} {:<8} {:<8} {:<8}" + " {:<10}" * len(group_cols)
    print(f"\n{'='*(52 + 10*len(group_cols))}")
    print(header_fmt.format(head[0], head[1], head[2], head[3], head[4], *head[5:]))
    print(f"{'-'*(52 + 10*len(group_cols))}")
    
    for r in results:
        row_vals = [
            r['model_name'][:20], 
            r['tiling'], 
            f"{r['fps']}/{r['frame_rate']}", 
            r['status'], 
            f"{r['m50']:.3f}"
        ]
        for g in group_cols:
            val = r['groups'].get(g, 0.0)
            row_vals.append(f"{val:.3f}")
            
        print(header_fmt.format(*row_vals))

if __name__ == "__main__":
    main()