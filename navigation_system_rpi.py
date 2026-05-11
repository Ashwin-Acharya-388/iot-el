"""
navigation_system_rpi.py
========================
Main navigation system for the Raspberry Pi 4B head-mounted assistant.

Architecture:
  Camera → ONNX Inference → ByteTrack → Temporal Smoothing
  → Clear Path Finder → Sliding Window Majority Vote → Voice Command

Target: 5-6 FPS @ 320×320 on RPi 4B with QAT INT8 model
Press Ctrl+C to stop gracefully.

Usage:
    python navigation_system_rpi.py
    python navigation_system_rpi.py --model ./models/yolov8n_qat.onnx
    python navigation_system_rpi.py --camera 0 --conf 0.45 --debug
"""

import os
import sys
import time
import signal
import argparse
import threading
import collections
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

DEFAULT_MODEL   = Path("./models/yolov8n_qat.onnx")
CAMERA_IDX      = 0
FRAME_SIZE      = (320, 320)
CONF_THRESHOLD  = 0.45
IOU_THRESHOLD   = 0.45
VOTE_WINDOW     = 5        # Sliding window for majority voting (frames)
COOLDOWN_SEC    = 1.5      # Minimum time between identical voice commands
DANGER_ZONE_Y   = 0.6      # Obstacles in bottom 60% of frame = closer/more dangerous

# Zone definitions (normalized x-coordinates in 320px frame)
# |--LEFT--|--CENTER--|--RIGHT--|
ZONE_LEFT_MAX   = 0.33
ZONE_RIGHT_MIN  = 0.67

# Danger multipliers: center obstacles are more dangerous than edges
ZONE_DANGER = {
    "left":   1.0,
    "center": 2.5,    # Center obstacles block the direct path
    "right":  1.0,
}

# Class ID → display name (must match dataset.yaml)
CLASS_NAMES = [
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
    "traffic light", "traffic sign", "pole", "wall", "fence", "curb", "sidewalk", "road",
]

# High-danger classes (close proximity = immediate stop/redirect)
HIGH_DANGER_CLASSES = {0, 1, 2, 3, 4, 5, 6, 7}  # Persons, vehicles


# ──────────────────────────────────────────────
# SIMPLE BYTETRACK-STYLE TRACKER
# ──────────────────────────────────────────────

class SimpleTracker:
    """
    Lightweight ByteTrack-inspired tracker.
    
    WHY TRACKING:
    - Frame-to-frame detection fluctuates (false positives, misses).
    - Tracking assigns stable IDs so we know object X from frame 5 is
      the same as object X in frame 6, enabling temporal smoothing.
    - ByteTrack uses two-level matching: high-confidence detections first,
      then low-confidence detections for objects that went missing briefly.
    
    This simplified version uses IoU matching with a Kalman-lite approach.
    For production, install the 'supervision' library which includes
    ByteTrack natively: sv.ByteTracker()
    """

    def __init__(self, max_age: int = 5, min_hits: int = 2, iou_thresh: float = 0.3):
        self.max_age    = max_age      # How many frames to keep a lost track
        self.min_hits   = min_hits     # Min detections to confirm a track
        self.iou_thresh = iou_thresh
        self._next_id   = 1
        self._tracks    = {}           # track_id → TrackState

    def _iou(self, box_a, box_b) -> float:
        """Compute IoU between two [x1,y1,x2,y2] boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        area_a = (ax2-ax1)*(ay2-ay1)
        area_b = (bx2-bx1)*(by2-by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections: list) -> list:
        """
        detections: list of (x1,y1,x2,y2,conf,class_id)
        Returns:   list of (x1,y1,x2,y2,conf,class_id,track_id)
        """
        if not detections:
            # Age out all tracks
            dead = [tid for tid, t in self._tracks.items() if t["age"] > self.max_age]
            for tid in dead:
                del self._tracks[tid]
            for tid in self._tracks:
                self._tracks[tid]["age"] += 1
            return []

        matched_det    = set()
        matched_track  = set()
        new_tracks     = []

        # Match detections to existing tracks via IoU
        for tid, track in self._tracks.items():
            best_iou  = self.iou_thresh
            best_didx = None
            for didx, det in enumerate(detections):
                if didx in matched_det:
                    continue
                iou = self._iou(track["box"], det[:4])
                if iou > best_iou:
                    best_iou  = iou
                    best_didx = didx

            if best_didx is not None:
                det = detections[best_didx]
                track["box"]   = det[:4]
                track["conf"]  = det[4]
                track["cls"]   = det[5]
                track["age"]   = 0
                track["hits"] += 1
                matched_det.add(best_didx)
                matched_track.add(tid)

        # Age unmatched tracks
        for tid in list(self._tracks.keys()):
            if tid not in matched_track:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self.max_age:
                    del self._tracks[tid]

        # Create new tracks for unmatched detections
        for didx, det in enumerate(detections):
            if didx not in matched_det:
                self._tracks[self._next_id] = {
                    "box":  det[:4],
                    "conf": det[4],
                    "cls":  det[5],
                    "age":  0,
                    "hits": 1,
                }
                self._next_id += 1

        # Return confirmed tracks
        out = []
        for tid, track in self._tracks.items():
            if track["hits"] >= self.min_hits and track["age"] == 0:
                x1, y1, x2, y2 = track["box"]
                out.append((x1, y1, x2, y2, track["conf"], track["cls"], tid))

        return out


# ──────────────────────────────────────────────
# TEMPORAL SMOOTHING
# ──────────────────────────────────────────────

class TemporalSmoother:
    """
    Average detection confidences across N consecutive frames.
    
    WHY: Single-frame detections are noisy. Averaging over a window
    ensures a person or car is not ignored because it was missed in
    one frame (false negative) or incorrectly detected (false positive).
    
    This mirrors the temporal smoothing used in the DISHA paper.
    """

    def __init__(self, window: int = 5):
        self.window  = window
        self._buffer = collections.deque(maxlen=window)

    def update(self, detections: list) -> list:
        """Add new detections and return smoothed set."""
        self._buffer.append(detections)
        if len(self._buffer) < 2:
            return detections
        # Flatten all frames and deduplicate by spatial proximity
        # Simple approach: return union of detections from all frames
        # weighted by recency (most recent frame has full weight)
        all_dets = []
        for frame_dets in self._buffer:
            all_dets.extend(frame_dets)
        return self._suppress_duplicates(all_dets)

    def _suppress_duplicates(self, dets: list, iou_thresh=0.5) -> list:
        """Non-maximum suppression across temporal buffer."""
        if not dets:
            return []
        dets  = sorted(dets, key=lambda d: d[4], reverse=True)  # sort by conf
        keep  = []
        used  = [False] * len(dets)
        for i, d in enumerate(dets):
            if used[i]:
                continue
            keep.append(d)
            for j in range(i+1, len(dets)):
                if not used[j] and self._iou(d[:4], dets[j][:4]) > iou_thresh:
                    used[j] = True
        return keep

    @staticmethod
    def _iou(a, b):
        ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
        ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
        inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
        return inter/ua if ua>0 else 0.0


# ──────────────────────────────────────────────
# CLEAR PATH FINDER
# ──────────────────────────────────────────────

COMMANDS = ["Left", "Slight Left", "Forward", "Slight Right", "Right", "Stop"]

def find_safe_direction(tracked_dets: list, frame_w: int = 320, frame_h: int = 320) -> str:
    """
    Identify the safest walking direction from current obstacle positions.
    
    ALGORITHM:
    ──────────────────────────────────────────────────────────────
    1. Score each of three horizontal zones (Left, Center, Right)
       based on obstacle density and class danger.
    
    2. Zone score = Σ (conf × area_fraction × danger_multiplier × height_weight)
       height_weight: obstacles lower in frame (closer) score higher.
    
    3. The zone with the LOWEST danger score is the safest to walk toward.
    
    4. Map safe zone → direction command:
         Left  safe → "Right"    (steer away from danger on the right)
         Center safe → "Forward"
         Right safe → "Left"
    
    5. If ALL zones have danger score above STOP threshold → "Stop"
    ──────────────────────────────────────────────────────────────
    """
    STOP_THRESHOLD = 0.60    # If safe zone still scores above this → Stop

    # Zone danger accumulator
    zone_scores = {"left": 0.0, "center": 0.0, "right": 0.0}

    for det in tracked_dets:
        # Support both 6-tuple (no track ID) and 7-tuple (with track ID)
        x1, y1, x2, y2, conf = det[0], det[1], det[2], det[3], det[4]
        cls_id = int(det[5])

        # Normalize coordinates
        cx = (x1 + x2) / 2 / frame_w
        cy = (y1 + y2) / 2 / frame_h
        w  = (x2 - x1) / frame_w
        h  = (y2 - y1) / frame_h

        # Area fraction of frame
        area_frac = w * h

        # Height weight: closer objects (lower in frame) are more dangerous
        # cy=1.0 at bottom (closest), cy=0 at top
        height_weight = max(0.2, cy)

        # Class-based danger multiplier
        class_danger = 1.5 if cls_id in HIGH_DANGER_CLASSES else 1.0

        danger = conf * area_frac * height_weight * class_danger

        # Assign to zone based on horizontal center
        if cx < ZONE_LEFT_MAX:
            zone_scores["left"]   += danger * ZONE_DANGER["left"]
        elif cx > ZONE_RIGHT_MIN:
            zone_scores["right"]  += danger * ZONE_DANGER["right"]
        else:
            zone_scores["center"] += danger * ZONE_DANGER["center"]

    # Determine safest zone
    safest_zone = min(zone_scores, key=zone_scores.get)
    safe_score  = zone_scores[safest_zone]

    # Check for stop condition
    if safe_score > STOP_THRESHOLD:
        return "Stop"

    # Determine gradient: how much worse are adjacent zones?
    left_s   = zone_scores["left"]
    center_s = zone_scores["center"]
    right_s  = zone_scores["right"]

    if safest_zone == "center":
        return "Forward"
    elif safest_zone == "left":
        # Safe path is on the LEFT side → steer LEFT
        # Check how far off center we need to go
        gap = right_s - left_s     # how much more dangerous is right vs left?
        return "Left" if gap > 0.2 else "Slight Left"
    else:
        # Safe path is on the RIGHT side → steer RIGHT
        gap = left_s - right_s
        return "Right" if gap > 0.2 else "Slight Right"


# ──────────────────────────────────────────────
# SLIDING WINDOW MAJORITY VOTE
# ──────────────────────────────────────────────

class DirectionVoter:
    """
    Smooths directional decisions across the last N frames using majority voting.
    
    WHY: Single-frame path decisions can flip due to detection noise.
    Majority voting over last 5 frames ensures stable, consistent guidance
    without confusing the user with rapidly alternating commands.
    """

    def __init__(self, window: int = VOTE_WINDOW):
        self._window  = window
        self._history = collections.deque(maxlen=window)

    def vote(self, direction: str) -> str:
        """Add direction to history and return majority vote."""
        self._history.append(direction)
        counts = collections.Counter(self._history)
        return counts.most_common(1)[0][0]

    @property
    def stable(self) -> bool:
        """True when the window is full (system has warmed up)."""
        return len(self._history) >= self._window


# ──────────────────────────────────────────────
# ONNX INFERENCE ENGINE
# ──────────────────────────────────────────────

class ONNXInference:
    """Wraps ONNX Runtime session for YOLOv8 inference."""

    def __init__(self, model_path: str):
        import onnxruntime as ort

        # Prefer CPU execution provider on RPi
        providers = ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4     # RPi 4B has 4 cores
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session    = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape= self.session.get_inputs()[0].shape   # [1,3,320,320]
        print(f"  ✓ ONNX model loaded: {Path(model_path).name}")
        print(f"    Input: {self.input_name} {self.input_shape}")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR frame → NCHW float32 tensor normalized to [0,1]."""
        import cv2
        img  = cv2.resize(frame, (320, 320))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img  = img.astype(np.float32) / 255.0
        img  = np.transpose(img, (2, 0, 1))
        img  = np.expand_dims(img, 0)
        return np.ascontiguousarray(img)

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        """Run inference and return raw output."""
        outputs = self.session.run(None, {self.input_name: tensor})
        return outputs[0]

    def postprocess(self, output: np.ndarray,
                    conf_thresh: float = 0.45,
                    iou_thresh:  float = 0.45) -> list:
        """
        Parse YOLOv8 ONNX output to list of (x1,y1,x2,y2,conf,class_id).
        
        YOLOv8 ONNX output shape: [1, num_classes+4, num_anchors]
        Layout: [cx, cy, w, h, class_0_conf, class_1_conf, ...]
        """
        # output shape: [1, 84, 8400] for standard YOLOv8
        preds = output[0]               # shape: [4+nc, anchors]
        preds = preds.T                 # → [anchors, 4+nc]

        boxes       = preds[:, :4]      # cx,cy,w,h (normalized 0-1)
        class_probs = preds[:, 4:]      # per-class confidence

        # Get max class and confidence per box
        class_ids   = np.argmax(class_probs, axis=1)
        confs       = class_probs[np.arange(len(class_ids)), class_ids]

        # Filter by confidence
        mask   = confs >= conf_thresh
        boxes  = boxes[mask]
        confs  = confs[mask]
        cids   = class_ids[mask]

        if len(boxes) == 0:
            return []

        # Convert cx,cy,w,h → x1,y1,x2,y2 (pixel coords in 320×320)
        cx, cy, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        x1 = (cx - w/2) * 320
        y1 = (cy - h/2) * 320
        x2 = (cx + w/2) * 320
        y2 = (cy + h/2) * 320

        dets = list(zip(x1, y1, x2, y2, confs, cids))

        # NMS
        dets = self._nms(dets, iou_thresh)
        return dets

    @staticmethod
    def _nms(dets: list, iou_thresh: float) -> list:
        if not dets:
            return []
        dets = sorted(dets, key=lambda d: d[4], reverse=True)
        keep = []
        used = [False]*len(dets)
        for i, d in enumerate(dets):
            if used[i]:
                continue
            keep.append(d)
            for j in range(i+1, len(dets)):
                if not used[j] and ONNXInference._iou(d[:4], dets[j][:4]) > iou_thresh:
                    used[j] = True
        return keep

    @staticmethod
    def _iou(a, b):
        ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
        ix1=max(ax1,bx1);iy1=max(ay1,by1);ix2=min(ax2,bx2);iy2=min(ay2,by2)
        inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
        return inter/ua if ua>0 else 0.0


# ──────────────────────────────────────────────
# MAIN NAVIGATION LOOP
# ──────────────────────────────────────────────

class NavigationSystem:

    def __init__(self, model_path: str, camera_idx: int = 0,
                 conf: float = CONF_THRESHOLD, debug: bool = False):
        self.conf     = conf
        self.debug    = debug
        self._running = False

        print("\n  Initializing Navigation System...")
        self.engine   = ONNXInference(model_path)
        self.tracker  = SimpleTracker(max_age=5, min_hits=2)
        self.smoother = TemporalSmoother(window=3)
        self.voter    = DirectionVoter(window=VOTE_WINDOW)

        # Import voice here to avoid slow startup
        from voice_commands import VoiceCommands
        self.voice    = VoiceCommands()

        # Camera
        import cv2
        self.cap = cv2.VideoCapture(camera_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)    # Reduce buffer for lower latency
        if not self.cap.isOpened():
            print(f"  [ERROR] Could not open camera {camera_idx}")
            sys.exit(1)

        print("  ✓ Camera opened.")
        print(f"  ✓ Confidence threshold: {conf}")
        print(f"  ✓ Debug mode: {'ON' if debug else 'OFF'}")

        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, *_):
        print("\n  Shutting down gracefully...")
        self._running = False

    def run(self):
        """Main capture-infer-command loop."""
        import cv2

        self._running = True
        frame_times   = collections.deque(maxlen=30)
        prev_command  = None

        print("\n  ▶ Navigation system running. Press Ctrl+C to stop.\n")

        while self._running:
            t_frame_start = time.perf_counter()

            # ── Capture ────────────────────────────
            ret, frame = self.cap.read()
            if not ret:
                print("  [WARN] Camera read failed; retrying...")
                time.sleep(0.05)
                continue

            # ── Inference ──────────────────────────
            tensor = self.engine.preprocess(frame)
            t0     = time.perf_counter()
            output = self.engine.infer(tensor)
            lat_ms = (time.perf_counter() - t0) * 1000
            dets   = self.engine.postprocess(output, self.conf)

            # ── Tracking ───────────────────────────
            tracked = self.tracker.update(dets)

            # ── Temporal smoothing ─────────────────
            smoothed = self.smoother.update(tracked)

            # ── Path finding ───────────────────────
            raw_direction = find_safe_direction(smoothed)

            # ── Majority voting ────────────────────
            voted_direction = self.voter.vote(raw_direction)

            # ── Voice output ───────────────────────
            if self.voter.stable and voted_direction != prev_command:
                self.voice.speak(voted_direction)
                prev_command = voted_direction

            # ── FPS tracking ───────────────────────
            t_frame_end = time.perf_counter()
            frame_times.append(t_frame_end - t_frame_start)
            fps = 1.0 / np.mean(frame_times) if frame_times else 0

            # ── Debug output ───────────────────────
            if self.debug:
                n_obs = len(smoothed)
                print(f"  FPS: {fps:4.1f} | Lat: {lat_ms:5.1f}ms | "
                      f"Obs: {n_obs:2d} | Raw: {raw_direction:<12} | "
                      f"Vote: {voted_direction}")

            # ── Optional display (disable on headless RPi) ─
            if self.debug and not self._is_headless():
                self._draw_debug(frame, smoothed, voted_direction, fps, lat_ms)

        # Cleanup
        self.cap.release()
        if not self._is_headless():
            import cv2
            cv2.destroyAllWindows()
        self.voice.shutdown()
        print("  ✓ Shutdown complete.")

    def _is_headless(self) -> bool:
        return os.environ.get("DISPLAY", "") == "" and not self.debug

    def _draw_debug(self, frame, dets, direction, fps, lat_ms):
        """Draw bounding boxes and overlay on frame for visual debugging."""
        import cv2
        vis = frame.copy()

        # Zone dividers
        cv2.line(vis, (int(320*ZONE_LEFT_MAX), 0),  (int(320*ZONE_LEFT_MAX), 320),  (200,200,0), 1)
        cv2.line(vis, (int(320*ZONE_RIGHT_MIN), 0), (int(320*ZONE_RIGHT_MIN), 320), (200,200,0), 1)

        for det in dets:
            x1,y1,x2,y2 = int(det[0]),int(det[1]),int(det[2]),int(det[3])
            conf  = det[4]
            cls   = int(det[5])
            label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
            color = (0,0,255) if cls in HIGH_DANGER_CLASSES else (0,200,100)
            cv2.rectangle(vis, (x1,y1), (x2,y2), color, 1)
            cv2.putText(vis, f"{label} {conf:.2f}", (x1, max(0, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Direction overlay
        cv2.putText(vis, f"→ {direction}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(vis, f"FPS:{fps:.1f} Lat:{lat_ms:.0f}ms", (10, 310),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

        cv2.imshow("Navigation Debug", vis)
        cv2.waitKey(1)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Head-Mounted Navigation System (RPi 4B)")
    parser.add_argument("--model",  default=str(DEFAULT_MODEL), help="ONNX model path")
    parser.add_argument("--camera", type=int, default=CAMERA_IDX)
    parser.add_argument("--conf",   type=float, default=CONF_THRESHOLD)
    parser.add_argument("--debug",  action="store_true", help="Show debug output and video")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"[ERROR] Model not found: {args.model}")
        print("Transfer models from laptop: rsync -av ./models/ pi@<IP>:~/navigation/models/")
        sys.exit(1)

    try:
        import onnxruntime
    except ImportError:
        print("[ERROR] pip install onnxruntime")
        sys.exit(1)

    nav = NavigationSystem(
        model_path = args.model,
        camera_idx = args.camera,
        conf       = args.conf,
        debug      = args.debug,
    )
    nav.run()


if __name__ == "__main__":
    main()
