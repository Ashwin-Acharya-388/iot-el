"""
navigation_freespace_rpi.py
===========================
Navigation system for RPi 4B using binary free-space segmentation.

Instead of detecting individual obstacles (bounding boxes), this system
detects WALKABLE GROUND using a segmentation model. Direction commands
are derived from where free space exists in the camera frame.

Architecture:
    Camera → ONNX Segmentation → Binary Mask → Zone Analyzer
    → Direction Command → Majority Vote → Voice Output

Target: 5-10 FPS @ 320×320 on RPi 4B with INT8 segmentation model
Press Ctrl+C to stop gracefully.

Usage:
    python navigation_freespace_rpi.py
    python navigation_freespace_rpi.py --model ./models/freespace_int8.onnx
    python navigation_freespace_rpi.py --debug --camera 0
"""

import os
import sys
import time
import signal
import argparse
import collections
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # Will fail with clear error at runtime


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

DEFAULT_MODEL   = Path("./models/freespace_int8.onnx")
CAMERA_IDX      = 0
FRAME_SIZE      = 320
VOTE_WINDOW     = 5        # Sliding window for majority voting
COOLDOWN_SEC    = 1.5      # Minimum time between identical voice commands

# Zone layout for direction command derivation
# The bottom half of the frame is divided into 5 vertical zones:
#
#   ┌──────┬──────┬──────┬──────┬──────┐
#   │ Far  │Slight│      │Slight│ Far  │
#   │ Left │ Left │Center│Right │Right │
#   │  0   │  1   │  2   │  3   │  4   │
#   └──────┴──────┴──────┴──────┴──────┘
#
NUM_ZONES = 5
ZONE_WEIGHTS = [-2.0, -1.0, 0.0, 1.0, 2.0]  # Left-to-right directional bias

# Thresholds for direction decisions
STOP_THRESHOLD      = 0.10   # <10% total walkable → Stop
STRONG_TURN_THRESH  = 1.0    # |center_of_mass| > 1.0 → hard turn
SLIGHT_TURN_THRESH  = 0.25   # |center_of_mass| > 0.25 → slight turn

# ImageNet normalization (must match training preprocessing)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ──────────────────────────────────────────────
# COMMANDS
# ──────────────────────────────────────────────

COMMANDS = ["Left", "Slight Left", "Forward", "Slight Right", "Right", "Stop"]


# ──────────────────────────────────────────────
# SEGMENTATION INFERENCE ENGINE
# ──────────────────────────────────────────────

class FreespaceInference:
    """
    Wraps ONNX Runtime for binary free-space segmentation.
    
    Input:  320×320 RGB image (normalized with ImageNet stats)
    Output: 320×320 binary mask (0=obstacle, 1=walkable)
    """

    def __init__(self, model_path: str):
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4     # RPi 4B has 4 cores
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        print(f"  ✓ ONNX segmentation model loaded: {Path(model_path).name}")
        print(f"    Input: {self.input_name} {self.input_shape}")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        BGR camera frame → normalized NCHW float32 tensor.
        
        Preprocessing chain:
            1. Resize to 320×320
            2. BGR → RGB
            3. Scale to [0, 1]
            4. Normalize with ImageNet mean/std
            5. Transpose to NCHW
            6. Add batch dimension
        """
        img = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, 0)          # → NCHW
        return np.ascontiguousarray(img)

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        """
        Run inference and return binary mask.
        
        ONNX output shape: [1, 2, H, W] (2-class logits)
        Returns: [H, W] binary mask (0=obstacle, 1=walkable)
        """
        outputs = self.session.run(None, {self.input_name: tensor})
        logits = outputs[0]  # [1, 2, H, W]

        # Argmax to get class predictions
        mask = np.argmax(logits[0], axis=0)  # [H, W], values in {0, 1}
        return mask.astype(np.uint8)


# ──────────────────────────────────────────────
# ZONE-BASED DIRECTION ANALYZER
# ──────────────────────────────────────────────

def mask_to_direction(binary_mask: np.ndarray) -> Tuple[str, dict]:
    """
    Convert binary segmentation mask to a navigation direction command.
    
    ALGORITHM:
    ──────────────────────────────────────────────────────────────
    1. Take the BOTTOM HALF of the mask (ground directly ahead).
    
    2. Divide into 5 vertical zones (Far Left → Far Right).
    
    3. Compute walkable fraction for each zone:
       zone_score[i] = (walkable pixels in zone i) / (total pixels in zone i)
    
    4. Compute weighted center of mass:
       CoM = Σ(weight[i] × zone_score[i]) / Σ(zone_score[i])
       where weights = [-2, -1, 0, +1, +2]
    
    5. Map CoM to direction:
       CoM < -1.0    → "Left"         (strong pull left)
       CoM < -0.25   → "Slight Left"  (gentle pull left)
       CoM > +1.0    → "Right"        (strong pull right)
       CoM > +0.25   → "Slight Right" (gentle pull right)
       else          → "Forward"      (centered free space)
    
    6. If total walkable fraction < 10% → "Stop"
    ──────────────────────────────────────────────────────────────
    
    Args:
        binary_mask: [H, W] numpy array, 1=walkable, 0=obstacle
    
    Returns:
        (direction_str, zone_info_dict)
    """
    H, W = binary_mask.shape

    # Only analyze bottom half (ground ahead, not sky)
    bottom_half = binary_mask[H // 2:, :]
    bh, bw = bottom_half.shape

    # Divide into zones
    zone_width = bw // NUM_ZONES
    zone_scores = []

    for i in range(NUM_ZONES):
        x_start = i * zone_width
        x_end = (i + 1) * zone_width if i < NUM_ZONES - 1 else bw
        zone = bottom_half[:, x_start:x_end]
        score = np.sum(zone) / max(zone.size, 1)
        zone_scores.append(score)

    # Total walkable fraction
    total_free = sum(zone_scores) / NUM_ZONES

    # Zone info for debugging
    zone_info = {
        "zones": zone_scores,
        "total_free": total_free,
        "zone_names": ["Far Left", "Slight Left", "Center", "Slight Right", "Far Right"],
    }

    # Stop condition: too little walkable space
    if total_free < STOP_THRESHOLD:
        zone_info["direction"] = "Stop"
        return "Stop", zone_info

    # Weighted center of mass
    weighted_sum = sum(w * s for w, s in zip(ZONE_WEIGHTS, zone_scores))
    score_sum = sum(zone_scores)
    center_of_mass = weighted_sum / max(score_sum, 1e-6)
    zone_info["center_of_mass"] = center_of_mass

    # Map to direction
    if center_of_mass < -STRONG_TURN_THRESH:
        direction = "Left"
    elif center_of_mass < -SLIGHT_TURN_THRESH:
        direction = "Slight Left"
    elif center_of_mass > STRONG_TURN_THRESH:
        direction = "Right"
    elif center_of_mass > SLIGHT_TURN_THRESH:
        direction = "Slight Right"
    else:
        direction = "Forward"

    zone_info["direction"] = direction
    return direction, zone_info


# ──────────────────────────────────────────────
# TEMPORAL SMOOTHING (reuse from existing system)
# ──────────────────────────────────────────────

class MaskSmoother:
    """
    Temporal smoothing for segmentation masks.
    
    Averages the last N binary masks to reduce flickering.
    A pixel is considered walkable only if it was walkable
    in the majority of recent frames.
    """

    def __init__(self, window: int = 3):
        self.window = window
        self._buffer = collections.deque(maxlen=window)

    def update(self, mask: np.ndarray) -> np.ndarray:
        """Add new mask and return temporally smoothed mask."""
        self._buffer.append(mask.astype(np.float32))
        if len(self._buffer) == 1:
            return mask

        # Average masks → threshold at 0.5 (majority vote per pixel)
        avg = np.mean(np.stack(self._buffer), axis=0)
        return (avg >= 0.5).astype(np.uint8)


# ──────────────────────────────────────────────
# SLIDING WINDOW MAJORITY VOTE (same as detection system)
# ──────────────────────────────────────────────

class DirectionVoter:
    """
    Smooths directional decisions across the last N frames.
    Prevents confusing the user with rapidly alternating commands.
    """

    def __init__(self, window: int = VOTE_WINDOW):
        self._window = window
        self._history = collections.deque(maxlen=window)

    def vote(self, direction: str) -> str:
        self._history.append(direction)
        counts = collections.Counter(self._history)
        return counts.most_common(1)[0][0]

    @property
    def stable(self) -> bool:
        return len(self._history) >= self._window


# ──────────────────────────────────────────────
# MAIN NAVIGATION LOOP
# ──────────────────────────────────────────────

class FreespaceNavigation:
    """
    Main navigation system using free-space segmentation.
    
    Pipeline per frame:
        1. Capture camera frame
        2. Run segmentation inference → binary mask
        3. Temporal smoothing on mask
        4. Zone analysis → raw direction
        5. Majority vote → stable direction
        6. Voice output (with cooldown)
    """

    def __init__(self, model_path: str, camera_idx: int = 0, debug: bool = False):
        self.debug = debug
        self._running = False

        print("\n  Initializing Free-Space Navigation System...")

        # Segmentation engine
        self.engine = FreespaceInference(model_path)

        # Smoothing & voting
        self.mask_smoother = MaskSmoother(window=3)
        self.voter = DirectionVoter(window=VOTE_WINDOW)

        # Voice output (reuse existing VoiceCommands)
        from voice_commands import VoiceCommands
        self.voice = VoiceCommands()

        # Camera
        self.cap = cv2.VideoCapture(camera_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            print(f"  [ERROR] Could not open camera {camera_idx}")
            sys.exit(1)

        print("  ✓ Camera opened")
        print(f"  ✓ Debug mode: {'ON' if debug else 'OFF'}")

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, *_):
        print("\n  Shutting down gracefully...")
        self._running = False

    def run(self):
        """Main capture → infer → command loop."""


        self._running = True
        frame_times = collections.deque(maxlen=30)
        prev_command = None

        print("\n  ▶ Free-space navigation running. Press Ctrl+C to stop.\n")

        while self._running:
            t_start = time.perf_counter()

            # ── Capture ──
            ret, frame = self.cap.read()
            if not ret:
                print("  [WARN] Camera read failed; retrying...")
                time.sleep(0.05)
                continue

            # ── Segmentation ──
            tensor = self.engine.preprocess(frame)
            t_infer = time.perf_counter()
            mask = self.engine.infer(tensor)
            lat_ms = (time.perf_counter() - t_infer) * 1000

            # ── Temporal smoothing ──
            smoothed_mask = self.mask_smoother.update(mask)

            # ── Zone analysis → direction ──
            raw_direction, zone_info = mask_to_direction(smoothed_mask)

            # ── Majority vote ──
            voted_direction = self.voter.vote(raw_direction)

            # ── Voice output ──
            if self.voter.stable and voted_direction != prev_command:
                # Include walkable percentage for awareness
                pct = zone_info["total_free"] * 100
                if voted_direction == "Stop":
                    self.voice.speak("Stop. Path blocked.")
                elif pct < 30:
                    self.voice.speak(f"{voted_direction}. Narrow path.")
                else:
                    self.voice.speak(voted_direction)
                prev_command = voted_direction

            # ── FPS tracking ──
            t_end = time.perf_counter()
            frame_times.append(t_end - t_start)
            fps = len(frame_times) / sum(frame_times) if frame_times else 0

            # ── Debug output ──
            if self.debug:
                zones_str = " | ".join(
                    f"{name}: {100*score:.0f}%"
                    for name, score in zip(zone_info["zone_names"], zone_info["zones"])
                )
                print(f"  FPS: {fps:4.1f} | Lat: {lat_ms:5.1f}ms | "
                      f"Free: {100*zone_info['total_free']:.0f}% | "
                      f"Raw: {raw_direction:<12} | Vote: {voted_direction}")
                print(f"    Zones: {zones_str}")

            # ── Visual debug ──
            if self.debug and not self._is_headless():
                self._draw_debug(frame, smoothed_mask, voted_direction, zone_info, fps, lat_ms)

        # Cleanup
        self.cap.release()
        if not self._is_headless():
            cv2.destroyAllWindows()
        self.voice.shutdown()
        print("  ✓ Shutdown complete.")

    def _is_headless(self) -> bool:
        return os.environ.get("DISPLAY", "") == ""

    def _draw_debug(self, frame, mask, direction, zone_info, fps, lat_ms):
        """Draw segmentation overlay for visual debugging."""


        vis = frame.copy()

        # Resize mask to match frame if needed
        if mask.shape[:2] != vis.shape[:2]:
            mask = cv2.resize(mask, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Green overlay on walkable areas
        walkable = mask > 0
        vis[walkable] = (vis[walkable] * 0.5 + np.array([0, 200, 0], dtype=np.uint8) * 0.5).astype(np.uint8)

        # Red tint on obstacles (subtle)
        non_walkable = ~walkable
        vis[non_walkable] = (vis[non_walkable] * 0.8 + np.array([0, 0, 100], dtype=np.uint8) * 0.2).astype(np.uint8)

        # Zone dividers (bottom half)
        h = vis.shape[0]
        zone_width = vis.shape[1] // NUM_ZONES
        for i in range(1, NUM_ZONES):
            x = i * zone_width
            cv2.line(vis, (x, h // 2), (x, h), (200, 200, 0), 1)

        # Zone scores
        for i, (score, name) in enumerate(zip(zone_info["zones"], zone_info["zone_names"])):
            x = i * zone_width + 5
            cv2.putText(vis, f"{100*score:.0f}%", (x, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Direction overlay
        color = (0, 255, 0) if direction != "Stop" else (0, 0, 255)
        cv2.putText(vis, f"→ {direction}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(vis, f"FPS:{fps:.1f} Lat:{lat_ms:.0f}ms Free:{100*zone_info['total_free']:.0f}%",
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        cv2.imshow("Free-Space Navigation", vis)
        cv2.waitKey(1)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Free-Space Navigation System (RPi 4B)")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="ONNX segmentation model path")
    parser.add_argument("--camera", type=int, default=CAMERA_IDX)
    parser.add_argument("--debug", action="store_true", help="Show debug output and video overlay")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"[ERROR] Model not found: {args.model}")
        print("Train and export the model first:")
        print("  python train_freespace.py")
        print("Then transfer to RPi:")
        print("  scp ./models/freespace_int8.onnx pi@<RPI_IP>:~/navigation/models/")
        sys.exit(1)

    try:
        import onnxruntime
    except ImportError:
        print("[ERROR] pip install onnxruntime")
        sys.exit(1)

    nav = FreespaceNavigation(
        model_path=args.model,
        camera_idx=args.camera,
        debug=args.debug,
    )
    nav.run()


if __name__ == "__main__":
    main()
