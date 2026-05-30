"""
test_accuracy_laptop.py
=======================
Evaluates all three models on the Cityscapes validation set and prints
a detailed comparison table with per-class accuracy for navigation-critical
classes.

Models evaluated:
  1. FP32 Baseline    (yolov8n_cityscapes.pt)
  2. INT8 Quantized   (yolov8n_int8.onnx)   — after quantization, pre-QAT
  3. QAT Fine-tuned   (yolov8n_qat.onnx)    — after QAT recovery

Usage:
    python test_accuracy_laptop.py
    python test_accuracy_laptop.py --model-dir ./models --data ./data/yolo_cityscapes/dataset.yaml
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

MODEL_DIR    = Path("./models")
DATASET_YAML = "./data/yolo_cityscapes/dataset.yaml"
REPORT_PATH  = Path("./models/accuracy_report.json")

CRITICAL_CLASSES = [
    "person", "rider", "car", "truck", "bus", "motorcycle", "bicycle",
    "traffic light", "traffic sign", "pole",
]

MODELS = {
    "FP32 Baseline":        "yolov8n_cityscapes.pt",
    "INT8 Quantized":       "yolov8n_int8.onnx",
    "QAT Fine-Tuned":       "yolov8n_qat.onnx",
}


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def banner(msg: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}\n")


def check_models(model_dir: Path) -> dict:
    """Find available model files."""
    found = {}
    for label, filename in MODELS.items():
        path = model_dir / filename
        if path.exists():
            found[label] = path
            print(f"  ✓ {label}: {path}")
        else:
            print(f"  ✗ {label}: {path} (not found)")
    return found


# ──────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────

def evaluate_model(label: str, model_path: Path, data_yaml: str) -> dict:
    """
    Run validation on a model (.pt or .onnx) and return metrics.
    Ultralytics handles both formats transparently.
    """
    from ultralytics import YOLO

    print(f"\n  Evaluating: {label} ({model_path.name})")
    start = time.time()

    try:
        model   = YOLO(str(model_path))
        metrics = model.val(
            data    = data_yaml,
            imgsz   = 320,
            device  = "cpu",      # Keep CPU for fair comparison across all models
            batch   = 1,
            verbose = False,
            plots   = False,
        )
        elapsed = time.time() - start

        names = model.names  # class index → name

        # Build per-class map50 dictionary
        per_class = {}
        if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
            for idx, cls_map in enumerate(metrics.box.maps):
                cls_name = names.get(idx, str(idx))
                per_class[cls_name] = round(float(cls_map), 4)

        return {
            "label":       label,
            "path":        str(model_path),
            "map50":       round(float(metrics.box.map50), 4),
            "map5095":     round(float(metrics.box.map), 4),
            "precision":   round(float(metrics.box.mp), 4),
            "recall":      round(float(metrics.box.mr), 4),
            "per_class":   per_class,
            "eval_time_s": round(elapsed, 1),
            "error":       None,
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"  [ERROR] {label} evaluation failed: {e}")
        return {
            "label":       label,
            "path":        str(model_path),
            "map50":       None,
            "map5095":     None,
            "precision":   None,
            "recall":      None,
            "per_class":   {},
            "eval_time_s": round(elapsed, 1),
            "error":       str(e),
        }


# ──────────────────────────────────────────────
# REPORTING
# ──────────────────────────────────────────────

def print_summary_table(results: list) -> None:
    """Print overall mAP comparison table."""
    print("\n")
    banner("OVERALL ACCURACY COMPARISON")
    header = f"  {'Model':<22} {'mAP50':>8}  {'mAP50-95':>10}  {'Precision':>10}  {'Recall':>8}  {'Δ mAP50 vs FP32':>16}"
    print(header)
    print(f"  {'-'*90}")

    baseline_map50 = None
    for r in results:
        if r["map50"] is None:
            print(f"  {r['label']:<22}  {'ERROR':>8}")
            continue

        m50 = r["map50"]
        m95 = r["map5095"]
        pre = r["precision"]
        rec = r["recall"]

        if baseline_map50 is None:
            baseline_map50 = m50
            delta_str = "  (baseline)"
        else:
            delta = m50 - baseline_map50
            sign  = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.4f} ({sign}{delta*100:.2f}%)"

        print(f"  {r['label']:<22} {m50:>8.4f}  {m95:>10.4f}  {pre:>10.4f}  {rec:>8.4f}  {delta_str:>16}")


def print_per_class_table(results: list) -> None:
    """Print per-class mAP50 for navigation-critical classes across all models."""
    banner("PER-CLASS mAP50 (Navigation-Critical Classes)")

    labels = [r["label"] for r in results if r["map50"] is not None]
    col_w  = 12

    # Header
    header = f"  {'Class':<22}"
    for lbl in labels:
        header += f"  {lbl[:col_w]:>{col_w}}"
    print(header)
    print(f"  {'-'*(22 + (col_w+2)*len(labels))}")

    for cls_name in CRITICAL_CLASSES:
        row = f"  {cls_name:<22}"
        for r in results:
            if r["map50"] is None:
                row += f"  {'ERR':>{col_w}}"
                continue
            val = r["per_class"].get(cls_name, None)
            if val is None:
                row += f"  {'—':>{col_w}}"
            else:
                # Highlight if QAT recovered vs INT8
                row += f"  {val:>{col_w}.4f}"
        print(row)

    # Recovery summary
    if len(results) >= 3 and all(r["map50"] is not None for r in results[:3]):
        fp32  = results[0]
        int8  = results[1]
        qat   = results[2]

        print(f"\n  QAT Recovery Summary (per class):")
        print(f"  {'Class':<22}  {'INT8 drop':>10}  {'QAT recovery':>14}  {'Net loss':>10}")
        print(f"  {'-'*65}")
        for cls_name in CRITICAL_CLASSES:
            fp32_v = fp32["per_class"].get(cls_name)
            int8_v = int8["per_class"].get(cls_name)
            qat_v  = qat ["per_class"].get(cls_name)
            if None in (fp32_v, int8_v, qat_v):
                continue
            drop     = int8_v - fp32_v
            recovery = qat_v  - int8_v
            net      = qat_v  - fp32_v
            print(f"  {cls_name:<22}  {drop:>+10.4f}  {recovery:>+14.4f}  {net:>+10.4f}")


def print_size_comparison(model_dir: Path) -> None:
    """Print model file sizes."""
    banner("MODEL SIZE COMPARISON")
    for label, filename in MODELS.items():
        path = model_dir / filename
        if path.exists():
            size_mb = path.stat().st_size / 1024**2
            print(f"  {label:<22}  {size_mb:>7.2f} MB  ({path.name})")


def print_rpi_performance_estimate() -> None:
    """Print expected RPi 4B performance based on known benchmarks."""
    banner("ESTIMATED RPi 4B PERFORMANCE (from DISHA methodology)")
    rows = [
        ("FP32 Baseline",   "2-3",   "330-500"),
        ("INT8 Quantized",  "4-5",   "200-250"),
        ("QAT Fine-Tuned",  "5-6 ✓", "150-170 ✓"),
    ]
    print(f"  {'Model':<22}  {'FPS':>8}  {'Latency (ms)':>14}")
    print(f"  {'-'*50}")
    for name, fps, lat in rows:
        print(f"  {name:<22}  {fps:>8}  {lat:>14}")
    print(f"\n  Target: 5-6 FPS @ 320×320 → ✓ achievable with QAT model\n")


# ──────────────────────────────────────────────
# LIVE WEBCAM TEST
# ──────────────────────────────────────────────

def test_with_webcam(model_path: Path, duration_seconds: int = 30) -> None:
    """Quick live webcam inference test on laptop."""
    import cv2
    from ultralytics import YOLO

    banner(f"LIVE WEBCAM TEST ({duration_seconds}s)")
    print(f"  Model: {model_path.name}")
    print(f"  Press 'q' to quit early.\n")

    model  = YOLO(str(model_path))
    cap    = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [WARN] Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

    start      = time.time()
    frame_cnt  = 0
    latencies  = []

    while time.time() - start < duration_seconds:
        ret, frame = cap.read()
        if not ret:
            break

        t0      = time.time()
        results = model(frame, imgsz=320, verbose=False)
        latency = (time.time() - t0) * 1000
        latencies.append(latency)
        frame_cnt += 1

        # Annotate and show
        annotated = results[0].plot()
        fps_str   = f"FPS: {1000/latency:.1f}  Lat: {latency:.0f}ms"
        cv2.putText(annotated, fps_str, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Laptop Test", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if latencies:
        print(f"\n  Frames:   {frame_cnt}")
        print(f"  Avg FPS:  {1000/np.mean(latencies):.1f}")
        print(f"  Avg lat:  {np.mean(latencies):.1f}ms")
        print(f"  Min lat:  {np.min(latencies):.1f}ms")
        print(f"  Max lat:  {np.max(latencies):.1f}ms")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--data",      default=DATASET_YAML)
    parser.add_argument("--webcam",    action="store_true", help="Also run live webcam test")
    parser.add_argument("--webcam-duration", type=int, default=30)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    banner("HEAD-MOUNTED NAVIGATION — ACCURACY EVALUATION")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics")
        sys.exit(1)

    # Find available models
    available = check_models(model_dir)
    if not available:
        print("\n[ERROR] No models found. Run the training pipeline first:")
        print("  python train_baseline.py")
        print("  python quantize_and_qat.py")
        sys.exit(1)

    # Check dataset
    if not Path(args.data).exists():
        print(f"\n[ERROR] Dataset YAML not found: {args.data}")
        sys.exit(1)

    # Evaluate each available model
    results = []
    for label, model_path in available.items():
        r = evaluate_model(label, model_path, args.data)
        results.append(r)

    # Print tables
    print_summary_table(results)
    print_per_class_table(results)
    print_size_comparison(model_dir)
    print_rpi_performance_estimate()

    # Save report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Full report saved → {REPORT_PATH}")

    # Optional webcam test
    if args.webcam:
        best_model = model_dir / "yolov8n_qat.onnx"
        if best_model.exists():
            test_with_webcam(best_model, args.webcam_duration)
        else:
            print(f"\n  [WARN] QAT model not found for webcam test.")

    banner("NEXT STEPS")
    print("  1. Review the comparison table above")
    print("  2. Transfer models to Raspberry Pi:")
    print("       rsync -av ./models/ pi@<RPi_IP>:~/navigation/models/")
    print("  3. Run on RPi:")
    print("       python navigation_system_rpi.py\n")


if __name__ == "__main__":
    main()
