"""
quantize_and_qat.py
===================
Quantizes the Cityscapes-fine-tuned YOLOv8n model to INT8 and then
performs Post-Quantization Fine-Tuning (QAT) to recover accuracy.

Pipeline:
  1. Load yolov8n_cityscapes.pt (FP32 baseline)
  2. Export to ONNX (FP32)
  3. Calibrate INT8 quantization using ~200-300 Cityscapes images
  4. Save quantized model (INT8, with accuracy drop)
  5. Perform QAT fine-tuning loop to recover the ~2-5% mAP drop
  6. Save final yolov8n_qat.onnx
  7. Compare mAP before quantization → after INT8 → after QAT

Usage:
    python quantize_and_qat.py
    python quantize_and_qat.py --model ./models/yolov8n_cityscapes.pt
    python quantize_and_qat.py --qat-epochs 10 --calib-images 300
"""

import os
import sys
import argparse
import time
import random
import json
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

DEFAULT_MODEL       = Path("./models/yolov8n_cityscapes.pt")
ONNX_FP32           = Path("./models/yolov8n_fp32.onnx")
ONNX_INT8           = Path("./models/yolov8n_int8.onnx")
ONNX_QAT            = Path("./models/yolov8n_qat.onnx")
DATASET_YAML        = "./data/yolo_cityscapes/dataset.yaml"
CALIB_DIR           = Path("./data/yolo_cityscapes/images/val")
RESULTS_JSON        = Path("./models/qat_comparison.json")

INPUT_SIZE          = (320, 320)
CALIB_N_IMAGES      = 300
QAT_EPOCHS          = 10
QAT_LR              = 1e-4           # Low LR for fine-tuning only


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def banner(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


def load_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        os.system("pip install opencv-python-headless")
        import cv2
        return cv2


def get_calib_images(calib_dir: Path, n: int) -> list:
    """Collect calibration image paths."""
    exts  = [".jpg", ".jpeg", ".png"]
    imgs  = [p for p in calib_dir.rglob("*") if p.suffix.lower() in exts]
    random.seed(42)
    random.shuffle(imgs)
    selected = imgs[:n]
    print(f"  Calibration images: {len(selected)} / {len(imgs)} available")
    return selected


def preprocess_image(img_path: Path, size=(320, 320)) -> np.ndarray:
    """Load and preprocess one image for ONNX inference."""
    cv2 = load_cv2()
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))      # HWC → CHW
    img = np.expand_dims(img, 0)            # CHW → NCHW
    return img


# ──────────────────────────────────────────────
# STEP 1: EXPORT TO ONNX (FP32)
# ──────────────────────────────────────────────

def export_to_onnx(model_pt: Path, onnx_out: Path) -> bool:
    """Export PyTorch YOLOv8 model to ONNX (FP32)."""
    from ultralytics import YOLO

    if onnx_out.exists():
        print(f"  ✓ FP32 ONNX already exists: {onnx_out}")
        return True

    banner("STEP 1: EXPORT TO ONNX (FP32)")
    model = YOLO(str(model_pt))
    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    model.export(
        format  = "onnx",
        imgsz   = INPUT_SIZE[0],
        opset   = 13,
        simplify= True,
        dynamic = False,      # Fixed batch=1 for RPi
    )

    # ultralytics saves next to .pt by default; move it
    default_onnx = model_pt.with_suffix(".onnx")
    if default_onnx.exists() and default_onnx != onnx_out:
        import shutil
        shutil.move(str(default_onnx), str(onnx_out))

    if onnx_out.exists():
        size_mb = onnx_out.stat().st_size / 1024**2
        print(f"  ✓ FP32 ONNX saved → {onnx_out} ({size_mb:.1f} MB)")
        return True

    print(f"  [ERROR] ONNX export failed.")
    return False


# ──────────────────────────────────────────────
# STEP 2: INT8 STATIC QUANTIZATION (ONNX Runtime)
# ──────────────────────────────────────────────

class CalibrationDataReader:
    """
    ONNX Runtime calibration data reader.
    Feeds calibration images to the quantizer to compute activation ranges.
    
    WHY: Static quantization needs to observe real data to choose the
    best int8 scale factors (min/max or percentile of activation histograms).
    Using 200-300 diverse street images gives a good statistical sample.
    """

    def __init__(self, image_paths: list, input_name: str = "images"):
        self.paths      = image_paths
        self.input_name = input_name
        self.idx        = 0

    def get_next(self):
        if self.idx >= len(self.paths):
            return None
        img = preprocess_image(self.paths[self.idx])
        self.idx += 1
        if img is None:
            return self.get_next()
        return {self.input_name: img}

    def rewind(self):
        self.idx = 0


def quantize_int8(onnx_fp32: Path, onnx_int8: Path, calib_images: list) -> bool:
    """
    Apply ONNX Runtime static INT8 quantization with calibration.

    WHY INT8:
    - FP32 → INT8 reduces model size ~4×
    - Reduces compute per operation ~4× (integer math vs float)
    - Enables 5-6 FPS on RPi 4B vs 2-3 FPS for FP32
    - Cost: ~2-5% mAP drop (recovered by QAT in next step)
    """
    try:
        from onnxruntime.quantization import (
            quantize_static,
            QuantFormat,
            QuantType,
            CalibrationMethod,
        )
    except ImportError:
        print("  Installing onnxruntime-tools...")
        os.system("pip install onnxruntime onnxruntime-tools")
        from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationMethod

    if onnx_int8.exists():
        print(f"  ✓ INT8 model already exists: {onnx_int8}")
        return True

    banner("STEP 2: INT8 STATIC QUANTIZATION")
    print(f"  Input:  {onnx_fp32}")
    print(f"  Output: {onnx_int8}")
    print(f"  Calibration images: {len(calib_images)}")

    # First determine input node name
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_fp32), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print(f"  Model input name: '{input_name}'")

    reader = CalibrationDataReader(calib_images, input_name=input_name)

    try:
        quantize_static(
            model_input             = str(onnx_fp32),
            model_output            = str(onnx_int8),
            calibration_data_reader = reader,
            quant_format            = QuantFormat.QDQ,   # Quantize-Dequantize nodes
            activation_type         = QuantType.QInt8,
            weight_type             = QuantType.QInt8,
            calibrate_method        = CalibrationMethod.MinMax,
            per_channel             = False,              # Simpler, more RPi-friendly
            reduce_range            = False,
        )
        size_mb = onnx_int8.stat().st_size / 1024**2
        print(f"  ✓ INT8 model saved → {onnx_int8} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  [ERROR] INT8 quantization failed: {e}")
        return False


# ──────────────────────────────────────────────
# STEP 3: QUANTIZATION AWARE TRAINING (QAT)
# ──────────────────────────────────────────────

def run_qat_ultralytics(baseline_pt: Path, qat_onnx_out: Path,
                         data_yaml: str, epochs: int, lr: float, device: str) -> bool:
    """
    Quantization Aware Training using Ultralytics + PyTorch.

    HOW QAT WORKS:
    ─────────────────────────────────────────────────────────
    During normal training, weights and activations are FP32.
    
    QAT inserts "fake quantization" nodes that:
      1. Round values to INT8 precision (forward pass)
      2. Use Straight-Through Estimator (STE) for gradients (backward pass)
         → gradients flow as if quantization didn't happen
    
    This teaches the model to be robust to INT8 rounding DURING training,
    so the final quantized model loses less accuracy.
    
    DISHA paper results: QAT recovered ~2-4% mAP vs plain INT8 quantization.
    ─────────────────────────────────────────────────────────

    Implementation note: Ultralytics v8 doesn't natively support QAT export,
    so we use a two-phase approach:
      Phase A: QAT fine-tuning with torch.quantization fake quant nodes
      Phase B: Convert to full INT8 and export to ONNX
    """
    import torch
    from ultralytics import YOLO

    banner("STEP 3: QUANTIZATION AWARE TRAINING (QAT)")
    print(f"  Epochs: {epochs}  |  LR: {lr}")
    print(f"  Device: {device}")
    print("""
  [INFO] QAT methodology (DISHA approach):
    • Load the Cityscapes fine-tuned FP32 model
    • Insert fake-quantization observers in Conv, BN, Linear layers
    • Fine-tune for {epochs} additional epochs with a low learning rate
    • The model learns to minimize loss DESPITE INT8 rounding noise
    • After QAT, convert and export to proper INT8 ONNX
  """.format(epochs=epochs))

    # ── Phase A: Fake-quant fine-tuning ──────────────────────────
    model = YOLO(str(baseline_pt))

    # Access the underlying PyTorch model
    torch_model = model.model
    torch_model.train()

    # Prepare model for QAT
    torch_model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    try:
        torch.quantization.prepare_qat(torch_model, inplace=True)
        print("  ✓ Fake quantization nodes inserted (QAT mode active).")
    except Exception as e:
        print(f"  [WARN] torch prepare_qat failed ({e}); using Ultralytics fine-tune instead.")
        # Fallback: standard fine-tune with lower LR (still helps recover accuracy)

    # Fine-tune with QAT-aware settings
    qat_run_dir = Path("./runs/qat_finetune")
    result = model.train(
        data          = data_yaml,
        epochs        = epochs,
        imgsz         = INPUT_SIZE[0],
        batch         = 8,
        device        = device,
        lr0           = lr,
        lrf           = 0.001,
        warmup_epochs = 1,
        patience      = epochs,           # Don't stop early during QAT
        project       = str(qat_run_dir.parent),
        name          = qat_run_dir.name,
        save          = True,
        plots         = False,
        verbose       = False,
    )

    # Get QAT-fine-tuned weights
    qat_best = qat_run_dir / "weights" / "best.pt"
    for candidate in qat_run_dir.parent.rglob("best.pt"):
        qat_best = candidate
        break

    if not qat_best.exists():
        print("  [ERROR] QAT fine-tune did not produce best.pt")
        return False

    # ── Phase B: Convert to INT8 ONNX ────────────────────────────
    print(f"\n  Converting QAT model to INT8 ONNX...")
    qat_model = YOLO(str(qat_best))
    qat_model.export(
        format   = "onnx",
        imgsz    = INPUT_SIZE[0],
        opset    = 13,
        simplify = True,
        dynamic  = False,
        int8     = True,           # Use Ultralytics INT8 export path
    )

    # Locate exported file
    default_onnx = qat_best.with_suffix(".onnx")
    if default_onnx.exists():
        import shutil
        shutil.move(str(default_onnx), str(qat_onnx_out))
    elif qat_onnx_out.exists():
        pass
    else:
        # If INT8 export not available in this ultralytics version, do standard
        print("  [FALLBACK] INT8 ONNX export not available; exporting FP32 QAT model.")
        qat_model.export(format="onnx", imgsz=INPUT_SIZE[0], opset=13, simplify=True)
        if default_onnx.exists():
            shutil.move(str(default_onnx), str(qat_onnx_out))
        else:
            # Quantize the QAT model's FP32 ONNX to INT8 via ONNX Runtime
            fp32_tmp = qat_onnx_out.parent / "qat_fp32_tmp.onnx"
            shutil.copy(str(default_onnx), str(fp32_tmp))
            calib_images = get_calib_images(CALIB_DIR, 100)
            quantize_int8(fp32_tmp, qat_onnx_out, calib_images)
            fp32_tmp.unlink(missing_ok=True)

    if qat_onnx_out.exists():
        size_mb = qat_onnx_out.stat().st_size / 1024**2
        print(f"  ✓ QAT model saved → {qat_onnx_out} ({size_mb:.1f} MB)")
        return True

    print("  [ERROR] QAT ONNX not found after export.")
    return False


# ──────────────────────────────────────────────
# STEP 4: COMPARE mAP ACROSS ALL THREE MODELS
# ──────────────────────────────────────────────

def benchmark_onnx_map(onnx_path: Path, data_yaml: str) -> dict:
    """
    Run mAP evaluation on an ONNX model via Ultralytics ONNX validator.
    Returns dict with map50 and map5095.
    """
    from ultralytics import YOLO

    print(f"  Evaluating: {onnx_path.name}")
    try:
        model   = YOLO(str(onnx_path))
        metrics = model.val(data=data_yaml, imgsz=320, device="cpu", verbose=False)
        return {
            "model":    onnx_path.name,
            "map50":    round(float(metrics.box.map50), 4),
            "map5095":  round(float(metrics.box.map), 4),
        }
    except Exception as e:
        print(f"  [WARN] Could not evaluate {onnx_path.name}: {e}")
        return {"model": onnx_path.name, "map50": None, "map5095": None}


def compare_models(fp32_pt: Path, int8_onnx: Path, qat_onnx: Path, data_yaml: str) -> None:
    """Print accuracy comparison table across all three models."""
    banner("MODEL ACCURACY COMPARISON")

    rows = []

    # FP32 baseline
    from ultralytics import YOLO
    try:
        m = YOLO(str(fp32_pt))
        metrics = m.val(data=data_yaml, imgsz=320, verbose=False)
        rows.append({
            "model": "FP32 Baseline (yolov8n_cityscapes.pt)",
            "map50": round(float(metrics.box.map50), 4),
            "map5095": round(float(metrics.box.map), 4),
        })
    except Exception as e:
        print(f"  [WARN] FP32 eval failed: {e}")
        rows.append({"model": "FP32 Baseline", "map50": "N/A", "map5095": "N/A"})

    # INT8
    if int8_onnx.exists():
        rows.append(benchmark_onnx_map(int8_onnx, data_yaml))

    # QAT
    if qat_onnx.exists():
        rows.append(benchmark_onnx_map(qat_onnx, data_yaml))

    # Print table
    print(f"\n  {'Model':<45} {'mAP50':>8}  {'mAP50-95':>10}  {'Δ mAP50':>10}")
    print(f"  {'-'*80}")
    baseline_map50 = None
    for row in rows:
        m50 = row["map50"]
        m95 = row["map5095"]
        delta = ""
        if baseline_map50 is None and isinstance(m50, float):
            baseline_map50 = m50
        elif isinstance(m50, float) and baseline_map50 is not None:
            delta = f"{m50 - baseline_map50:+.4f}"
        print(f"  {row['model']:<45} {str(m50):>8}  {str(m95):>10}  {delta:>10}")

    # Save results
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  ✓ Results saved → {RESULTS_JSON}")

    if len(rows) >= 3 and all(isinstance(r["map50"], float) for r in rows[:3]):
        int8_drop    = rows[0]["map50"] - rows[1]["map50"]
        qat_recovery = rows[2]["map50"] - rows[1]["map50"]
        print(f"\n  INT8 accuracy drop:     {int8_drop:.4f} ({int8_drop*100:.2f}%)")
        print(f"  QAT accuracy recovery:  {qat_recovery:.4f} ({qat_recovery*100:.2f}%)")
        print(f"  Net loss after QAT:     {rows[0]['map50']-rows[2]['map50']:.4f}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default=str(DEFAULT_MODEL))
    parser.add_argument("--data",         default=DATASET_YAML)
    parser.add_argument("--calib-dir",    default=str(CALIB_DIR))
    parser.add_argument("--calib-images", type=int, default=CALIB_N_IMAGES)
    parser.add_argument("--qat-epochs",   type=int, default=QAT_EPOCHS)
    parser.add_argument("--qat-lr",       type=float, default=QAT_LR)
    parser.add_argument("--device",       default="auto")
    parser.add_argument("--skip-int8",    action="store_true")
    parser.add_argument("--skip-qat",     action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    args = parser.parse_args()

    import torch
    if args.device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model_pt = Path(args.model)
    if not model_pt.exists():
        print(f"[ERROR] Model not found: {model_pt}")
        print(f"Run:  python train_baseline.py  first.")
        sys.exit(1)

    Path("./models").mkdir(exist_ok=True)

    # Step 1: FP32 ONNX export
    if not export_to_onnx(model_pt, ONNX_FP32):
        sys.exit(1)

    # Step 2: INT8 quantization
    if not args.skip_int8:
        calib_imgs = get_calib_images(Path(args.calib_dir), args.calib_images)
        if not calib_imgs:
            print(f"[ERROR] No calibration images in {args.calib_dir}")
            sys.exit(1)
        quantize_int8(ONNX_FP32, ONNX_INT8, calib_imgs)

    # Step 3: QAT
    if not args.skip_qat:
        run_qat_ultralytics(
            baseline_pt  = model_pt,
            qat_onnx_out = ONNX_QAT,
            data_yaml    = args.data,
            epochs       = args.qat_epochs,
            lr           = args.qat_lr,
            device       = device,
        )

    # Step 4: Compare
    if not args.skip_compare:
        compare_models(model_pt, ONNX_INT8, ONNX_QAT, args.data)

    banner("DONE")
    print(f"  Models saved in ./models/:")
    print(f"    yolov8n_fp32.onnx   — FP32 baseline")
    print(f"    yolov8n_int8.onnx   — INT8 quantized (no QAT)")
    print(f"    yolov8n_qat.onnx    — INT8 + QAT fine-tuning ← use this on RPi")
    print(f"\n  Next step:")
    print(f"    python test_accuracy_laptop.py")
    print(f"    # Then transfer models to Raspberry Pi\n")


if __name__ == "__main__":
    main()
