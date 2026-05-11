"""
train_baseline.py
=================
Fine-tunes YOLOv8n (pre-trained on COCO) on the Cityscapes dataset
for sidewalk and obstacle navigation.

Optimized for maximum robustness while maintaining RPi deployability.
Trains at 416×416 for richer features, then quantizes to 320×320 for inference.

Run on Kaggle GPU (recommended) or laptop with GPU.

Usage:
    python train_baseline.py
    python train_baseline.py --epochs 120 --device cuda:0
    python train_baseline.py --resume   # Resume interrupted run
    python train_baseline.py --imgsz 416 --batch 32  # Kaggle GPU settings
"""

import os
import sys
import argparse
import time
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

DEFAULT_EPOCHS      = 120         # More epochs for thorough convergence (early stopping will cut short if needed)
DEFAULT_BATCH       = 32          # Use 32 on Kaggle T4/P100; reduce to 16 or 8 if OOM
DEFAULT_IMGSZ       = 416         # Train at 416 for richer feature learning; quantize to 320 for RPi
DEFAULT_LR0         = 1e-3        # Initial learning rate
DEFAULT_LRF         = 0.01        # Final LR fraction (cosine decay)
DEFAULT_WARMUP      = 5           # Longer warmup for stable early gradients
DEFAULT_PATIENCE    = 25          # Generous patience — let the model converge fully

MODEL_BASE          = "yolov8n.pt"
DATASET_YAML        = "./data/yolo_cityscapes/dataset.yaml"
OUTPUT_DIR          = Path("./runs/train_baseline")
FINAL_MODEL         = Path("./models/yolov8n_cityscapes.pt")

# Navigation-critical classes (subset of full class list for focused reporting)
CRITICAL_CLASSES = [
    "person", "rider", "car", "bus", "truck", "motorcycle", "bicycle",
    "traffic light", "traffic sign", "pole", "wall", "fence", "sidewalk",
]


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def banner(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


def check_dataset(yaml_path: str) -> bool:
    """Verify dataset.yaml and key directories exist."""
    p = Path(yaml_path)
    if not p.exists():
        print(f"  [ERROR] Dataset YAML not found: {yaml_path}")
        print(f"  Run:  python download_datasets.py")
        return False
    return True


def detect_device(requested: str) -> str:
    """Auto-detect best available device."""
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"  ✓ CUDA available: {name}")
        return "0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  ✓ Apple Silicon MPS available.")
        return "mps"
    else:
        print("  ⚠ No GPU found — training on CPU (slow, ~10× longer).")
        return "cpu"


def estimate_time(epochs: int, device: str, imgsz: int = 416) -> str:
    """Rough training time estimate."""
    # Scale factor for larger image sizes
    scale = (imgsz / 320) ** 2
    if device == "cpu":
        mins_per_epoch = 20 * scale
    elif device == "mps":
        mins_per_epoch = 5 * scale
    else:
        mins_per_epoch = 1.5 * scale   # ~2.5 min/epoch at 416 on T4
    total = epochs * mins_per_epoch
    return f"~{total/60:.1f} hours" if total > 90 else f"~{int(total)} minutes"


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────

def train(args) -> Path:
    """Run YOLOv8n fine-tuning on Cityscapes."""
    from ultralytics import YOLO

    device = detect_device(args.device)

    banner("BASELINE TRAINING: YOLOv8n → Cityscapes (Optimized)")
    print(f"  Model:    {args.model}")
    print(f"  Dataset:  {args.data}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Batch:    {args.batch}")
    print(f"  ImgSz:    {args.imgsz}×{args.imgsz} (train) → 320×320 (RPi inference)")
    print(f"  Device:   {device}")
    print(f"  Est. time:{estimate_time(args.epochs, device, args.imgsz)}")
    print(f"  Augments: mosaic+mixup+copy_paste+erasing (robust mode)")

    FINAL_MODEL.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    print(f"\n  Starting training...\n")
    start = time.time()

    results = model.train(
        data            = args.data,
        epochs          = args.epochs,
        imgsz           = args.imgsz,
        batch           = args.batch,
        device          = device,
        project         = str(OUTPUT_DIR.parent),
        name            = OUTPUT_DIR.name,
        lr0             = DEFAULT_LR0,
        lrf             = DEFAULT_LRF,
        warmup_epochs   = DEFAULT_WARMUP,
        patience        = DEFAULT_PATIENCE,
        resume          = args.resume,
        cos_lr          = True,          # Cosine LR decay — smoother convergence than step decay

        # ── Robust Augmentation Suite ──────────────────
        # Stronger augmentation = more generalization = fewer false detections on RPi
        hsv_h           = 0.015,
        hsv_s           = 0.7,
        hsv_v           = 0.4,
        degrees         = 0.0,           # No rotation (head-mounted camera is level)
        translate       = 0.15,          # ↑ from 0.1: more spatial diversity
        scale           = 0.5,
        shear           = 0.0,           # No shear for navigation context
        perspective     = 0.0,           # No perspective warp
        fliplr          = 0.5,
        flipud          = 0.0,           # Never flip vertically (sky≠ground)
        mosaic          = 1.0,           # Always mosaic for diverse scenes
        mixup           = 0.15,          # ↑ from 0.1: helps generalize across scenes
        copy_paste      = 0.1,           # Paste objects across images for rare-class boost
        erasing         = 0.2,           # Random erasing for occlusion robustness

        # ── Multi-Scale Training ───────────────────────
        # Randomly varies input size each batch between 0.5× and 1.5× imgsz
        # This makes the model robust to objects at varying distances
        multi_scale     = True,

        # Save settings
        save            = True,
        save_period     = 10,
        plots           = True,
        verbose         = True,
    )

    elapsed = time.time() - start
    print(f"\n  ✓ Training complete in {elapsed/3600:.2f} hours.")

    # Copy best weights to ./models/
    best_weights = OUTPUT_DIR / "weights" / "best.pt"
    if not best_weights.exists():
        # Try nested location
        for candidate in OUTPUT_DIR.parent.rglob("best.pt"):
            best_weights = candidate
            break

    if best_weights.exists():
        import shutil
        shutil.copy2(best_weights, FINAL_MODEL)
        print(f"  ✓ Best model saved → {FINAL_MODEL.resolve()}")
    else:
        print(f"  [WARN] Could not find best.pt; check {OUTPUT_DIR}")

    return FINAL_MODEL


# ──────────────────────────────────────────────
# VALIDATION & REPORTING
# ──────────────────────────────────────────────

def validate_and_report(model_path: Path, data_yaml: str, device: str) -> dict:
    """Run validation and print per-class accuracy table."""
    from ultralytics import YOLO

    banner("VALIDATION RESULTS")
    model = YOLO(str(model_path))
    metrics = model.val(data=data_yaml, imgsz=320, device=device, verbose=False)

    map50    = metrics.box.map50
    map5095  = metrics.box.map

    print(f"  mAP@50:      {map50:.4f}")
    print(f"  mAP@50-95:   {map5095:.4f}")

    # Per-class results
    names = model.names
    print(f"\n  Per-class mAP@50 (navigation-critical classes):")
    print(f"  {'Class':<20} {'mAP50':>8}")
    print(f"  {'-'*30}")

    results = {}
    if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
        for i, (cls_map) in enumerate(metrics.box.maps):
            cls_name = names.get(i, str(i))
            if cls_name in CRITICAL_CLASSES:
                results[cls_name] = float(cls_map)
                print(f"  {cls_name:<20} {cls_map:>8.4f}")

    print(f"\n  Overall mAP50:    {map50:.4f}")
    print(f"  Overall mAP50-95: {map5095:.4f}")

    return {"map50": map50, "map5095": map5095, "per_class": results}


def generate_confusion_matrix(model_path: Path, data_yaml: str, device: str) -> None:
    """Generate and save confusion matrix."""
    from ultralytics import YOLO
    print("\n  Generating confusion matrix...")
    model = YOLO(str(model_path))
    model.val(
        data    = data_yaml,
        imgsz   = 320,
        device  = device,
        plots   = True,
        project = str(OUTPUT_DIR.parent),
        name    = "confusion_matrix",
    )
    print(f"  ✓ Confusion matrix saved in {OUTPUT_DIR.parent}/confusion_matrix/")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n baseline on Cityscapes.")
    parser.add_argument("--model",   default=MODEL_BASE,    help="Base model weights")
    parser.add_argument("--data",    default=DATASET_YAML,  help="Path to dataset.yaml")
    parser.add_argument("--epochs",  type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch",   type=int, default=DEFAULT_BATCH)
    parser.add_argument("--imgsz",   type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--device",  default="auto",         help="cuda:0, mps, cpu, or auto")
    parser.add_argument("--resume",  action="store_true",    help="Resume from last checkpoint")
    parser.add_argument("--no-confusion", action="store_true")
    args = parser.parse_args()

    # Pre-flight checks
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  [ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    if not check_dataset(args.data):
        sys.exit(1)

    banner("HEAD-MOUNTED NAVIGATION — BASELINE TRAINING")

    # Train
    model_path = train(args)

    # Validate
    device = detect_device(args.device)
    metrics = validate_and_report(model_path, args.data, device)

    # Confusion matrix
    if not args.no_confusion:
        try:
            generate_confusion_matrix(model_path, args.data, device)
        except Exception as e:
            print(f"  [WARN] Confusion matrix generation failed: {e}")

    banner("NEXT STEP")
    print(f"  Run quantization and QAT (will compress to 320×320 INT8 for RPi):")
    print(f"    python quantize_and_qat.py --model {model_path}\n")

    return metrics


if __name__ == "__main__":
    main()
