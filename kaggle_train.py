"""
kaggle_train.py
===============
Complete Kaggle GPU training script for the Head-Mounted Navigation Assistant.

This script handles everything end-to-end on Kaggle:
  1. Install dependencies
  2. Locate Cityscapes dataset from Kaggle input
  3. Convert Cityscapes → YOLO format
  4. Train YOLOv8n with optimized augmentations
  5. Run INT8 quantization + QAT
  6. Save all models to /kaggle/working/ for download

HOW TO USE ON KAGGLE:
─────────────────────
1. Create a new Kaggle Notebook (GPU T4 ×2 accelerator)
2. Add Cityscapes as a dataset (see README for upload instructions)
3. Paste this entire script into a single code cell
4. Run the cell — training takes ~1-2 hours on T4

Alternatively, upload this .py file and run:
    !python kaggle_train.py
"""

import os
import sys
import json
import shutil
import random
import time
import zipfile
from pathlib import Path

# ──────────────────────────────────────────────
# STEP 0: INSTALL DEPENDENCIES
# ──────────────────────────────────────────────

def install_deps():
    print("Installing dependencies...")
    os.system("pip install -q ultralytics>=8.2.0 onnx onnxruntime onnxsim opencv-python-headless")
    print("✓ Dependencies installed.\n")

install_deps()

import numpy as np

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

# Kaggle paths
KAGGLE_INPUT  = Path("/kaggle/input")
KAGGLE_WORK   = Path("/kaggle/working")
DATA_ROOT     = KAGGLE_WORK / "data"
YOLO_DIR      = DATA_ROOT / "yolo_cityscapes"
MODELS_DIR    = KAGGLE_WORK / "models"
RUNS_DIR      = KAGGLE_WORK / "runs"

# Training config (optimized for Kaggle T4 GPU)
TRAIN_EPOCHS  = 120
TRAIN_BATCH   = 32       # T4 16GB can handle batch=32 at 416px for YOLOv8n
TRAIN_IMGSZ   = 416      # Train at 416 for richer features
INFER_IMGSZ   = 320      # Quantize/deploy at 320 for RPi

# Cityscapes class mapping
CITYSCAPES_CLASSES = {
    "person": 0, "rider": 1, "car": 2, "truck": 3, "bus": 4,
    "train": 5, "motorcycle": 6, "bicycle": 7, "traffic light": 8,
    "traffic sign": 9, "pole": 10, "wall": 11, "fence": 12,
    "curb": 13, "sidewalk": 14, "road": 15,
}
YOLO_CLASS_NAMES = list(CITYSCAPES_CLASSES.keys())


# ──────────────────────────────────────────────
# STEP 1: FIND CITYSCAPES DATA (zips OR pre-extracted)
# ──────────────────────────────────────────────

def find_cityscapes_data():
    """
    Locate Cityscapes data from Kaggle dataset input.
    Kaggle auto-extracts uploaded zips, so we check for:
      1. Pre-extracted directories (leftImg8bit/, gtFine/)
      2. Zip files as fallback
    Returns: (img_root, ann_root) — exact paths to the directories.
    """
    print("="*60)
    print("  STEP 1: LOCATING CITYSCAPES DATASET")
    print("="*60)

    # Debug: show what's in /kaggle/input/ (recursive, 3 levels deep)
    print("\n  Scanning /kaggle/input/ ...")
    def show_tree(path, depth=0, max_depth=3):
        if depth >= max_depth:
            return
        try:
            for p in sorted(path.iterdir()):
                indent = "    " + "   " * depth
                if p.is_dir():
                    print(f"{indent}📁 {p.name}/")
                    show_tree(p, depth+1, max_depth)
                else:
                    sz = p.stat().st_size / 1e6
                    print(f"{indent}📄 {p.name} ({sz:.0f} MB)")
        except PermissionError:
            pass
    show_tree(KAGGLE_INPUT)

    # ── Strategy 1: Look for pre-extracted directories ──
    # Kaggle auto-extracts zips into separate folders, so
    # leftImg8bit/ and gtFine/ may be in DIFFERENT parent dirs.
    # We find their exact paths independently.
    img_root = None  # path to the leftImg8bit/ directory itself
    ann_root = None  # path to the gtFine/ directory itself

    for match in KAGGLE_INPUT.rglob("leftImg8bit"):
        if match.is_dir() and (match / "train").exists():
            img_root = match
            break

    for match in KAGGLE_INPUT.rglob("gtFine"):
        if match.is_dir() and (match / "train").exists():
            ann_root = match
            break

    if img_root and ann_root:
        print(f"\n  ✓ Found pre-extracted Cityscapes data!")
        print(f"    Images:      {img_root}")
        print(f"    Annotations: {ann_root}")
        return img_root, ann_root

    # ── Strategy 2: Look for zip files and extract ──
    img_zip = None
    ann_zip = None

    for p in KAGGLE_INPUT.rglob("*.zip"):
        name = p.name.lower()
        if "leftimg8bit" in name:
            img_zip = p
        elif "gtfine" in name:
            ann_zip = p

    for p in KAGGLE_WORK.glob("*.zip"):
        name = p.name.lower()
        if "leftimg8bit" in name:
            img_zip = p
        elif "gtfine" in name:
            ann_zip = p

    if img_zip and ann_zip:
        print(f"\n  ✓ Found zip files — extracting...")
        print(f"    Images zip:      {img_zip} ({img_zip.stat().st_size/1e9:.1f} GB)")
        print(f"    Annotations zip: {ann_zip} ({ann_zip.stat().st_size/1e6:.0f} MB)")

        extract_dir = DATA_ROOT / "cityscapes"
        extract_dir.mkdir(parents=True, exist_ok=True)

        for zf_path in [ann_zip, img_zip]:
            print(f"    Extracting {zf_path.name}...")
            with zipfile.ZipFile(zf_path, "r") as zf:
                zf.extractall(extract_dir)

        # After extraction, find the dirs
        img_root = None
        ann_root = None
        for match in extract_dir.rglob("leftImg8bit"):
            if match.is_dir() and (match / "train").exists():
                img_root = match
                break
        for match in extract_dir.rglob("gtFine"):
            if match.is_dir() and (match / "train").exists():
                ann_root = match
                break

        if img_root and ann_root:
            print(f"  ✓ Extraction complete.")
            return img_root, ann_root

    # ── Nothing found ──
    print("\n  ✗ Could not find Cityscapes data!")
    print("    Looked for: leftImg8bit/train/ and gtFine/train/")
    if img_root:
        print(f"    Found images at: {img_root}")
    else:
        print("    Images (leftImg8bit/): NOT FOUND")
    if ann_root:
        print(f"    Found annotations at: {ann_root}")
    else:
        print("    Annotations (gtFine/): NOT FOUND")
    print("\n  Make sure you added 'iot-el-dataset' as Input to this notebook.")
    sys.exit(1)


# ──────────────────────────────────────────────
# STEP 2: CONVERT TO YOLO FORMAT
# ──────────────────────────────────────────────

def polygon_to_bbox(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def convert_annotation(json_path, img_w=2048, img_h=1024):
    with open(json_path) as f:
        data = json.load(f)

    lines = []
    for obj in data.get("objects", []):
        label = obj.get("label", "").lower()
        if label.startswith("person"): label = "person"
        elif label.startswith("rider"): label = "rider"
        elif label.startswith("car"):   label = "car"

        if label not in CITYSCAPES_CLASSES:
            continue

        class_id = CITYSCAPES_CLASSES[label]
        polygon  = obj.get("polygon", [])
        if len(polygon) < 3:
            continue

        x1, y1, x2, y2 = polygon_to_bbox(polygon)
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue

        cx = max(0, min(1, (x1+x2)/2/img_w))
        cy = max(0, min(1, (y1+y2)/2/img_h))
        w  = max(0, min(1, (x2-x1)/img_w))
        h  = max(0, min(1, (y2-y1)/img_h))
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


def build_yolo_dataset(img_root, ann_root, val_fraction=0.2):
    """Convert Cityscapes to YOLO format.
    
    Args:
        img_root: Path to leftImg8bit/ directory (contains train/, val/, test/)
        ann_root: Path to gtFine/ directory (contains train/, val/, test/)
    """
    print("\n" + "="*60)
    print("  STEP 2: CONVERTING TO YOLO FORMAT")
    print("="*60)
    print(f"    Images: {img_root}")
    print(f"    Annots: {ann_root}")

    if not img_root.exists() or not ann_root.exists():
        print(f"  [ERROR] Data directories not found!")
        print(f"    img_root exists: {img_root.exists()}")
        print(f"    ann_root exists: {ann_root.exists()}")
        sys.exit(1)

    # Collect pairs
    pairs = []
    for split in ["train", "val", "test"]:
        img_split = img_root / split
        if not img_split.exists():
            continue
        for city_dir in sorted(img_split.iterdir()):
            for img_file in sorted(city_dir.glob("*_leftImg8bit.png")):
                stem = img_file.stem.replace("_leftImg8bit", "")
                ann_file = ann_root / split / city_dir.name / f"{stem}_gtFine_polygons.json"
                if ann_file.exists():
                    pairs.append((img_file, ann_file))

    print(f"  Found {len(pairs)} image/annotation pairs.")

    random.seed(42)
    random.shuffle(pairs)
    n_val   = int(len(pairs) * val_fraction)
    val_set = pairs[:n_val]
    trn_set = pairs[n_val:]

    for split_name, split_pairs in [("train", trn_set), ("val", val_set)]:
        img_out = YOLO_DIR / "images" / split_name
        lbl_out = YOLO_DIR / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        skipped = 0
        for img_path, ann_path in split_pairs:
            label_lines = convert_annotation(ann_path)
            if not label_lines:
                skipped += 1
                continue
            dst_img = img_out / img_path.name
            if not dst_img.exists():
                try:
                    dst_img.symlink_to(img_path.resolve())
                except Exception:
                    shutil.copy2(img_path, dst_img)
            lbl_file = lbl_out / (img_path.stem + ".txt")
            lbl_file.write_text("\n".join(label_lines))

        print(f"    {split_name}: {len(split_pairs)-skipped} usable, {skipped} skipped")

    # Write dataset.yaml
    yaml_content = f"""# Cityscapes YOLO dataset (Kaggle)
path: {YOLO_DIR.resolve()}
train: images/train
val:   images/val

nc: {len(YOLO_CLASS_NAMES)}
names: {YOLO_CLASS_NAMES}
"""
    (YOLO_DIR / "dataset.yaml").write_text(yaml_content)
    print(f"  ✓ dataset.yaml written.")
    return str(YOLO_DIR / "dataset.yaml")


# ──────────────────────────────────────────────
# STEP 3: TRAIN YOLOv8n
# ──────────────────────────────────────────────

def train_model(data_yaml):
    from ultralytics import YOLO
    import torch

    print("\n" + "="*60)
    print("  STEP 3: TRAINING YOLOv8n (OPTIMIZED)")
    print("="*60)

    device = "0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        try:
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  VRAM: {vram:.1f} GB")
        except Exception:
            print("  VRAM: (could not detect)")
    else:
        print("  ⚠ No GPU — training will be slow!")

    print(f"  Epochs: {TRAIN_EPOCHS}")
    print(f"  Batch:  {TRAIN_BATCH}")
    print(f"  ImgSz:  {TRAIN_IMGSZ}×{TRAIN_IMGSZ}")

    model = YOLO("yolov8n.pt")

    start = time.time()
    results = model.train(
        data            = data_yaml,
        epochs          = TRAIN_EPOCHS,
        imgsz           = TRAIN_IMGSZ,
        batch           = TRAIN_BATCH,
        device          = device,
        project         = str(RUNS_DIR),
        name            = "train_baseline",
        lr0             = 1e-3,
        lrf             = 0.01,
        warmup_epochs   = 5,
        patience        = 25,
        cos_lr          = True,

        # Robust augmentations
        hsv_h           = 0.015,
        hsv_s           = 0.7,
        hsv_v           = 0.4,
        degrees         = 0.0,
        translate       = 0.15,
        scale           = 0.5,
        fliplr          = 0.5,
        flipud          = 0.0,
        mosaic          = 1.0,
        mixup           = 0.15,
        copy_paste      = 0.1,
        erasing         = 0.2,
        multi_scale     = True,

        save            = True,
        save_period     = 10,
        plots           = True,
        verbose         = True,
    )
    elapsed = time.time() - start
    print(f"\n  ✓ Training complete in {elapsed/3600:.2f} hours.")

    # Save best model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_pt = None
    for candidate in RUNS_DIR.rglob("best.pt"):
        best_pt = candidate
        break

    final_pt = MODELS_DIR / "yolov8n_cityscapes.pt"
    if best_pt and best_pt.exists():
        shutil.copy2(best_pt, final_pt)
        print(f"  ✓ Best model → {final_pt}")
    else:
        print("  [WARN] best.pt not found!")

    return final_pt


# ──────────────────────────────────────────────
# STEP 4: QUANTIZE + QAT
# ──────────────────────────────────────────────

def quantize_model(model_pt, data_yaml):
    from ultralytics import YOLO

    print("\n" + "="*60)
    print("  STEP 4: QUANTIZATION + QAT")
    print("="*60)

    if not model_pt.exists():
        print(f"  [ERROR] Model not found: {model_pt}")
        return

    # Export FP32 ONNX at 320 (RPi inference size)
    print("\n  4a. Exporting FP32 ONNX (320×320)...")
    model = YOLO(str(model_pt))
    model.export(format="onnx", imgsz=INFER_IMGSZ, opset=13, simplify=True, dynamic=False)

    fp32_onnx_src = model_pt.with_suffix(".onnx")
    fp32_onnx_dst = MODELS_DIR / "yolov8n_fp32.onnx"
    if fp32_onnx_src.exists():
        shutil.move(str(fp32_onnx_src), str(fp32_onnx_dst))
        sz = fp32_onnx_dst.stat().st_size / 1024**2
        print(f"  ✓ FP32 ONNX → {fp32_onnx_dst} ({sz:.1f} MB)")

    # INT8 static quantization
    print("\n  4b. INT8 Static Quantization...")
    try:
        from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationMethod
        import onnxruntime as ort

        # Calibration data reader
        class CalibReader:
            def __init__(self, img_dir, input_name, n=200):
                import cv2
                exts = [".jpg", ".jpeg", ".png"]
                self.paths = [p for p in Path(img_dir).rglob("*") if p.suffix.lower() in exts][:n]
                self.input_name = input_name
                self.idx = 0
                self.cv2 = cv2

            def get_next(self):
                if self.idx >= len(self.paths):
                    return None
                img = self.cv2.imread(str(self.paths[self.idx]))
                self.idx += 1
                if img is None:
                    return self.get_next()
                img = self.cv2.resize(img, (INFER_IMGSZ, INFER_IMGSZ))
                img = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                img = np.expand_dims(img, 0)
                return {self.input_name: np.ascontiguousarray(img)}

            def rewind(self):
                self.idx = 0

        sess = ort.InferenceSession(str(fp32_onnx_dst), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name

        calib_dir = YOLO_DIR / "images" / "val"
        reader = CalibReader(calib_dir, input_name, n=200)

        int8_onnx = MODELS_DIR / "yolov8n_int8.onnx"
        quantize_static(
            model_input=str(fp32_onnx_dst),
            model_output=str(int8_onnx),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            per_channel=False,
            reduce_range=False,
        )
        sz = int8_onnx.stat().st_size / 1024**2
        print(f"  ✓ INT8 ONNX → {int8_onnx} ({sz:.1f} MB)")
    except Exception as e:
        print(f"  [WARN] INT8 quantization failed: {e}")

    # QAT fine-tuning
    print("\n  4c. QAT Fine-Tuning (10 epochs)...")
    try:
        qat_model = YOLO(str(model_pt))
        qat_model.train(
            data=data_yaml, epochs=10, imgsz=INFER_IMGSZ, batch=16,
            device="0", lr0=1e-4, lrf=0.001, warmup_epochs=1, patience=10,
            project=str(RUNS_DIR), name="qat_finetune",
            save=True, plots=False, verbose=False,
        )
        qat_best = None
        for c in RUNS_DIR.rglob("qat_finetune*/weights/best.pt"):
            qat_best = c
            break
        if qat_best and qat_best.exists():
            qat_yolo = YOLO(str(qat_best))
            qat_yolo.export(format="onnx", imgsz=INFER_IMGSZ, opset=13, simplify=True)
            qat_onnx_src = qat_best.with_suffix(".onnx")
            qat_onnx_dst = MODELS_DIR / "yolov8n_qat.onnx"
            if qat_onnx_src.exists():
                shutil.move(str(qat_onnx_src), str(qat_onnx_dst))
                sz = qat_onnx_dst.stat().st_size / 1024**2
                print(f"  ✓ QAT ONNX → {qat_onnx_dst} ({sz:.1f} MB)")
    except Exception as e:
        print(f"  [WARN] QAT failed: {e}")

    # Summary
    print("\n" + "="*60)
    print("  MODEL SUMMARY")
    print("="*60)
    for f in sorted(MODELS_DIR.glob("*")):
        sz = f.stat().st_size / 1024**2
        print(f"    {f.name:30s}  {sz:6.1f} MB")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  HEAD-MOUNTED NAVIGATION — KAGGLE TRAINING PIPELINE")
    print("="*60 + "\n")

    # Step 1: Find dataset (pre-extracted dirs or zips → extract)
    img_root, ann_root = find_cityscapes_data()

    # Step 2: Convert to YOLO
    data_yaml = build_yolo_dataset(img_root, ann_root)

    # Step 4: Train
    model_pt = train_model(data_yaml)

    # Step 5: Quantize
    quantize_model(model_pt, data_yaml)

    print("\n" + "="*60)
    print("  ✅ ALL DONE!")
    print("="*60)
    print(f"\n  Download your models from: {MODELS_DIR}")
    print("  Key file for RPi: yolov8n_qat.onnx")
    print("\n  Transfer to RPi:")
    print("    scp models/yolov8n_qat.onnx pi@<RPI_IP>:~/navigation/models/")
    print("    Then run: python navigation_system_rpi.py\n")


if __name__ == "__main__":
    main()
