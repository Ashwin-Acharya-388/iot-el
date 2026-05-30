"""
kaggle_freespace.py
===================
Complete Kaggle GPU training script for the Free-Space Navigation System.

This script handles everything end-to-end on Kaggle:
    1. Install dependencies
    2. Download ADE20K from Kaggle dataset or HuggingFace
    3. Remap ADE20K → binary walkable/non-walkable masks
    4. Train DeepLabV3-MobileNetV3-Large (binary segmentation)
    5. Export to ONNX + INT8 quantization
    6. Save all models to /kaggle/working/ for download

HOW TO USE ON KAGGLE:
─────────────────────
1. Create a new Kaggle Notebook (GPU T4 ×2 accelerator)
2. Add "ADE20K Scene Parsing" as a dataset input
   (Search: "ade20k" on Kaggle Datasets → Add to notebook)
3. Paste this entire script into a single code cell
4. Run the cell — training takes ~30-60 min on T4

Alternatively, upload this .py file and run:
    !python kaggle_freespace.py
"""

import os
import sys
import json
import time
import shutil
import random
from pathlib import Path

# ──────────────────────────────────────────────
# STEP 0: INSTALL DEPENDENCIES
# ──────────────────────────────────────────────

def install_deps():
    print("Installing dependencies...")
    os.system("pip install -q onnx onnxruntime onnxsim onnxscript opencv-python-headless")
    print("✓ Dependencies installed.\n")

install_deps()

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Modern / Legacy PyTorch AMP compatibility
try:
    from torch.amp import autocast, GradScaler
    HAS_MODERN_AMP = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    HAS_MODERN_AMP = False

# Pretrained weights compatibility
try:
    from torchvision.models.segmentation import (
        deeplabv3_mobilenet_v3_large,
        DeepLabV3_MobileNet_V3_Large_Weights,
    )
    HAS_NEW_WEIGHTS = True
except ImportError:
    try:
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
        HAS_NEW_WEIGHTS = False
    except ImportError as e:
        print(f"[FATAL ERROR] Failed to import deeplabv3_mobilenet_v3_large from torchvision: {e}")
        sys.exit(1)


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

# Detect if running on Kaggle or locally
IS_KAGGLE = os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ

if IS_KAGGLE:
    KAGGLE_INPUT = Path("/kaggle/input")
    KAGGLE_WORK  = Path("/kaggle/working")
else:
    # Local development fallback
    KAGGLE_INPUT = Path("./data")
    KAGGLE_WORK  = Path(".")
    # Create directories locally to prevent FileNotFoundError
    KAGGLE_INPUT.mkdir(parents=True, exist_ok=True)

DATA_DIR     = KAGGLE_WORK / "data" / "freespace"
MODELS_DIR   = KAGGLE_WORK / "models"
RUNS_DIR     = KAGGLE_WORK / "runs" / "freespace"

# Training config
TRAIN_EPOCHS = 40
TRAIN_BATCH  = 12       # Slightly smaller for stronger aug overhead
IMG_SIZE     = 320
LR           = 5e-4     # Lower peak LR for stability with warmup
NUM_CLASSES  = 2
PATIENCE     = 12       # More patience — augmented training needs more time
WORKERS      = 2        # Kaggle often has limited CPU cores
WARMUP_EPOCHS = 3       # Linear warmup before cosine decay
UNFREEZE_EPOCH = 5      # Unfreeze backbone after this epoch

# ADE20K walkable pixel values (1-indexed class IDs in _seg.png)
# These pixel values in the annotation PNGs represent walkable surfaces
WALKABLE_PIXEL_VALUES = {
    4,   # floor, flooring
    7,   # road, route
    10,  # grass
    12,  # sidewalk, pavement
    14,  # earth, ground
    18,  # field
    22,  # path
    29,  # rug, carpet
}

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ──────────────────────────────────────────────
# STEP 1: FIND ADE20K DATA
# ──────────────────────────────────────────────

def find_ade20k():
    """
    Locate ADE20K data from Kaggle input datasets or download it from MIT CSAIL/HuggingFace LFS.
    """
    print("=" * 60)
    print("  STEP 1: LOCATING ADE20K DATASET")
    print("=" * 60)

    # Strategy 0: Check if already downloaded/extracted in /kaggle/working (or local .)
    working_candidate = KAGGLE_WORK / "ADEChallengeData2016"
    if working_candidate.exists() and (working_candidate / "images" / "training").exists():
        print(f"\n  ✓ Found ADE20K in workspace: {working_candidate}")
        return working_candidate

    # Scan input if it exists
    ade20k_root = None
    if KAGGLE_INPUT.exists():
        print(f"\n  Scanning {KAGGLE_INPUT} ...")
        
        # Safe, non-recursive tree display
        def show_tree(path, depth=0, max_depth=2, max_files_per_dir=10):
            if depth >= max_depth:
                return
            try:
                files_printed = 0
                for p in sorted(path.iterdir()):
                    indent = "    " + "   " * depth
                    if p.is_dir():
                        print(f"{indent}📁 {p.name}/")
                        show_tree(p, depth + 1, max_depth, max_files_per_dir)
                    else:
                        if files_printed < max_files_per_dir:
                            sz = p.stat().st_size / 1e6
                            print(f"{indent}📄 {p.name} ({sz:.1f} MB)")
                            files_printed += 1
                        elif files_printed == max_files_per_dir:
                            print(f"{indent}... and more files")
                            files_printed += 1
            except (PermissionError, FileNotFoundError):
                pass
                
        show_tree(KAGGLE_INPUT)

        # Strategy 1: Smart, shallow search for ADEChallengeData2016 structure
        print("\n  Searching for ADEChallengeData2016/ ...")
        try:
            for candidate in KAGGLE_INPUT.iterdir():
                if candidate.is_dir():
                    if candidate.name == "ADEChallengeData2016":
                        ade20k_root = candidate
                        break
                    # Check one level down
                    for sub in candidate.iterdir():
                        if sub.is_dir() and sub.name == "ADEChallengeData2016":
                            ade20k_root = sub
                            break
                if ade20k_root:
                    break
        except Exception:
            pass

        # Strategy 2: Look for images/training or images/train directories
        if not ade20k_root:
            print("  Searching for images/training/ structure ...")
            try:
                for match in KAGGLE_INPUT.rglob("images"):
                    if (match / "training").exists() or (match / "train").exists():
                        ade20k_root = match.parent
                        print(f"    Found via images path: {ade20k_root}")
                        break
            except Exception:
                pass

        # Strategy 3: Look for any directory with "ade20k" in the name, then probe inside
        if not ade20k_root:
            print("  Searching for directories containing 'ade20k' ...")
            try:
                for candidate in KAGGLE_INPUT.iterdir():
                    if candidate.is_dir() and "ade20k" in candidate.name.lower():
                        print(f"    Found ADE20K directory: {candidate}")
                        # Check if THIS directory has the standard structure inside
                        for sub in [candidate, *candidate.iterdir()]:
                            if not sub.is_dir():
                                continue
                            img_dir = sub / "images"
                            if img_dir.exists():
                                for split_name in ["training", "train"]:
                                    if (img_dir / split_name).exists():
                                        ade20k_root = sub
                                        print(f"    Resolved root: {ade20k_root}")
                                        break
                            if ade20k_root:
                                break
                        # If still not found, check for ADEChallengeData2016 inside
                        if not ade20k_root:
                            for sub in candidate.rglob("ADEChallengeData2016"):
                                if sub.is_dir():
                                    ade20k_root = sub
                                    print(f"    Resolved root: {ade20k_root}")
                                    break
                        # If still not found, the ade20k dir itself might BE the root
                        if not ade20k_root:
                            if (candidate / "images").exists() or (candidate / "annotations").exists():
                                ade20k_root = candidate
                                print(f"    Using as root: {ade20k_root}")
                        if ade20k_root:
                            break
            except Exception:
                pass

    if ade20k_root:
        print(f"\n  ✓ Found ADE20K in inputs at: {ade20k_root}")
        print(f"    Contents:")
        try:
            for p in sorted(ade20k_root.iterdir())[:15]:
                label = "📁" if p.is_dir() else "📄"
                print(f"      {label} {p.name}")
        except Exception:
            pass
        return ade20k_root

    # Strategy 5: Automatically download from HuggingFace mirror or MIT CSAIL!
    print("\n  [INFO] ADE20K was not found in inputs.")
    print("  [INFO] Initiating automatic download...")
    
    import urllib.request
    import zipfile
    
    zip_path = KAGGLE_WORK / "ADEChallengeData2016.zip"
    extract_dir = KAGGLE_WORK
    final_dir = extract_dir / "ADEChallengeData2016"
    
    # Try Hugging Face first (insanely fast CDN, ~100MB/s), fallback to MIT CSAIL (sometimes slow or offline)
    hf_url = "https://huggingface.co/datasets/zbwxp/ade/resolve/main/ADEChallengeData2016.zip"
    mit_url = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
    
    download_success = False
    for name, url in [("HuggingFace CDN (Fast)", hf_url), ("MIT CSAIL (Official)", mit_url)]:
        print(f"\n    Trying download from {name} ...")
        print(f"    Saving to {zip_path}")
        print("    Note: Make sure 'Internet on' is enabled in your Kaggle Notebook settings (right sidebar -> Session options).")
        
        try:
            if zip_path.exists():
                zip_path.unlink()
                
            last_reported = -1
            def progress_hook(block_num, block_size, total_size):
                nonlocal last_reported
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, int((downloaded * 100) / total_size))
                    if percent % 10 == 0 and percent != last_reported:
                        print(f"      Downloaded: {downloaded / 1024**2:.1f} MB / {total_size / 1024**2:.1f} MB ({percent}%)")
                        last_reported = percent
                else:
                    if block_num % 1000 == 0:
                        print(f"      Downloaded: {downloaded / 1024**2:.1f} MB")
            
            urllib.request.urlretrieve(url, zip_path, progress_hook)
            print(f"    ✓ Download complete! Size: {zip_path.stat().st_size / 1024**2:.1f} MB")
            download_success = True
            break
        except Exception as e:
            print(f"    ✗ Download from {name} failed: {e}")
            if zip_path.exists():
                try:
                    zip_path.unlink()
                except:
                    pass
                    
    if download_success:
        try:
            print("  [INFO] Extracting zip file (this takes ~1-2 mins) ...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print("  ✓ Extraction complete!")
            
            # Clean up zip file
            if zip_path.exists():
                zip_path.unlink()
                print("  ✓ Removed temporary ZIP file.")
                
            if final_dir.exists():
                print(f"  ✓ Resolved downloaded ADE20K root to: {final_dir}")
                return final_dir
        except Exception as e:
            print(f"  [ERROR] Extraction failed: {e}")
    print("\n  ✗ FATAL: Unable to find or download the ADE20K dataset.")
    print("  To resolve this:")
    print("  1. Enable 'Internet' in Kaggle settings (right sidebar -> Session options -> Internet on) and re-run.")
    print("  OR")
    print("  2. Manually add an active ADE20K Dataset by searching for 'ADEChallengeData2016' under '+ Add Input'.")
    sys.exit(1)


# ──────────────────────────────────────────────
# STEP 2: REMAP TO BINARY FREE SPACE
# ──────────────────────────────────────────────

def decode_mask(mask_path: Path) -> np.ndarray:
    """Load ADE20K annotation mask. Handles both single-channel and RGB formats."""
    img = Image.open(mask_path)
    arr = np.array(img)
    if arr.ndim == 2:
        return arr
    elif arr.ndim == 3:
        r, g = arr[:, :, 0], arr[:, :, 1]
        return (r.astype(np.int32) // 10) * 256 + g.astype(np.int32)
    raise ValueError(f"Unexpected mask shape: {arr.shape}")


def convert_ade20k_to_binary(ade20k_root: Path) -> Path:
    """Convert ADE20K to binary free-space dataset."""
    print("\n" + "=" * 60)
    print("  STEP 2: CONVERTING ADE20K → BINARY FREE SPACE")
    print("=" * 60)

    img_base = ade20k_root / "images"
    ann_base = ade20k_root / "annotations"

    if not img_base.exists():
        print(f"  [ERROR] {img_base} not found")
        sys.exit(1)

    # Detect split names
    train_name = "training" if (img_base / "training").exists() else "train"
    val_name = "validation" if (img_base / "validation").exists() else "val"

    for split, dir_name in [("train", train_name), ("val", val_name)]:
        img_dir = img_base / dir_name
        ann_dir = ann_base / dir_name

        if not img_dir.exists():
            print(f"  [WARN] {img_dir} not found — skipping {split}")
            continue

        out_img = DATA_DIR / "images" / split
        out_mask = DATA_DIR / "masks" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_mask.mkdir(parents=True, exist_ok=True)

        img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        converted = 0
        has_walkable = 0

        print(f"\n  Processing {split} ({len(img_files)} images)...")

        for img_path in img_files:
            # Find annotation: support stem.png, stem_seg.png, etc.
            ann_path = None
            for candidate_name in [f"{img_path.stem}.png", f"{img_path.stem}_seg.png", f"{img_path.stem}.tif"]:
                candidate = ann_dir / candidate_name
                if candidate.exists():
                    ann_path = candidate
                    break
            if not ann_path:
                continue

            try:
                class_mask = decode_mask(ann_path)
            except Exception:
                continue

            # Binary mask
            binary = np.isin(class_mask, list(WALKABLE_PIXEL_VALUES)).astype(np.uint8)

            # Resize
            img = Image.open(img_path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            
            mask_pil = Image.fromarray(binary)
            mask_pil = mask_pil.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
            binary_resized = np.array(mask_pil)

            # Track stats
            frac = binary_resized.sum() / binary_resized.size
            if frac > 0.01:
                has_walkable += 1

            # Save
            img.save(out_img / f"{img_path.stem}.jpg", quality=95)
            Image.fromarray(binary_resized * 255).save(out_mask / f"{img_path.stem}.png")

            converted += 1
            if converted % 1000 == 0:
                print(f"    Processed {converted}...")

        print(f"    {split}: {converted} images, {has_walkable} with walkable area ({100*has_walkable/max(converted,1):.0f}%)")

    print(f"\n  ✓ Binary free-space dataset ready at: {DATA_DIR}")
    return DATA_DIR


# ──────────────────────────────────────────────
# STEP 3: DATASET & TRAINING
# ──────────────────────────────────────────────

class FreespaceDataset(Dataset):
    """Binary free-space segmentation dataset with strong augmentations."""

    def __init__(self, img_dir, mask_dir, img_size=320, augment=True):
        self.img_size = img_size
        self.augment = augment
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.samples = []
        for img_path in sorted(Path(img_dir).glob("*.jpg")):
            mask_path = Path(mask_dir) / f"{img_path.stem}.png"
            if mask_path.exists():
                self.samples.append((img_path, mask_path))
        print(f"    Dataset: {len(self.samples)} samples from {img_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.augment:
            img, mask = self._augment(img, mask)

        img_t = transforms.ToTensor()(img)
        img_t = self.normalize(img_t)
        mask_t = torch.from_numpy((np.array(mask) > 127).astype(np.int64))
        return img_t, mask_t

    def _augment(self, img, mask):
        from torchvision.transforms import functional as TF

        # 1) Random horizontal flip (50%)
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # 2) Multi-scale random crop (70% chance) — scale 0.5x to 2.0x
        if random.random() > 0.3:
            scale = random.uniform(0.5, 2.0)
            new_h = int(self.img_size * scale)
            new_w = int(self.img_size * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            mask = mask.resize((new_w, new_h), Image.NEAREST)
            # Pad if too small
            if new_h < self.img_size or new_w < self.img_size:
                pad_h = max(0, self.img_size - new_h)
                pad_w = max(0, self.img_size - new_w)
                img = TF.pad(img, [0, 0, pad_w, pad_h], fill=0)
                mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=0)
                new_h = max(new_h, self.img_size)
                new_w = max(new_w, self.img_size)
            # Random crop to img_size
            y = random.randint(0, new_h - self.img_size)
            x = random.randint(0, new_w - self.img_size)
            img = TF.crop(img, y, x, self.img_size, self.img_size)
            mask = TF.crop(mask, y, x, self.img_size, self.img_size)

        # 3) Random rotation ±15° (40% chance)
        if random.random() > 0.6:
            angle = random.uniform(-15, 15)
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)

        # 4) Color jitter (60% chance) — brightness, contrast, saturation, hue
        if random.random() > 0.4:
            img = TF.adjust_brightness(img, random.uniform(0.6, 1.4))
            img = TF.adjust_contrast(img, random.uniform(0.6, 1.4))
            img = TF.adjust_saturation(img, random.uniform(0.7, 1.3))
            img = TF.adjust_hue(img, random.uniform(-0.05, 0.05))

        # 5) Gaussian blur (20% chance)
        if random.random() > 0.8:
            img = TF.gaussian_blur(img, kernel_size=5, sigma=random.uniform(0.1, 2.0))

        return img, mask


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_soft = F.softmax(pred, dim=1)[:, 1]
        target_f = target.float()
        intersection = (pred_soft * target_f).sum(dim=(1, 2))
        union = pred_soft.sum(dim=(1, 2)) + target_f.sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        # Weight the walkable class higher (class 1) to combat class imbalance
        # Most pixels are non-walkable, so upweight walkable to improve recall
        weight = torch.tensor([1.0, 3.0])
        if device:
            weight = weight.to(device)
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return 0.5 * self.ce(pred, target) + 0.5 * self.dice(pred, target)


def compute_iou(pred, target, nc=2):
    ious = {}
    for c in range(nc):
        inter = ((pred == c) & (target == c)).sum().item()
        union = ((pred == c) | (target == c)).sum().item()
        ious[c] = inter / max(union, 1)
    ious["mean"] = sum(ious.values()) / nc
    return ious


def create_model(nc=2):
    if HAS_NEW_WEIGHTS:
        model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
    else:
        # Fallback for older torchvision versions
        model = deeplabv3_mobilenet_v3_large(pretrained=True)
        
    in_ch = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_ch, nc, kernel_size=1)
    if model.aux_classifier is not None:
        aux_in = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(aux_in, nc, kernel_size=1)
    return model


def train_model(data_dir):
    """Train DeepLabV3-MobileNetV3-Large on binary free-space masks."""
    print("\n" + "=" * 60)
    print("  STEP 3: TRAINING (DeepLabV3 + MobileNetV3-Large)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠ No GPU — training will be slow!")

    # Datasets
    train_ds = FreespaceDataset(data_dir / "images" / "train",
                                data_dir / "masks" / "train", augment=True)
    val_ds = FreespaceDataset(data_dir / "images" / "val",
                              data_dir / "masks" / "val", augment=False)

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH, shuffle=True,
                              num_workers=WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_BATCH, shuffle=False,
                            num_workers=WORKERS, pin_memory=True)

    # Model
    model = create_model(NUM_CLASSES).to(device)

    # ── Freeze backbone initially — only train classifier head for first epochs ──
    for name, param in model.named_parameters():
        if "classifier" not in name and "aux_classifier" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Phase 1 (frozen backbone): {trainable:,} / {total:,} trainable params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = CombinedLoss(device=device)
    
    if HAS_MODERN_AMP:
        scaler = GradScaler('cuda', enabled=torch.cuda.is_available())
    else:
        scaler = GradScaler(enabled=torch.cuda.is_available())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    best_iou = 0.0
    patience_counter = 0
    history = []
    backbone_unfrozen = False

    print(f"\n  Training: {TRAIN_EPOCHS} epochs, batch {TRAIN_BATCH}, LR {LR}")
    print(f"  Warmup: {WARMUP_EPOCHS} epochs, Unfreeze backbone at epoch {UNFREEZE_EPOCH}")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}\n")

    for epoch in range(1, TRAIN_EPOCHS + 1):
        t0 = time.time()

        # ── Progressive unfreezing: unfreeze entire backbone after UNFREEZE_EPOCH ──
        if epoch == UNFREEZE_EPOCH and not backbone_unfrozen:
            for param in model.parameters():
                param.requires_grad = True
            backbone_unfrozen = True
            # Re-create optimizer with differential LR: backbone gets 0.1x LR
            backbone_params = []
            head_params = []
            for name, param in model.named_parameters():
                if "classifier" in name or "aux_classifier" in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)
            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": LR * 0.1},
                {"params": head_params, "lr": LR},
            ], weight_decay=1e-4)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  ── Epoch {epoch}: Unfroze backbone! {trainable:,} trainable params ──")
            print(f"     Backbone LR: {LR * 0.1:.1e}, Head LR: {LR:.1e}")

        # ── Learning rate: warmup then cosine ──
        if epoch <= WARMUP_EPOCHS:
            warmup_lr = LR * epoch / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                # Scale by the ratio set at optimizer creation
                if backbone_unfrozen and pg is optimizer.param_groups[0]:
                    pg["lr"] = warmup_lr * 0.1
                else:
                    pg["lr"] = warmup_lr
        else:
            # Cosine decay from WARMUP_EPOCHS+1 to TRAIN_EPOCHS
            progress = (epoch - WARMUP_EPOCHS) / max(TRAIN_EPOCHS - WARMUP_EPOCHS, 1)
            cos_lr = 1e-6 + 0.5 * (LR - 1e-6) * (1 + np.cos(np.pi * progress))
            for pg in optimizer.param_groups:
                if backbone_unfrozen and pg is optimizer.param_groups[0]:
                    pg["lr"] = cos_lr * 0.1
                else:
                    pg["lr"] = cos_lr

        # ── Train ──
        model.train()
        train_loss = 0
        n = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            if HAS_MODERN_AMP:
                autocast_ctx = autocast('cuda', enabled=torch.cuda.is_available())
            else:
                autocast_ctx = autocast(enabled=torch.cuda.is_available())
                
            with autocast_ctx:
                out = model(images)["out"]
                if out.shape[-2:] != masks.shape[-2:]:
                    out = F.interpolate(out, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = criterion(out, masks)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            n += 1
        train_loss /= max(n, 1)

        # ── Validate ──
        model.eval()
        val_loss = 0
        all_ious = []
        n = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                if HAS_MODERN_AMP:
                    autocast_ctx = autocast('cuda', enabled=torch.cuda.is_available())
                else:
                    autocast_ctx = autocast(enabled=torch.cuda.is_available())
                    
                with autocast_ctx:
                    out = model(images)["out"]
                    if out.shape[-2:] != masks.shape[-2:]:
                        out = F.interpolate(out, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                    val_loss += criterion(out, masks).item()
                    
                preds = out.argmax(dim=1)
                ious = compute_iou(preds, masks)
                all_ious.append(ious["mean"])
                n += 1
        val_loss /= max(n, 1)
        mean_iou = np.mean(all_ious)

        lr = optimizer.param_groups[-1]["lr"]  # Head LR
        elapsed = time.time() - t0

        phase = "[HEAD]" if not backbone_unfrozen else "[FULL]"
        print(f"  Epoch {epoch:3d}/{TRAIN_EPOCHS} {phase} | "
              f"TrLoss: {train_loss:.4f} | VaLoss: {val_loss:.4f} | "
              f"mIoU: {100*mean_iou:.1f}% | LR: {lr:.2e} | {elapsed:.0f}s")

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "mean_iou": mean_iou})

        if mean_iou > best_iou:
            best_iou = mean_iou
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / "freespace_best.pth")
            print(f"    ★ New best! mIoU: {100*best_iou:.1f}%")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\n  ⏹ Early stopping (no improvement for {PATIENCE} epochs)")
            break

    # Save history
    with open(RUNS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  ✓ Best Mean IoU: {100*best_iou:.1f}%")
    print(f"  ✓ Model saved: {MODELS_DIR / 'freespace_best.pth'}")
    return MODELS_DIR / "freespace_best.pth"


# ──────────────────────────────────────────────
# STEP 4: EXPORT TO ONNX + INT8
# ──────────────────────────────────────────────

def export_and_quantize(model_path, data_dir):
    """Export to ONNX and quantize to INT8."""
    print("\n" + "=" * 60)
    print("  STEP 4: ONNX EXPORT + INT8 QUANTIZATION")
    print("=" * 60)

    # ── Load model ──
    model = create_model(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # ── Wrap to return only logits ──
    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x)["out"]

    wrapped = Wrapper(model)
    wrapped.eval()

    # ── Export FP32 ONNX ──
    fp32_path = MODELS_DIR / "freespace_fp32.onnx"
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    try:
        # Try legacy exporter first (works on all PyTorch versions)
        torch.onnx.export(wrapped, dummy, str(fp32_path),
                          input_names=["image"], output_names=["mask"],
                          opset_version=13, dynamo=False)
    except TypeError:
        # Older PyTorch without dynamo kwarg
        torch.onnx.export(wrapped, dummy, str(fp32_path),
                          input_names=["image"], output_names=["mask"],
                          opset_version=13)
    sz = fp32_path.stat().st_size / 1024**2
    print(f"  ✓ FP32 ONNX: {fp32_path} ({sz:.1f} MB)")

    # Simplify
    try:
        import onnxsim, onnx
        m = onnx.load(str(fp32_path))
        m_sim, ok = onnxsim.simplify(m)
        if ok:
            onnx.save(m_sim, str(fp32_path))
            sz = fp32_path.stat().st_size / 1024**2
            print(f"  ✓ Simplified: {sz:.1f} MB")
    except Exception as e:
        print(f"  [INFO] Simplification skipped: {e}")

    # ── INT8 quantization ──
    int8_path = MODELS_DIR / "freespace_int8.onnx"
    try:
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationMethod
        import cv2

        class CalibReader:
            def __init__(self, img_dir, input_name, n=200):
                exts = {".jpg", ".png"}
                self.paths = [p for p in Path(img_dir).rglob("*") if p.suffix.lower() in exts][:n]
                self.input_name = input_name
                self.idx = 0

            def get_next(self):
                if self.idx >= len(self.paths):
                    return None
                img = cv2.imread(str(self.paths[self.idx]))
                self.idx += 1
                if img is None:
                    return self.get_next()
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                img = (img - MEAN) / STD
                img = np.transpose(img, (2, 0, 1))[np.newaxis]
                return {self.input_name: np.ascontiguousarray(img)}

            def rewind(self):
                self.idx = 0

        sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name

        reader = CalibReader(data_dir / "images" / "val", input_name)

        quantize_static(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )
        sz = int8_path.stat().st_size / 1024**2
        print(f"  ✓ INT8 ONNX: {int8_path} ({sz:.1f} MB)")
    except Exception as e:
        print(f"  [WARN] INT8 quantization failed: {e}")
        print(f"  You can still use the FP32 model on RPi (slower but works).")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  MODEL SUMMARY")
    print("=" * 60)
    for f in sorted(MODELS_DIR.glob("*")):
        sz = f.stat().st_size / 1024**2
        print(f"    {f.name:35s}  {sz:6.1f} MB")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  HEAD-MOUNTED NAVIGATION — FREE SPACE TRAINING (KAGGLE)")
    print("=" * 60 + "\n")

    # Step 1: Find ADE20K
    ade20k_root = find_ade20k()

    # Step 2: Convert to binary free space
    data_dir = convert_ade20k_to_binary(ade20k_root)

    # Step 3: Train
    model_path = train_model(data_dir)

    # Step 4: Export + Quantize
    export_and_quantize(model_path, data_dir)

    print("\n" + "=" * 60)
    print("  ✅ ALL DONE!")
    print("=" * 60)
    print(f"\n  Download your models from: {MODELS_DIR}")
    print(f"  Key file for RPi: freespace_int8.onnx")
    print(f"\n  Transfer to RPi:")
    print(f"    scp models/freespace_int8.onnx pi@<RPI_IP>:~/navigation/models/")
    print(f"    Then run: python navigation_freespace_rpi.py")
    print()


if __name__ == "__main__":
    main()
