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

# WALKABLE_PIXEL_VALUES remapped to strictly indoor walkable surfaces
WALKABLE_PIXEL_VALUES = {
    4,   # floor, flooring
    29,  # rug, carpet
}

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ──────────────────────────────────────────────
# STEP 1: FIND ADE20K DATA
# ──────────────────────────────────────────────

INDOOR_KEYWORDS = {'corridor', 'hallway', 'office', 'living_room', 'kitchen', 'lobby', 'room'}

def get_indoor_stems(ade20k_root: Path) -> set:
    """Parse sceneCategories.txt if it exists to find indoor image names."""
    stems = set()
    categories_file = ade20k_root / "sceneCategories.txt"
    if not categories_file.exists():
        # Search recursively for sceneCategories.txt
        found = list(ade20k_root.rglob("sceneCategories.txt"))
        if found:
            categories_file = found[0]
            
    if categories_file.exists():
        try:
            with open(categories_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        stem = parts[0]
                        category = parts[1].lower()
                        if any(kw in category for kw in INDOOR_KEYWORDS):
                            stems.add(stem)
            print(f"  [FILTER] Parsed {len(stems)} indoor image stems from {categories_file.name}")
        except Exception as e:
            print(f"  [WARN] Failed to parse categories file: {e}")
    return stems

def find_ade20k():
    """
    Locate ADE20K data dynamically from Kaggle input datasets or download it.
    """
    print("=" * 60)
    print("  STEP 1: LOCATING ADE20K DATASET")
    print("=" * 60)

    # Strategy 0: Check if already downloaded/extracted in /kaggle/working
    working_candidate = KAGGLE_WORK / "ADEChallengeData2016"
    if working_candidate.exists() and (working_candidate / "images" / "training").exists():
        print(f"\n  ✓ Found ADE20K in workspace: {working_candidate}")
        return working_candidate

    # Scan input if it exists
    ade20k_root = None
    if KAGGLE_INPUT.exists():
        print(f"\n  Scanning {KAGGLE_INPUT} recursively for ADE20K...")
        
        # 1. Search for sceneCategories.txt
        found_categories = list(KAGGLE_INPUT.rglob("sceneCategories.txt"))
        if found_categories:
            ade20k_root = found_categories[0].parent
            print(f"    ✓ Found ADE20K root via sceneCategories.txt at: {ade20k_root}")
            return ade20k_root
            
        # 2. Search for images/training or images/train
        found_images = list(KAGGLE_INPUT.rglob("images"))
        for img_dir in found_images:
            if (img_dir / "training").exists() or (img_dir / "train").exists():
                ade20k_root = img_dir.parent
                print(f"    ✓ Found ADE20K root via images/ folder at: {ade20k_root}")
                return ade20k_root
                
        # 3. Search for annotations/training
        found_ann = list(KAGGLE_INPUT.rglob("annotations"))
        for ann_dir in found_ann:
            if (ann_dir / "training").exists() or (ann_dir / "train").exists():
                ade20k_root = ann_dir.parent
                print(f"    ✓ Found ADE20K root via annotations/ folder at: {ade20k_root}")
                return ade20k_root

    # Strategy 5: Automatically download from HuggingFace mirror or MIT CSAIL!
    print("\n  [INFO] ADE20K was not found in inputs. Initiating automatic download...")
    
    import urllib.request
    import zipfile
    
    zip_path = KAGGLE_WORK / "ADEChallengeData2016.zip"
    extract_dir = KAGGLE_WORK
    final_dir = extract_dir / "ADEChallengeData2016"
    
    hf_url = "https://huggingface.co/datasets/zbwxp/ade/resolve/main/ADEChallengeData2016.zip"
    mit_url = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
    
    download_success = False
    for name, url in [("HuggingFace CDN (Fast)", hf_url), ("MIT CSAIL (Official)", mit_url)]:
        print(f"\n    Trying download from {name} ...")
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
            
            urllib.request.urlretrieve(url, zip_path, progress_hook)
            download_success = True
            break
        except Exception as e:
            print(f"    ✗ Download from {name} failed: {e}")
            
    if download_success:
        try:
            print("  [INFO] Extracting zip file...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            if zip_path.exists():
                zip_path.unlink()
            if final_dir.exists():
                return final_dir
        except Exception as e:
            print(f"  [ERROR] Extraction failed: {e}")
            
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
    """Convert ADE20K to binary free-space dataset, keeping only indoor categories."""
    print("\n" + "=" * 60)
    print("  STEP 2: CONVERTING ADE20K → BINARY FREE SPACE (INDOOR ONLY)")
    print("=" * 60)

    # Get indoor categories stems
    indoor_stems = get_indoor_stems(ade20k_root)

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

        img_files = sorted(list(img_dir.rglob("*.jpg")) + list(img_dir.rglob("*.png")))
        converted = 0
        skipped_not_indoor = 0
        has_walkable = 0

        print(f"\n  Processing {split} ({len(img_files)} images)...")

        for img_path in img_files:
            # Indoor filtering
            is_indoor = False
            if indoor_stems and img_path.stem in indoor_stems:
                is_indoor = True
            
            # Always check path keywords as well to handle nested folder structures
            path_str = str(img_path).lower()
            if any(kw in path_str for kw in INDOOR_KEYWORDS):
                is_indoor = True

            if not is_indoor:
                skipped_not_indoor += 1
                continue

            # Find corresponding annotation (handles flat and hierarchical mirrored paths)
            rel_path = img_path.relative_to(img_dir)
            ann_path = None
            for ext in [".png", ".tif", ".tiff"]:
                # Standard matching
                candidate = ann_dir / rel_path.with_suffix(ext)
                if candidate.exists():
                    ann_path = candidate
                    break
                # MIT CSAIL '_seg' suffix matching
                candidate_seg = ann_dir / rel_path.parent / f"{rel_path.stem}_seg{ext}"
                if candidate_seg.exists():
                    ann_path = candidate_seg
                    break
            
            if not ann_path:
                continue

            try:
                class_mask = decode_mask(ann_path)
            except Exception:
                continue

            # Binary mask mapping: walkable=1, obstacle=0
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

        print(f"    {split}: {converted} indoor images converted, skipped {skipped_not_indoor} outdoor images, {has_walkable} with walkable area ({100*has_walkable/max(converted,1):.0f}%)")

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


def compute_class_weights(mask_dir, num_samples=500):
    """
    Dynamically scan a sample of training masks to calculate class weights
    based on inverse pixel frequencies.
    """
    print("\n  Estimating class weights dynamically...")
    mask_paths = list(Path(mask_dir).glob("*.png"))
    if not mask_paths:
        print("    [WARN] No masks found to compute weights, using default [1.0, 3.0]")
        return torch.tensor([1.0, 3.0])
    
    samples = random.sample(mask_paths, min(num_samples, len(mask_paths)))
    
    total_pixels = 0
    class_counts = np.zeros(2, dtype=np.int64)
    
    for p in samples:
        try:
            m = np.array(Image.open(p))
            # 0 = non-walkable, 255 = walkable
            walkable = (m > 127).astype(np.int64)
            
            c1 = walkable.sum()
            c0 = walkable.size - c1
            
            class_counts[0] += c0
            class_counts[1] += c1
            total_pixels += walkable.size
        except Exception as e:
            continue
            
    if class_counts[0] == 0 or class_counts[1] == 0:
        print("    [WARN] One of the classes has 0 pixels in sample, using default [1.0, 3.0]")
        return torch.tensor([1.0, 3.0])
        
    # Inverse frequency weights: total_pixels / (num_classes * class_count)
    weights = total_pixels / (2.0 * class_counts)
    # Normalize weights so that the weight for class 0 (obstacle) is 1.0
    weights = weights / weights[0]
    
    print(f"    Class counts: Obstacle={class_counts[0]:,}, Walkable={class_counts[1]:,}")
    print(f"    Dynamic class weights: Obstacle={weights[0]:.4f}, Walkable={weights[1]:.4f}")
    
    return torch.tensor(weights, dtype=torch.float32)


class CombinedLoss(nn.Module):
    def __init__(self, weight, device=None):
        super().__init__()
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
    class_weights = compute_class_weights(data_dir / "masks" / "train")
    criterion = CombinedLoss(weight=class_weights, device=device)
    
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
