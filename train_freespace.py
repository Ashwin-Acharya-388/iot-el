"""
train_freespace.py
==================
Fine-tunes a DeepLabV3-MobileNetV3-Large model on binary free-space masks
for the Head-Mounted Navigation Assistant.

Model: DeepLabV3 with MobileNetV3-Large backbone (pretrained on COCO)
Task:  Binary semantic segmentation (walkable / non-walkable)
Input: 320×320 RGB image
Output: 320×320 binary mask

Training on Kaggle T4 GPU: ~30 min for 20 epochs on full ADE20K.

Usage:
    python train_freespace.py
    python train_freespace.py --data ./data/freespace/dataset.yaml --epochs 25
    python train_freespace.py --resume ./models/freespace_best.pth
"""

import os
import sys
import time
import json
import shutil
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

DEFAULT_DATA_DIR    = Path("./data/freespace")
DEFAULT_MODEL_DIR   = Path("./models")
DEFAULT_RUNS_DIR    = Path("./runs/freespace")

DEFAULT_EPOCHS      = 25
DEFAULT_BATCH       = 16      # Fits on T4 16GB at 320×320
DEFAULT_LR          = 1e-3    # Fine-tuning LR
DEFAULT_IMG_SIZE    = 320
DEFAULT_WORKERS     = 4

NUM_CLASSES         = 2       # walkable / non-walkable


# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────

class FreespaceDataset(Dataset):
    """
    Binary free-space segmentation dataset.
    
    Expects:
        images/<split>/*.jpg  — RGB input images (320×320)
        masks/<split>/*.png   — Binary masks (0=obstacle, 255=walkable)
    """

    def __init__(self, img_dir: Path, mask_dir: Path, img_size: int = 320,
                 augment: bool = True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment

        # Collect pairs
        self.samples = []
        for img_path in sorted(img_dir.glob("*.jpg")):
            mask_path = mask_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                self.samples.append((img_path, mask_path))

        if not self.samples:
            raise FileNotFoundError(
                f"No image/mask pairs found in {img_dir} + {mask_dir}"
            )

        # Normalization (ImageNet stats — matches MobileNetV3 pretrained weights)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Load image and mask
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Single channel

        # Resize (should already be 320×320 from prepare_ade20k.py)
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # Augmentation (applied consistently to image AND mask)
        if self.augment:
            img, mask = self._augment(img, mask)

        # Convert to tensor
        img_tensor = transforms.ToTensor()(img)  # [3, H, W] in [0, 1]
        img_tensor = self.normalize(img_tensor)

        mask_np = np.array(mask)
        mask_tensor = torch.from_numpy((mask_np > 127).astype(np.int64))  # Binary: 0 or 1

        return img_tensor, mask_tensor

    def _augment(self, img: Image.Image, mask: Image.Image):
        """Apply synchronized augmentations to image and mask."""
        import random

        # Random horizontal flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Random resized crop (scale 0.7–1.0)
        if random.random() > 0.3:
            scale = random.uniform(0.7, 1.0)
            new_size = int(self.img_size * scale)
            x = random.randint(0, self.img_size - new_size)
            y = random.randint(0, self.img_size - new_size)
            img = img.crop((x, y, x + new_size, y + new_size))
            mask = mask.crop((x, y, x + new_size, y + new_size))
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # Random brightness/contrast (image only, not mask)
        if random.random() > 0.5:
            from torchvision.transforms import functional as TF
            img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
            img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))

        # Random color jitter (image only)
        if random.random() > 0.5:
            from torchvision.transforms import functional as TF
            img = TF.adjust_saturation(img, random.uniform(0.7, 1.3))
            img = TF.adjust_hue(img, random.uniform(-0.1, 0.1))

        return img, mask


# ──────────────────────────────────────────────
# LOSS FUNCTIONS
# ──────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Better than cross-entropy alone for imbalanced segmentation
    (walkable area often dominates or is very sparse).
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: [B, 2, H, W] logits → softmax → take walkable channel
        pred_soft = F.softmax(pred, dim=1)[:, 1]  # [B, H, W]
        target_float = target.float()              # [B, H, W]

        intersection = (pred_soft * target_float).sum(dim=(1, 2))
        union = pred_soft.sum(dim=(1, 2)) + target_float.sum(dim=(1, 2))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combined Cross-Entropy + Dice loss.
    
    CE provides stable gradients; Dice handles class imbalance.
    """
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(pred, target) + self.dice_weight * self.dice(pred, target)


# ──────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────

def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> dict:
    """
    Compute per-class IoU and mean IoU.
    
    pred:   [B, H, W] class predictions (argmax of logits)
    target: [B, H, W] ground truth class IDs
    """
    ious = {}
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        iou = intersection / max(union, 1)
        ious[cls] = iou

    ious["mean"] = sum(ious.values()) / num_classes
    return ious


# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────

def create_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Create DeepLabV3 with MobileNetV3-Large backbone.
    
    The pretrained model is trained on COCO (21 classes).
    We replace the classifier head for 2 classes (walkable / non-walkable).
    
    Parameters: ~11M total, ~3.5M trainable (with frozen backbone option)
    """
    if pretrained:
        model = deeplabv3_mobilenet_v3_large(
            weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        )
    else:
        model = deeplabv3_mobilenet_v3_large(weights=None)

    # Replace classifier for binary segmentation
    # The classifier is: Sequential(Conv2d(256, 256), BN, ReLU, Conv2d(256, num_classes))
    in_channels = model.classifier[4].in_channels  # 256
    model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    # Also replace aux classifier if present
    if model.aux_classifier is not None:
        aux_in = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(aux_in, num_classes, kernel_size=1)

    return model


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────

def banner(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


def detect_device(requested: str) -> torch.device:
    """Auto-detect best device."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"  ✓ CUDA available: {name}")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  ✓ Apple Silicon MPS available")
        return torch.device("mps")
    else:
        print("  ⚠ No GPU found — training on CPU (slow)")
        return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0
    n_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        output = model(images)["out"]  # [B, 2, H, W]

        # Ensure output matches mask size
        if output.shape[-2:] != masks.shape[-2:]:
            output = F.interpolate(output, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        loss = criterion(output, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate and return loss + IoU metrics."""
    model.eval()
    total_loss = 0
    n_batches = 0
    all_ious = {"non_walkable": [], "walkable": [], "mean": []}

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        output = model(images)["out"]
        if output.shape[-2:] != masks.shape[-2:]:
            output = F.interpolate(output, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        loss = criterion(output, masks)
        total_loss += loss.item()
        n_batches += 1

        # IoU
        preds = output.argmax(dim=1)  # [B, H, W]
        ious = compute_iou(preds, masks)
        all_ious["non_walkable"].append(ious[0])
        all_ious["walkable"].append(ious[1])
        all_ious["mean"].append(ious["mean"])

    avg_loss = total_loss / max(n_batches, 1)
    avg_ious = {k: np.mean(v) for k, v in all_ious.items()}
    return avg_loss, avg_ious


def train(args):
    """Full training loop."""
    device = detect_device(args.device)
    data_dir = Path(args.data_dir)

    banner("FREE SPACE SEGMENTATION TRAINING")
    print(f"  Data:      {data_dir}")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Batch:     {args.batch}")
    print(f"  LR:        {args.lr}")
    print(f"  Img size:  {args.img_size}×{args.img_size}")
    print(f"  Device:    {device}")

    # ── Datasets ──
    train_ds = FreespaceDataset(
        img_dir=data_dir / "images" / "train",
        mask_dir=data_dir / "masks" / "train",
        img_size=args.img_size,
        augment=True,
    )
    val_ds = FreespaceDataset(
        img_dir=data_dir / "images" / "val",
        mask_dir=data_dir / "masks" / "val",
        img_size=args.img_size,
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples:   {len(val_ds)}")

    # ── Model ──
    model = create_model(num_classes=NUM_CLASSES, pretrained=True)

    if args.resume:
        print(f"  Resuming from: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))

    model = model.to(device)

    # ── Optimizer & Scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)

    # ── Training loop ──
    model_dir = Path(args.model_dir)
    runs_dir = Path(args.runs_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    best_iou = 0.0
    best_epoch = 0
    history = []
    patience_counter = 0

    banner("TRAINING BEGINS")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_loss, val_ious = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0

        # Log
        walkable_iou = val_ious["walkable"]
        mean_iou = val_ious["mean"]

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Walkable IoU: {100*walkable_iou:.1f}% | "
              f"Mean IoU: {100*mean_iou:.1f}% | "
              f"LR: {lr:.2e} | "
              f"Time: {elapsed:.0f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "walkable_iou": walkable_iou,
            "mean_iou": mean_iou,
            "lr": lr,
        })

        # Save best
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_epoch = epoch
            patience_counter = 0
            best_path = model_dir / "freespace_best.pth"
            torch.save(model.state_dict(), best_path)
            print(f"    ★ New best model! Mean IoU: {100*best_iou:.1f}% → {best_path}")
        else:
            patience_counter += 1

        # Save periodic checkpoint
        if epoch % 5 == 0:
            ckpt_path = runs_dir / f"checkpoint_epoch{epoch}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou": best_iou,
            }, ckpt_path)

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n  ⏹ Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # ── Final report ──
    banner("TRAINING COMPLETE")
    print(f"  Best Mean IoU:      {100*best_iou:.1f}%")
    print(f"  Best epoch:         {best_epoch}")
    print(f"  Model saved:        {model_dir / 'freespace_best.pth'}")

    # Save history
    history_path = runs_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Training history:   {history_path}")

    return model_dir / "freespace_best.pth"


# ──────────────────────────────────────────────
# ONNX EXPORT & INT8 QUANTIZATION
# ──────────────────────────────────────────────

def export_onnx(model_path: Path, output_path: Path, img_size: int = 320):
    """Export trained model to ONNX format."""
    banner("EXPORTING TO ONNX")

    model = create_model(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, img_size, img_size)

    # Wrap model to only return 'out' key
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x)["out"]

    wrapped = ModelWrapper(model)
    wrapped.eval()

    torch.onnx.export(
        wrapped,
        dummy,
        str(output_path),
        input_names=["image"],
        output_names=["mask"],
        opset_version=13,
        dynamic_axes=None,  # Fixed 320×320 input
    )

    sz = output_path.stat().st_size / 1024**2
    print(f"  ✓ ONNX model exported: {output_path} ({sz:.1f} MB)")

    # Simplify ONNX graph
    try:
        import onnxsim
        import onnx
        model_onnx = onnx.load(str(output_path))
        model_simplified, check = onnxsim.simplify(model_onnx)
        if check:
            onnx.save(model_simplified, str(output_path))
            sz = output_path.stat().st_size / 1024**2
            print(f"  ✓ Simplified ONNX: {sz:.1f} MB")
    except ImportError:
        print("  [INFO] onnxsim not installed — skipping simplification")

    return output_path


def quantize_int8(fp32_onnx: Path, int8_onnx: Path, calib_dir: Path, n_calib: int = 200):
    """Quantize ONNX model to INT8 using static quantization."""
    banner("INT8 QUANTIZATION")

    try:
        import onnxruntime as ort
        from onnxruntime.quantization import (
            quantize_static, QuantFormat, QuantType, CalibrationMethod,
        )
    except ImportError:
        print("  [ERROR] onnxruntime.quantization not available")
        print("  Install: pip install onnxruntime onnxruntime-tools")
        return None

    import cv2

    class CalibrationReader:
        def __init__(self, img_dir, input_name, img_size=320, n=200):
            exts = {".jpg", ".jpeg", ".png"}
            self.paths = [p for p in Path(img_dir).rglob("*") if p.suffix.lower() in exts][:n]
            self.input_name = input_name
            self.img_size = img_size
            self.idx = 0
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        def get_next(self):
            if self.idx >= len(self.paths):
                return None
            img = cv2.imread(str(self.paths[self.idx]))
            self.idx += 1
            if img is None:
                return self.get_next()
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = (img - self.mean) / self.std
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, 0)
            return {self.input_name: np.ascontiguousarray(img)}

        def rewind(self):
            self.idx = 0

    # Get input name from model
    sess = ort.InferenceSession(str(fp32_onnx), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    reader = CalibrationReader(calib_dir, input_name, n=n_calib)

    quantize_static(
        model_input=str(fp32_onnx),
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
    print(f"  ✓ INT8 model saved: {int8_onnx} ({sz:.1f} MB)")
    return int8_onnx


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train free-space segmentation model.")
    parser.add_argument("--data-dir",   type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--model-dir",  type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--runs-dir",   type=str, default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--epochs",     type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch",      type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    parser.add_argument("--img-size",   type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--workers",    type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--patience",   type=int, default=8)
    parser.add_argument("--device",     type=str, default="auto")
    parser.add_argument("--resume",     type=str, default=None)
    parser.add_argument("--export-only", action="store_true",
                        help="Skip training, just export existing model to ONNX+INT8")
    args = parser.parse_args()

    if args.export_only:
        # Export mode
        model_path = Path(args.model_dir) / "freespace_best.pth"
        if not model_path.exists():
            print(f"  [ERROR] Model not found: {model_path}")
            sys.exit(1)

        fp32_onnx = Path(args.model_dir) / "freespace_fp32.onnx"
        export_onnx(model_path, fp32_onnx, args.img_size)

        int8_onnx = Path(args.model_dir) / "freespace_int8.onnx"
        calib_dir = Path(args.data_dir) / "images" / "val"
        quantize_int8(fp32_onnx, int8_onnx, calib_dir)

        banner("EXPORT COMPLETE")
        print(f"  FP32 ONNX: {fp32_onnx}")
        print(f"  INT8 ONNX: {int8_onnx}")
        print(f"\n  Transfer to RPi:")
        print(f"    scp {int8_onnx} pi@<RPI_IP>:~/navigation/models/")
        print(f"    python navigation_freespace_rpi.py --model ./models/freespace_int8.onnx")
        return

    # Full training pipeline
    best_model = train(args)

    # Export to ONNX
    fp32_onnx = Path(args.model_dir) / "freespace_fp32.onnx"
    export_onnx(best_model, fp32_onnx, args.img_size)

    # INT8 quantization
    int8_onnx = Path(args.model_dir) / "freespace_int8.onnx"
    calib_dir = Path(args.data_dir) / "images" / "val"
    quantize_int8(fp32_onnx, int8_onnx, calib_dir)

    banner("ALL DONE — NEXT STEPS")
    print(f"  1. Transfer model to RPi:")
    print(f"     scp {int8_onnx} pi@<RPI_IP>:~/navigation/models/")
    print(f"  2. Run navigation system:")
    print(f"     python navigation_freespace_rpi.py --model ./models/freespace_int8.onnx")
    print()


if __name__ == "__main__":
    main()
