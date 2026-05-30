"""
prepare_ade20k.py
=================
Converts ADE20K 150-class segmentation masks into binary free-space masks
for the Head-Mounted Navigation Assistant.

Binary classes:
    0 = Non-walkable (obstacles, walls, sky, furniture, etc.)
    1 = Walkable (floor, road, sidewalk, ground, path, earth, rug, grass)

Works with:
    • ADE20K SceneParse150 (Kaggle: "ade20k-scene-parsing")
    • Raw ADE20K from MIT CSAIL (RGB-encoded _seg.png)

Usage:
    python prepare_ade20k.py
    python prepare_ade20k.py --ade20k-dir ./data/ade20k --output-dir ./data/freespace
    python prepare_ade20k.py --visualize 20     # show 20 overlay samples
"""

import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

# ADE20K SceneParse150 class indices (0-indexed, as stored in annotation PNGs)
# In the annotation files, pixel value 0 = background (unannotated).
# Class IDs 1–150 map to the 150 object/stuff categories.
# We list the 1-indexed pixel values that represent walkable surfaces.
WALKABLE_PIXEL_VALUES = {
    4,   # floor, flooring
    7,   # road, route
    10,  # grass
    12,  # sidewalk, pavement
    14,  # earth, ground
    18,  # field
    22,  # path
    29,  # rug, carpet
    30,  # stairs (debatable — include for accessibility; remove if too noisy)
    53,  # runway
    55,  # dirt track
}

# Conservative set (remove grass, stairs, field for tighter "paved path" focus):
WALKABLE_CONSERVATIVE = {4, 7, 12, 14, 22, 29}

# Default output structure
DEFAULT_ADE20K_DIR = Path("./data/ade20k")
DEFAULT_OUTPUT_DIR = Path("./data/freespace")

IMG_SIZE = 320  # Resize to match deployment resolution


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def banner(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


def find_ade20k_root(base_dir: Path) -> Path:
    """
    Locate ADE20K data directory.
    
    Supports multiple Kaggle upload structures:
        /data/ade20k/ADEChallengeData2016/
        /data/ade20k/images/training/ + annotations/training/
        /kaggle/input/ade20k*/
    """
    # Check for ADEChallengeData2016 structure (most common on Kaggle)
    for candidate in [
        base_dir / "ADEChallengeData2016",
        base_dir,
        Path("/kaggle/input"),
    ]:
        if not candidate.exists():
            continue
        # Look for the standard structure
        for match in candidate.rglob("images"):
            if (match / "training").exists() or (match / "train").exists():
                return match.parent
        # Also check for flat ADE20K structure
        for match in candidate.rglob("training"):
            if list(match.glob("*.jpg")) or list(match.glob("*.png")):
                return match.parent.parent

    print(f"  [ERROR] Could not find ADE20K data under {base_dir}")
    print(f"  Expected structure:")
    print(f"    {base_dir}/ADEChallengeData2016/images/training/")
    print(f"    {base_dir}/ADEChallengeData2016/annotations/training/")
    print(f"\n  Download from Kaggle: 'ade20k-scene-parsing' dataset")
    sys.exit(1)


def decode_ade20k_mask(mask_path: Path) -> np.ndarray:
    """
    Load an ADE20K annotation mask and return class IDs as a 2D numpy array.
    
    ADE20K SceneParse150 format:
        - Annotations are single-channel PNGs (or indexed-color PNGs)
        - Pixel value 0 = background/unlabeled
        - Pixel values 1-150 = class IDs
    
    Raw ADE20K format (from MIT):
        - _seg.png files encode class in RGB: class_id = (R//10)*256 + G
        - We detect this by checking if the image is RGB
    """
    img = Image.open(mask_path)
    arr = np.array(img)

    if arr.ndim == 2:
        # SceneParse150 format: single-channel, values 0-150
        return arr
    elif arr.ndim == 3:
        # Raw ADE20K RGB-encoded format
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        class_ids = (r.astype(np.int32) // 10) * 256 + g.astype(np.int32)
        return class_ids
    else:
        raise ValueError(f"Unexpected mask shape: {arr.shape} for {mask_path}")


def mask_to_binary(class_mask: np.ndarray, walkable_ids: set) -> np.ndarray:
    """Convert multi-class mask to binary: 1=walkable, 0=non-walkable."""
    return np.isin(class_mask, list(walkable_ids)).astype(np.uint8)


def resize_image(img: Image.Image, size: int) -> Image.Image:
    """Resize image to size×size with bilinear interpolation."""
    return img.resize((size, size), Image.BILINEAR)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Resize mask to size×size with nearest-neighbor (preserves class IDs)."""
    pil_mask = Image.fromarray(mask)
    pil_mask = pil_mask.resize((size, size), Image.NEAREST)
    return np.array(pil_mask)


# ──────────────────────────────────────────────
# DATASET CONVERSION
# ──────────────────────────────────────────────

def convert_split(
    img_dir: Path,
    ann_dir: Path,
    out_img_dir: Path,
    out_mask_dir: Path,
    walkable_ids: set,
    img_size: int = 320,
) -> dict:
    """
    Convert one split (train or val) of ADE20K to binary free-space masks.
    
    Returns stats dict with counts.
    """
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    # Collect image/annotation pairs
    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    
    stats = {"total": 0, "converted": 0, "skipped_no_ann": 0, "has_walkable": 0}

    for img_path in img_files:
        stats["total"] += 1
        
        # Find corresponding annotation
        stem = img_path.stem
        ann_path = None
        for ext in [".png", ".tif", ".tiff"]:
            candidate = ann_dir / f"{stem}{ext}"
            if candidate.exists():
                ann_path = candidate
                break
        
        if ann_path is None:
            stats["skipped_no_ann"] += 1
            continue

        # Decode mask
        try:
            class_mask = decode_ade20k_mask(ann_path)
        except Exception as e:
            print(f"    [WARN] Failed to decode {ann_path}: {e}")
            continue

        # Convert to binary
        binary_mask = mask_to_binary(class_mask, walkable_ids)

        # Resize
        img = Image.open(img_path).convert("RGB")
        img_resized = resize_image(img, img_size)
        mask_resized = resize_mask(binary_mask, img_size)

        # Track walkable coverage
        walkable_frac = mask_resized.sum() / mask_resized.size
        if walkable_frac > 0.01:  # At least 1% walkable
            stats["has_walkable"] += 1

        # Save
        img_resized.save(out_img_dir / f"{stem}.jpg", quality=95)
        # Save mask as single-channel PNG (0 or 1 → scale to 0 or 255 for visibility)
        mask_pil = Image.fromarray(mask_resized * 255)
        mask_pil.save(out_mask_dir / f"{stem}.png")

        stats["converted"] += 1

        if stats["converted"] % 500 == 0:
            print(f"    Processed {stats['converted']}/{stats['total']}...")

    return stats


def prepare_dataset(ade20k_root: Path, output_dir: Path, walkable_ids: set,
                    img_size: int = 320) -> str:
    """
    Full conversion pipeline: ADE20K → binary free-space dataset.
    
    Returns path to generated dataset config YAML.
    """
    banner("PREPARING ADE20K → BINARY FREE SPACE DATASET")

    # Detect directory structure
    # SceneParse150 structure: images/training, annotations/training
    # Alternative: images/train, annotations/train
    img_base = ade20k_root / "images"
    ann_base = ade20k_root / "annotations"

    if not img_base.exists():
        print(f"  [ERROR] Images directory not found: {img_base}")
        sys.exit(1)

    # Determine split directory names
    train_name = "training" if (img_base / "training").exists() else "train"
    val_name = "validation" if (img_base / "validation").exists() else "val"

    for split_name, split_dir_name in [("train", train_name), ("val", val_name)]:
        img_dir = img_base / split_dir_name
        ann_dir = ann_base / split_dir_name

        if not img_dir.exists():
            print(f"  [WARN] Split directory not found: {img_dir}")
            continue

        print(f"\n  Processing {split_name} split ({img_dir})...")
        
        out_img = output_dir / "images" / split_name
        out_mask = output_dir / "masks" / split_name

        stats = convert_split(img_dir, ann_dir, out_img, out_mask, walkable_ids, img_size)

        print(f"    Total images:    {stats['total']}")
        print(f"    Converted:       {stats['converted']}")
        print(f"    Has walkable:    {stats['has_walkable']} ({100*stats['has_walkable']/max(stats['converted'],1):.0f}%)")
        print(f"    Skipped (no ann):{stats['skipped_no_ann']}")

    # Write dataset info YAML (for training script)
    yaml_path = output_dir / "dataset.yaml"
    yaml_content = f"""# Binary Free Space Segmentation Dataset
# Generated from ADE20K SceneParse150
# Classes: 0=non-walkable, 1=walkable

path: {output_dir.resolve()}
train_images: images/train
train_masks:  masks/train
val_images:   images/val
val_masks:    masks/val

nc: 2
names:
  0: non_walkable
  1: walkable

img_size: {img_size}

# Walkable ADE20K pixel values used:
# {sorted(walkable_ids)}
"""
    yaml_path.write_text(yaml_content)
    print(f"\n  ✓ Dataset YAML written: {yaml_path}")

    return str(yaml_path)


# ──────────────────────────────────────────────
# VISUALIZATION
# ──────────────────────────────────────────────

def visualize_samples(output_dir: Path, n_samples: int = 10):
    """
    Create overlay images showing walkable areas highlighted in green.
    Saves to output_dir/visualizations/
    """
    try:
        import cv2
    except ImportError:
        print("  [WARN] OpenCV not installed — skipping visualization.")
        return

    banner("VISUALIZING BINARY MASKS")

    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Use validation split for visualization
    img_dir = output_dir / "images" / "val"
    mask_dir = output_dir / "masks" / "val"

    if not img_dir.exists():
        img_dir = output_dir / "images" / "train"
        mask_dir = output_dir / "masks" / "train"

    img_files = sorted(img_dir.glob("*.jpg"))
    if not img_files:
        print("  No images found for visualization.")
        return

    samples = random.sample(img_files, min(n_samples, len(img_files)))

    for i, img_path in enumerate(samples):
        mask_path = mask_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            continue

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Create green overlay on walkable areas
        overlay = img.copy()
        walkable = mask > 127  # Binary threshold
        overlay[walkable] = (overlay[walkable] * 0.5 + np.array([0, 200, 0]) * 0.5).astype(np.uint8)

        # Add non-walkable red tint
        non_walkable = ~walkable
        overlay[non_walkable] = (overlay[non_walkable] * 0.7 + np.array([0, 0, 100]) * 0.3).astype(np.uint8)

        # Side-by-side: original | overlay
        combined = np.hstack([img, overlay])

        # Add label
        cv2.putText(combined, "Original", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, "Walkable (green) / Obstacle (red)", (img.shape[1] + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Walkable percentage
        pct = 100 * walkable.sum() / walkable.size
        cv2.putText(combined, f"Walkable: {pct:.0f}%", (img.shape[1] + 10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out_path = vis_dir / f"sample_{i:03d}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), combined)

    print(f"  ✓ {len(samples)} visualizations saved to {vis_dir}")
    print(f"  Review these to verify walkable area masks look correct!")


# ──────────────────────────────────────────────
# DATASET STATISTICS
# ──────────────────────────────────────────────

def print_dataset_stats(output_dir: Path):
    """Print summary statistics of the converted dataset."""
    banner("DATASET STATISTICS")

    for split in ["train", "val"]:
        mask_dir = output_dir / "masks" / split
        if not mask_dir.exists():
            continue

        masks = list(mask_dir.glob("*.png"))
        if not masks:
            continue

        walkable_fracs = []
        for mp in random.sample(masks, min(200, len(masks))):
            m = np.array(Image.open(mp))
            frac = (m > 127).sum() / m.size
            walkable_fracs.append(frac)

        arr = np.array(walkable_fracs)
        print(f"  {split} split:")
        print(f"    Images:           {len(masks)}")
        print(f"    Walkable area:    {100*arr.mean():.1f}% ± {100*arr.std():.1f}%")
        print(f"    Min walkable:     {100*arr.min():.1f}%")
        print(f"    Max walkable:     {100*arr.max():.1f}%")
        print(f"    >50% walkable:    {100*(arr > 0.5).sum()/len(arr):.0f}% of images")
        print()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert ADE20K to binary free-space segmentation dataset."
    )
    parser.add_argument("--ade20k-dir", type=Path, default=DEFAULT_ADE20K_DIR,
                        help="Path to ADE20K dataset root")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for binary masks")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE,
                        help="Resize images to this size (default: 320)")
    parser.add_argument("--conservative", action="store_true",
                        help="Use conservative walkable set (no grass/stairs/field)")
    parser.add_argument("--visualize", type=int, default=10,
                        help="Number of samples to visualize (0 to skip)")
    args = parser.parse_args()

    banner("HEAD-MOUNTED NAVIGATION — ADE20K PREPARATION")

    # Determine walkable classes
    walkable_ids = WALKABLE_CONSERVATIVE if args.conservative else WALKABLE_PIXEL_VALUES
    print(f"  Walkable pixel values: {sorted(walkable_ids)}")
    print(f"  Mode: {'conservative (paved only)' if args.conservative else 'inclusive (grass, stairs, field)'}")

    # Find ADE20K root
    ade20k_root = find_ade20k_root(args.ade20k_dir)
    print(f"  ADE20K root: {ade20k_root}")

    # Convert
    yaml_path = prepare_dataset(ade20k_root, args.output_dir, walkable_ids, args.img_size)

    # Stats
    print_dataset_stats(args.output_dir)

    # Visualize
    if args.visualize > 0:
        visualize_samples(args.output_dir, args.visualize)

    banner("NEXT STEP")
    print(f"  1. Review visualizations in {args.output_dir}/visualizations/")
    print(f"  2. If masks look correct, run training:")
    print(f"     python train_freespace.py --data {yaml_path}")
    print()


if __name__ == "__main__":
    main()
