"""
download_datasets.py
====================
Downloads, extracts, and converts the Cityscapes dataset to YOLO format
for fine-tuning YOLOv8n on sidewalk/obstacle detection.

Cityscapes requires FREE registration at: https://www.cityscapes-dataset.com/
The script will guide you through the download process.

Usage:
    python download_datasets.py
    python download_datasets.py --manual   # If you've already downloaded manually
    python download_datasets.py --mapillary  # Also download Mapillary Vistas
"""

import os
import sys
import json
import shutil
import argparse
import zipfile
import tarfile
import random
import getpass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Installing required packages...")
    os.system("pip install requests tqdm")
    import requests
    from tqdm import tqdm

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

DATA_ROOT = Path("./data")
CITYSCAPES_DIR = DATA_ROOT / "cityscapes"
YOLO_DIR = DATA_ROOT / "yolo_cityscapes"

# Cityscapes → YOLO class mapping
# We keep only INSTANCE-LEVEL object detection classes.
# Surface/region classes (road, sidewalk, curb, wall, fence) are REMOVED because
# their polygons span huge image areas → degenerate bounding boxes that destroy
# detector accuracy. Every Cityscapes detection benchmark uses these 8-11 classes.
CITYSCAPES_CLASSES = {
    # Class Name              : YOLO idx
    "person":                   0,
    "rider":                    1,   # cyclist / scooter rider
    "car":                      2,
    "truck":                    3,
    "bus":                      4,
    "train":                    5,
    "motorcycle":               6,
    "bicycle":                  7,
    "traffic light":            8,
    "traffic sign":             9,
    "pole":                    10,
}

YOLO_CLASS_NAMES = list(CITYSCAPES_CLASSES.keys())

# Cityscapes labelIds for each semantic category (from official trainIds)
# Reference: https://github.com/mcordts/cityscapesScripts
# Only instance-level detection classes are included.
CITYSCAPES_LABEL_IDS = {
    17: "pole",
    19: "traffic light",
    20: "traffic sign",
    24: "person",
    25: "rider",
    26: "car",
    27: "truck",
    28: "bus",
    29: "train",
    30: "motorcycle",
    31: "bicycle",
}

# Map label IDs to our YOLO classes (skip anything not in CITYSCAPES_CLASSES)
LABEL_TO_YOLO = {}
for lid, name in CITYSCAPES_LABEL_IDS.items():
    if name in CITYSCAPES_CLASSES:
        LABEL_TO_YOLO[lid] = CITYSCAPES_CLASSES[name]

CITYSCAPES_API_URL = "https://www.cityscapes-dataset.com/file-handling/"


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def banner(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


def download_with_progress(url: str, dest: Path, cookies: dict) -> bool:
    """Stream-download a file from Cityscapes with progress bar."""
    try:
        resp = requests.get(url, stream=True, cookies=cookies, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        return False


def login_cityscapes(username: str, password: str) -> Optional[dict]:
    """Authenticate with Cityscapes and return session cookies."""
    login_url = "https://www.cityscapes-dataset.com/login/"
    session = requests.Session()
    try:
        # Get CSRF token
        resp = session.get(login_url, timeout=30)
        csrf = None
        for line in resp.text.split("\n"):
            if "csrfmiddlewaretoken" in line:
                csrf = line.split('value="')[1].split('"')[0]
                break
        if not csrf:
            print("  [WARN] Could not extract CSRF token; trying without it.")

        payload = {
            "username": username,
            "password": password,
            "csrfmiddlewaretoken": csrf or "",
            "next": "/downloads/",
        }
        headers = {"Referer": login_url}
        resp = session.post(login_url, data=payload, headers=headers, timeout=30)
        if "logout" in resp.text.lower() or resp.url.endswith("/downloads/"):
            print("  ✓ Login successful.")
            return dict(session.cookies)
        else:
            print("  [ERROR] Login failed. Check your credentials.")
            return None
    except Exception as e:
        print(f"  [ERROR] Login error: {e}")
        return None


# ──────────────────────────────────────────────
# DATASET DOWNLOAD
# ──────────────────────────────────────────────

def download_cityscapes(manual: bool = False) -> bool:
    """
    Download the Cityscapes dataset.
    Returns True if files are present and ready.
    """
    banner("CITYSCAPES DATASET DOWNLOAD")

    raw_dir = CITYSCAPES_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Files we need: left images + fine annotations
    needed_files = {
        "leftImg8bit_trainvaltest.zip": "2975 training images",
        "gtFine_trainvaltest.zip":      "Fine pixel annotations",
    }

    # Check if already present
    all_present = all((raw_dir / f).exists() for f in needed_files)
    if all_present:
        print("  ✓ Cityscapes archives already downloaded.")
        return True

    print("""
  Cityscapes requires FREE registration for academic/research use.

  STEPS:
    1. Go to: https://www.cityscapes-dataset.com/register/
    2. Register with your academic email
    3. Wait for activation email (usually instant)
    4. Come back here and enter your credentials

  Files needed (~3 GB total):
    • leftImg8bit_trainvaltest.zip  (~11 GB raw, ~2 GB compressed)
    • gtFine_trainvaltest.zip       (~240 MB)

  ALTERNATIVE (manual download):
    Download both files from https://www.cityscapes-dataset.com/downloads/
    Place them in: ./data/cityscapes/raw/
    Then re-run with: python download_datasets.py --manual
""")

    if manual:
        print("  Manual mode: checking for existing archives...")
        missing = [f for f in needed_files if not (raw_dir / f).exists()]
        if missing:
            print(f"  [ERROR] Missing files: {missing}")
            print(f"  Place them in: {raw_dir.resolve()}")
            return False
        return True

    username = input("  Cityscapes username (email): ").strip()
    password = getpass.getpass("  Cityscapes password: ")

    cookies = login_cityscapes(username, password)
    if not cookies:
        return False

    # File IDs from Cityscapes download page
    file_ids = {
        "leftImg8bit_trainvaltest.zip": "?packageID=3",
        "gtFine_trainvaltest.zip":      "?packageID=1",
    }

    for filename, pkg_id in file_ids.items():
        dest = raw_dir / filename
        if dest.exists():
            print(f"  ✓ {filename} already exists, skipping.")
            continue
        url = f"{CITYSCAPES_API_URL}{pkg_id}"
        print(f"\n  Downloading {filename}...")
        if not download_with_progress(url, dest, cookies):
            print(f"\n  [FALLBACK] Automated download failed for {filename}.")
            print(f"  Please download manually from:")
            print(f"  https://www.cityscapes-dataset.com/downloads/")
            print(f"  Save to: {dest.resolve()}")
            return False

    return True


def extract_cityscapes() -> bool:
    """Extract downloaded Cityscapes zip archives."""
    raw_dir = CITYSCAPES_DIR / "raw"
    extract_dir = CITYSCAPES_DIR / "extracted"

    if (extract_dir / "leftImg8bit").exists():
        print("  ✓ Cityscapes already extracted.")
        return True

    extract_dir.mkdir(parents=True, exist_ok=True)

    for zf_name in ["leftImg8bit_trainvaltest.zip", "gtFine_trainvaltest.zip"]:
        zf_path = raw_dir / zf_name
        if not zf_path.exists():
            print(f"  [ERROR] Missing archive: {zf_path}")
            return False

        print(f"  Extracting {zf_name}...")
        with zipfile.ZipFile(zf_path, "r") as zf:
            members = zf.namelist()
            for member in tqdm(members, desc=f"  {zf_name}"):
                zf.extract(member, extract_dir)

    print("  ✓ Extraction complete.")
    return True


# ──────────────────────────────────────────────
# CITYSCAPES → YOLO CONVERSION
# ──────────────────────────────────────────────

def polygon_to_bbox(polygon: list) -> tuple:
    """Convert a Cityscapes polygon to (x_min, y_min, x_max, y_max)."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h) -> tuple:
    """Convert absolute bbox to YOLO normalized (cx, cy, w, h)."""
    cx = (x_min + x_max) / 2 / img_w
    cy = (y_min + y_max) / 2 / img_h
    w  = (x_max - x_min) / img_w
    h  = (y_max - y_min) / img_h
    return cx, cy, w, h


# Bounding box quality thresholds
MIN_BOX_PX       = 10      # Minimum box dimension in pixels
MAX_AREA_RATIO   = 0.30    # Skip boxes covering >30% of image area
MAX_ASPECT_RATIO = 8.0     # Skip boxes with extreme aspect ratios (annotation artifacts)


def convert_annotation(json_path: Path, img_w: int = 2048, img_h: int = 1024) -> list:
    """
    Parse a Cityscapes instanceIds JSON file and return YOLO label lines.
    Cityscapes fine annotations have per-object polygons with label names.
    Includes quality filtering to skip degenerate bounding boxes.
    """
    with open(json_path) as f:
        data = json.load(f)

    lines = []
    for obj in data.get("objects", []):
        label = obj.get("label", "").lower()

        # Map compound labels (e.g. "persongroup" → "person")
        if label.startswith("person"):
            label = "person"
        elif label.startswith("rider"):
            label = "rider"
        elif label.startswith("car"):
            label = "car"
        elif label.startswith("truck"):
            label = "truck"
        elif label.startswith("bus"):
            label = "bus"
        elif label.startswith("motorcycle"):
            label = "motorcycle"
        elif label.startswith("bicycle"):
            label = "bicycle"
        elif label.startswith("train"):
            label = "train"

        if label not in CITYSCAPES_CLASSES:
            continue

        class_id = CITYSCAPES_CLASSES[label]
        polygon   = obj.get("polygon", [])
        if len(polygon) < 3:
            continue

        x1, y1, x2, y2 = polygon_to_bbox(polygon)

        box_w = x2 - x1
        box_h = y2 - y1

        # Skip boxes that are too small (< MIN_BOX_PX pixels)
        if box_w < MIN_BOX_PX or box_h < MIN_BOX_PX:
            continue

        # Skip boxes with extreme aspect ratios (annotation artifacts)
        aspect = max(box_w, box_h) / max(min(box_w, box_h), 1)
        if aspect > MAX_ASPECT_RATIO:
            continue

        # Skip boxes covering too much of the image (degenerate region annotations)
        area_ratio = (box_w * box_h) / (img_w * img_h)
        if area_ratio > MAX_AREA_RATIO:
            continue

        cx, cy, w, h = bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)
        # Clamp to [0, 1]
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w  = max(0, min(1, w))
        h  = max(0, min(1, h))

        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


def build_yolo_dataset(val_fraction: float = 0.2) -> bool:
    """
    Walk extracted Cityscapes directory, convert all annotations,
    and create YOLO-format dataset with train/val split.
    """
    banner("CONVERTING TO YOLO FORMAT")

    extract_dir = CITYSCAPES_DIR / "extracted"
    img_root    = extract_dir / "leftImg8bit"
    ann_root    = extract_dir / "gtFine"

    if not img_root.exists() or not ann_root.exists():
        print(f"  [ERROR] Extracted data not found at {extract_dir}")
        return False

    # Collect all image/annotation pairs
    pairs = []
    for split in ["train", "val", "test"]:
        img_split = img_root / split
        if not img_split.exists():
            continue
        for city_dir in sorted(img_split.iterdir()):
            for img_file in sorted(city_dir.glob("*_leftImg8bit.png")):
                stem = img_file.stem.replace("_leftImg8bit", "")
                ann_file = (ann_root / split / city_dir.name
                            / f"{stem}_gtFine_polygons.json")
                if ann_file.exists():
                    pairs.append((img_file, ann_file))

    print(f"  Found {len(pairs)} image/annotation pairs.")

    if not pairs:
        print("  [ERROR] No pairs found. Check extraction.")
        return False

    # Shuffle and split
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
        for img_path, ann_path in tqdm(split_pairs, desc=f"  {split_name}"):
            # Convert annotation
            label_lines = convert_annotation(ann_path)

            if not label_lines:
                skipped += 1
                continue

            # Symlink/copy image
            dst_img = img_out / img_path.name
            if not dst_img.exists():
                try:
                    dst_img.symlink_to(img_path.resolve())
                except Exception:
                    shutil.copy2(img_path, dst_img)

            # Write label file
            lbl_file = lbl_out / (img_path.stem + ".txt")
            lbl_file.write_text("\n".join(label_lines))

        print(f"    {split_name}: {len(split_pairs) - skipped} usable, {skipped} skipped (no target classes).")

    return True


def write_dataset_yaml() -> None:
    """Write dataset.yaml for YOLO training."""
    yaml_path = YOLO_DIR / "dataset.yaml"
    content = f"""# Cityscapes YOLO dataset config
# Generated by download_datasets.py

path: {YOLO_DIR.resolve()}
train: images/train
val:   images/val

nc: {len(YOLO_CLASS_NAMES)}
names: {YOLO_CLASS_NAMES}
"""
    yaml_path.write_text(content)
    print(f"  ✓ dataset.yaml written → {yaml_path.resolve()}")


# ──────────────────────────────────────────────
# OPTIONAL: MAPILLARY VISTAS
# ──────────────────────────────────────────────

def download_mapillary() -> None:
    """Guide user to download Mapillary Vistas (optional secondary dataset)."""
    banner("MAPILLARY VISTAS (OPTIONAL)")
    print("""
  Mapillary Vistas is a large-scale street-level imagery dataset with
  rich annotations including 124 semantic classes.

  Download instructions:
    1. Create account at: https://www.mapillary.com/dataset/vistas
    2. Request access (usually approved within 24h)
    3. Download 'mapillary_vistas_v2.0.zip' (~25 GB)
    4. Place in ./data/mapillary/raw/

  NOTE: Mapillary integration into this pipeline is optional.
  The Cityscapes dataset alone is sufficient for good navigation performance.

  If you want to add Mapillary later, run:
      python download_datasets.py --mapillary --manual
""")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download and prepare navigation datasets.")
    parser.add_argument("--manual",    action="store_true", help="Skip download, use pre-placed archives")
    parser.add_argument("--mapillary", action="store_true", help="Show Mapillary download instructions")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extraction if already done")
    args = parser.parse_args()

    banner("HEAD-MOUNTED NAVIGATION ASSISTANT — DATASET PREPARATION")
    print(f"  Output directory: {YOLO_DIR.resolve()}")

    # Step 1: Download Cityscapes
    if not download_cityscapes(manual=args.manual):
        print("\n[FAIL] Could not obtain Cityscapes archives.")
        sys.exit(1)

    # Step 2: Extract
    if not args.skip_extract:
        if not extract_cityscapes():
            print("\n[FAIL] Extraction failed.")
            sys.exit(1)

    # Step 3: Convert to YOLO format
    if not build_yolo_dataset():
        print("\n[FAIL] Conversion failed.")
        sys.exit(1)

    # Step 4: Write YAML
    write_dataset_yaml()

    # Optional: Mapillary
    if args.mapillary:
        download_mapillary()

    banner("DONE")
    print(f"  Dataset ready at: {YOLO_DIR.resolve()}")
    print(f"  Classes ({len(YOLO_CLASS_NAMES)}): {YOLO_CLASS_NAMES}")
    print(f"\n  Next step:\n    python train_baseline.py\n")


if __name__ == "__main__":
    main()
