"""
robustness_eval/01_prepare_grayscale.py
=======================================
Converts an RGB YOLO dataset to a grayscale dataset.

What it does:
  - Reads every image from images/train, images/val, images/test
  - Converts each image to grayscale using OpenCV
  - Saves the result as a 3-channel BGR image (required by YOLO)
  - Copies all YOLO label (.txt) files unchanged
  - Creates a properly formatted data_grayscale.yaml
  - Never modifies the original dataset

Usage:
  python 01_prepare_grayscale.py --dataset_root PATH --eval_root PATH

  --dataset_root : Path to the original YOLO dataset root
                   (must contain images/ and labels/ subdirectories)
  --eval_root    : Path to the robustness_eval/ folder
                   (where results/ will be created)

Example:
  python robustness_eval/01_prepare_grayscale.py \
      --dataset_root "C:/Users/monst/datasets/final_11class_split" \
      --eval_root    "C:/Users/monst/OneDrive/Documents/Naiem/portable alifs cap/VehicleDetector/robustness_eval"
"""

import argparse
import os
import shutil
import sys
import cv2

# ---------------------------------------------------------------------------
# Allow importing config from within or outside the package
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import (
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    VARIANT_GRAYSCALE,
    ensure_dirs,
    get_dataset_dir,
    get_yaml_path,
    get_results_dir,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_splits(dataset_root: str):
    """Yields (split_name, images_dir, labels_dir) for train / val / test."""
    for split in ("train", "val", "test"):
        images_dir = os.path.join(dataset_root, "images", split)
        labels_dir = os.path.join(dataset_root, "labels", split)
        if os.path.isdir(images_dir):
            yield split, images_dir, labels_dir
        else:
            print(f"  [SKIP] Split '{split}' not found at: {images_dir}")


def _is_image(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def convert_to_grayscale(
    src_images_dir: str,
    dst_images_dir: str,
    src_labels_dir: str,
    dst_labels_dir: str,
    split: str,
) -> int:
    """
    Converts all images in src_images_dir to 3-channel grayscale and
    copies labels unchanged.

    Returns the number of images processed.
    """
    ensure_dirs(dst_images_dir, dst_labels_dir)

    image_files = [f for f in os.listdir(src_images_dir) if _is_image(f)]
    total = len(image_files)

    if total == 0:
        print(f"  [WARN] No images found in {src_images_dir}")
        return 0

    print(f"  Processing '{split}': {total} images …")

    for i, filename in enumerate(image_files, start=1):
        # ---- Convert image ------------------------------------------------
        src_img_path = os.path.join(src_images_dir, filename)
        dst_img_path = os.path.join(dst_images_dir, filename)

        img = cv2.imread(src_img_path)
        if img is None:
            print(f"    [WARN] Could not read: {src_img_path} — skipping")
            continue

        gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_bgr  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Back to 3-ch for YOLO
        cv2.imwrite(dst_img_path, gray_bgr)

        # ---- Copy label (unchanged) ----------------------------------------
        stem       = os.path.splitext(filename)[0]
        label_src  = os.path.join(src_labels_dir, stem + ".txt")
        label_dst  = os.path.join(dst_labels_dir, stem + ".txt")
        if os.path.isfile(label_src):
            shutil.copy2(label_src, label_dst)

        # ---- Progress print every 100 images or at end --------------------
        if i % 100 == 0 or i == total:
            print(f"    {i}/{total} done", end="\r")

import numpy as np


def verify_grayscale_images(dst_images_dir: str) -> bool:
    """Verifies that all images in dst_images_dir are 3-channel grayscale images (B == G == R)."""
    if not os.path.isdir(dst_images_dir):
        print(f"  [ERROR] Directory not found for verification: {dst_images_dir}")
        return False

    files = [f for f in os.listdir(dst_images_dir) if _is_image(f)]
    if not files:
        print(f"  [ERROR] No images found to verify in: {dst_images_dir}")
        return False

    for fname in files:
        img_path = os.path.join(dst_images_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [ERROR] Verification failed: Could not read {img_path}")
            return False
        if len(img.shape) != 3 or img.shape[2] != 3:
            print(f"  [ERROR] Verification failed: {fname} does not have 3 channels (shape: {img.shape})")
            return False
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if not (np.array_equal(b, g) and np.array_equal(g, r)):
            print(f"  [ERROR] Verification failed: {fname} is not grayscale (channels B, G, R differ)")
            return False

    print(f"  [OK] Verified {len(files)} images in {os.path.basename(dst_images_dir)} are 3-channel grayscale.")
    return True


def write_grayscale_yaml(
    eval_root: str,
    grayscale_dataset_dir: str,
    nc: int,
    names: list,
) -> str:
    """
    Writes a YOLO-compatible data YAML for the grayscale dataset.
    Returns the path to the written YAML file.
    """
    yaml_path = get_yaml_path(eval_root, VARIANT_GRAYSCALE)
    ensure_dirs(os.path.dirname(yaml_path))

    # Normalize slashes for YOLO compatibility (forward slashes recommended)
    base = grayscale_dataset_dir.replace("\\", "/")

    content = (
        f"# Auto-generated by 01_prepare_grayscale.py\n"
        f"# Grayscale variant of the original YOLO dataset\n\n"
        f"path: {base}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )

    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write(content)

    print(f"\n[OK] YAML written → {yaml_path}")
    return yaml_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an RGB YOLO dataset to grayscale (3-channel)."
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the original YOLO dataset root (contains images/ and labels/).",
    )
    parser.add_argument(
        "--eval_root",
        required=True,
        help="Path to the robustness_eval/ folder (results/ will be created here).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing grayscale dataset if it already exists.",
    )
    args = parser.parse_args()

    # ---- Validate inputs ---------------------------------------------------
    if not os.path.isdir(args.dataset_root):
        print(f"[ERROR] dataset_root does not exist: {args.dataset_root}")
        sys.exit(1)
    if not os.path.isdir(args.eval_root):
        print(f"[ERROR] eval_root does not exist: {args.eval_root}")
        sys.exit(1)

    dataset_root = os.path.abspath(args.dataset_root)
    eval_root    = os.path.abspath(args.eval_root)

    # ---- Derive output directory -------------------------------------------
    grayscale_dir = get_dataset_dir(dataset_root, VARIANT_GRAYSCALE)
    print(f"\n{'='*60}")
    print(f"  Grayscale Dataset Preparation")
    print(f"{'='*60}")
    print(f"  Source      : {dataset_root}")
    print(f"  Destination : {grayscale_dir}")
    print(f"{'='*60}\n")

    if os.path.isdir(grayscale_dir) and not args.force:
        print("[INFO] Grayscale dataset directory already exists.")
        print("       Use --force to overwrite. Skipping conversion.")
    else:
        if os.path.isdir(grayscale_dir) and args.force:
            print("[INFO] --force specified. Clearing existing grayscale dataset …")
            shutil.rmtree(grayscale_dir)

        grand_total = 0
        for split, img_dir, lbl_dir in _iter_splits(dataset_root):
            dst_img = os.path.join(grayscale_dir, "images", split)
            dst_lbl = os.path.join(grayscale_dir, "labels", split)
            count = convert_to_grayscale(img_dir, dst_img, lbl_dir, dst_lbl, split)
            grand_total += count

        print(f"\n[OK] Grayscale conversion complete. Total images processed: {grand_total}")

        # Verification check
        for split in ("train", "val", "test"):
            dst_img = os.path.join(grayscale_dir, "images", split)
            if os.path.isdir(dst_img):
                if not verify_grayscale_images(dst_img):
                    print(f"[ERROR] Verification failed for split '{split}'. Aborting.")
                    sys.exit(1)

    # ---- Write YAML --------------------------------------------------------
    write_grayscale_yaml(
        eval_root=eval_root,
        grayscale_dataset_dir=grayscale_dir,
        nc=len(CLASS_NAMES),
        names=CLASS_NAMES,
    )

    print("\n[DONE] Step 1 complete. Next: run 02_prepare_lowres.py\n")


if __name__ == "__main__":
    main()
