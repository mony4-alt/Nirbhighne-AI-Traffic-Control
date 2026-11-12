"""
robustness_eval/03_train.py
============================
Trains a fresh YOLOv8m model on one of the dataset variants.

This script:
  - Initialises a new YOLOv8m model (downloads pretrained backbone if needed)
  - Trains using the hyperparameters defined in config.py
  - Copies the resulting best.pt to results/weights/best_<variant>.pt
  - Does NOT touch the original model (weights/best.pt)

Variants:
  grayscale : Train on the grayscale dataset
  lowres    : Train on the low-resolution dataset
  rgb       : Train on the original RGB dataset (optional baseline retraining)

Usage:
  python 03_train.py --variant grayscale --dataset_root PATH --eval_root PATH

  --variant      : One of: rgb, grayscale, lowres
  --dataset_root : Path to the ORIGINAL dataset root
  --eval_root    : Path to the robustness_eval/ folder
  --resume       : (optional) Path to a checkpoint to resume from

Example:
  python robustness_eval/03_train.py \
      --variant grayscale \
      --dataset_root "C:/Users/monst/datasets/final_11class_split" \
      --eval_root    "C:/Users/monst/OneDrive/Documents/Naiem/portable alifs cap/VehicleDetector/robustness_eval"
"""

import argparse
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import (
    TRAIN_CONFIG,
    VARIANT_RGB,
    VARIANT_GRAYSCALE,
    VARIANT_LOWRES,
    VARIANT_WEIGHT_NAMES,
    ensure_dirs,
    get_device,
    get_runs_dir,
    get_weights_dir,
    get_yaml_path,
)

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics is not installed. Run: pip install ultralytics>=8.2.0")
    sys.exit(1)


VALID_VARIANTS = (VARIANT_RGB, VARIANT_GRAYSCALE, VARIANT_LOWRES)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_variant(
    variant: str,
    eval_root: str,
    dataset_root: str,
    resume_path: str | None = None,
) -> str:
    """
    Trains YOLOv8m on the specified variant's dataset.

    Returns the path to the saved best_<variant>.pt file.
    """
    yaml_path   = get_yaml_path(eval_root, variant)
    weights_dir = get_weights_dir(eval_root)
    runs_dir    = get_runs_dir(eval_root)
    device      = get_device()

    ensure_dirs(weights_dir, runs_dir)

    # ---- Verify YAML exists ------------------------------------------------
    if not os.path.isfile(yaml_path):
        print(f"[ERROR] data YAML not found: {yaml_path}")
        if variant == VARIANT_RGB:
            print("        Run with --variant rgb first, or supply a pre-existing rgb YAML.")
        else:
            step = "01" if variant == VARIANT_GRAYSCALE else "02"
            print(f"        Please run {step}_prepare_{variant}.py first.")
        sys.exit(1)

    # ---- Initialise model --------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Training YOLOv8m — variant: {variant.upper()}")
    print(f"{'='*60}")
    print(f"  Data YAML : {yaml_path}")
    print(f"  Device    : {device.upper()}")
    print(f"  Epochs    : {TRAIN_CONFIG['epochs']}")
    print(f"  Img size  : {TRAIN_CONFIG['imgsz']}")
    print(f"  Batch     : {TRAIN_CONFIG['batch']}")
    print(f"  Optimizer : {TRAIN_CONFIG['optimizer']}")
    print(f"{'='*60}\n")

    if resume_path:
        print(f"[INFO] Resuming from checkpoint: {resume_path}")
        model = YOLO(resume_path)
    else:
        # Always start from the official pretrained YOLOv8m backbone
        model = YOLO("yolov8m.pt")

    # ---- Run training ------------------------------------------------------
    run_name = f"train_{variant}"

    results = model.train(
        data      = yaml_path,
        epochs    = TRAIN_CONFIG["epochs"],
        imgsz     = TRAIN_CONFIG["imgsz"],
        batch     = TRAIN_CONFIG["batch"],
        optimizer = TRAIN_CONFIG["optimizer"],
        patience  = TRAIN_CONFIG["patience"],
        workers   = TRAIN_CONFIG["workers"],
        seed      = TRAIN_CONFIG["seed"],
        verbose   = TRAIN_CONFIG["verbose"],
        exist_ok  = TRAIN_CONFIG["exist_ok"],
        val       = TRAIN_CONFIG["val"],
        save      = TRAIN_CONFIG["save"],
        plots     = TRAIN_CONFIG["plots"],
        device    = device,
        project   = runs_dir,
        name      = run_name,
        resume    = bool(resume_path),
    )

    # ---- Copy best.pt to our weights directory ----------------------------
    # Ultralytics saves best.pt inside project/name/weights/best.pt
    run_weights_dir  = os.path.join(runs_dir, run_name, "weights")
    src_best         = os.path.join(run_weights_dir, "best.pt")
    dst_name         = VARIANT_WEIGHT_NAMES[variant]
    dst_best         = os.path.join(weights_dir, dst_name)

    if os.path.isfile(src_best):
        shutil.copy2(src_best, dst_best)
        print(f"\n[OK] Best checkpoint saved → {dst_best}")
    else:
        print(f"\n[WARN] Could not find best.pt at {src_best}")
        print("       Please manually copy the best checkpoint to the weights/ directory.")

    return dst_best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8m on a dataset variant (grayscale or lowres)."
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=VALID_VARIANTS,
        help="Dataset variant to train on.",
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the ORIGINAL YOLO dataset root.",
    )
    parser.add_argument(
        "--eval_root",
        required=True,
        help="Path to the robustness_eval/ folder.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="(Optional) Path to a checkpoint file to resume training from.",
    )
    args = parser.parse_args()

    # ---- Validate inputs ---------------------------------------------------
    if not os.path.isdir(args.dataset_root):
        print(f"[ERROR] dataset_root does not exist: {args.dataset_root}")
        sys.exit(1)
    if not os.path.isdir(args.eval_root):
        print(f"[ERROR] eval_root does not exist: {args.eval_root}")
        sys.exit(1)
    if args.resume and not os.path.isfile(args.resume):
        print(f"[ERROR] Resume checkpoint not found: {args.resume}")
        sys.exit(1)

    saved_path = train_variant(
        variant      = args.variant,
        eval_root    = os.path.abspath(args.eval_root),
        dataset_root = os.path.abspath(args.dataset_root),
        resume_path  = args.resume,
    )

    variant = args.variant
    next_variant = VARIANT_LOWRES if variant == VARIANT_GRAYSCALE else None
    print(f"\n[DONE] Training complete for variant: {variant}")
    if next_variant:
        print(f"       Next: run 03_train.py --variant {next_variant}")
    print(f"       Then: run 04_evaluate.py for all three models\n")


if __name__ == "__main__":
    main()
