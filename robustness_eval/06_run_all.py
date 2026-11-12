"""
robustness_eval/06_run_all.py
==============================
Master orchestrator for the complete robustness evaluation pipeline.

Runs all steps in sequence:
  Step 1: Generate YOLO YAML files for all variants
  Step 2: Convert dataset to grayscale (01_prepare_grayscale.py)
  Step 3: Create low-resolution dataset (02_prepare_lowres.py)
  Step 4: Train on grayscale dataset (03_train.py --variant grayscale)
  Step 5: Train on low-res dataset (03_train.py --variant lowres)
  Step 6: Evaluate original RGB model (04_evaluate.py --label rgb)
  Step 7: Evaluate grayscale model (04_evaluate.py --label grayscale)
  Step 8: Evaluate low-res model (04_evaluate.py --label lowres)
  Step 9: Generate comparison tables and charts (05_compare.py)

Usage:
  python 06_run_all.py \
      --dataset_root PATH \
      --original_model PATH \
      --eval_root PATH

  --dataset_root   : Path to the original YOLO dataset (images/ + labels/ inside)
  --original_model : Path to best.pt (the pre-trained original RGB model)
  --eval_root      : Path to this robustness_eval/ folder

Optional flags:
  --skip_prepare   : Skip dataset preparation (steps 1-3) if datasets already exist
  --skip_training  : Skip model training (steps 4-5) if models already exist
  --skip_eval      : Skip evaluation (steps 6-8) if CSVs already exist
  --eval_split     : Dataset split to evaluate on (default: test)
  --force          : Force-overwrite existing datasets

Example (full pipeline):
  python robustness_eval/06_run_all.py \
      --dataset_root   "C:/Users/monst/datasets/final_11class_split" \
      --original_model "C:/Users/monst/OneDrive/Documents/Naiem/portable alifs cap/VehicleDetector/weights/best.pt" \
      --eval_root      "C:/Users/monst/OneDrive/Documents/Naiem/portable alifs cap/VehicleDetector/robustness_eval"

Example (skip training, only re-evaluate and compare):
  python robustness_eval/06_run_all.py \
      --dataset_root   "C:/Users/monst/datasets/final_11class_split" \
      --original_model "C:/Users/monst/.../weights/best.pt" \
      --eval_root      "C:/Users/monst/.../robustness_eval" \
      --skip_prepare --skip_training
"""

import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import (
    VARIANT_RGB,
    VARIANT_GRAYSCALE,
    VARIANT_LOWRES,
    VARIANT_WEIGHT_NAMES,
    ensure_dirs,
    get_metrics_dir,
    get_weights_dir,
    get_yaml_path,
)
from generate_yamls import generate_all_yamls


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

def run_step(step_name: str, args: list[str]) -> int:
    """
    Runs a Python script as a subprocess using the current interpreter.
    Returns the process return code (0 = success).
    """
    cmd = [sys.executable] + args
    print(f"\n{'='*60}")
    print(f"  STEP: {step_name}")
    print(f"  CMD : {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    elapsed = time.time() - start

    status = "PASSED" if result.returncode == 0 else "FAILED"
    print(f"\n  [{status}] {step_name}  ({elapsed:.1f}s)\n")
    return result.returncode


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    dataset_root:    str,
    original_model:  str,
    eval_root:       str,
    skip_prepare:    bool,
    skip_training:   bool,
    skip_eval:       bool,
    eval_split:      str,
    force:           bool,
) -> None:

    dataset_root   = os.path.abspath(dataset_root)
    original_model = os.path.abspath(original_model)
    eval_root      = os.path.abspath(eval_root)

    weights_dir = get_weights_dir(eval_root)
    metrics_dir = get_metrics_dir(eval_root)

    print(f"\n{'#'*60}")
    print(f"  ROBUSTNESS EVALUATION PIPELINE")
    print(f"{'#'*60}")
    print(f"  Dataset root   : {dataset_root}")
    print(f"  Original model : {original_model}")
    print(f"  Eval root      : {eval_root}")
    print(f"  Eval split     : {eval_split}")
    print(f"  Skip prepare   : {skip_prepare}")
    print(f"  Skip training  : {skip_training}")
    print(f"  Skip eval      : {skip_eval}")
    print(f"{'#'*60}\n")

    errors = []

    # ------------------------------------------------------------------ #
    # STEP 1: Generate YAML files                                         #
    # ------------------------------------------------------------------ #
    if not skip_prepare:
        print("\n[STEP 1] Generating data YAML files …")
        generate_all_yamls(dataset_root, eval_root)

    # ------------------------------------------------------------------ #
    # STEP 2: Grayscale dataset preparation                               #
    # ------------------------------------------------------------------ #
    if not skip_prepare:
        force_flag = ["--force"] if force else []
        rc = run_step(
            "Grayscale Dataset Preparation",
            [
                os.path.join(SCRIPT_DIR, "01_prepare_grayscale.py"),
                "--dataset_root", dataset_root,
                "--eval_root",    eval_root,
            ] + force_flag,
        )
        if rc != 0:
            errors.append("01_prepare_grayscale.py failed")

    # ------------------------------------------------------------------ #
    # STEP 3: Low-resolution dataset preparation                          #
    # ------------------------------------------------------------------ #
    if not skip_prepare:
        force_flag = ["--force"] if force else []
        rc = run_step(
            "Low-Resolution Dataset Preparation",
            [
                os.path.join(SCRIPT_DIR, "02_prepare_lowres.py"),
                "--dataset_root", dataset_root,
                "--eval_root",    eval_root,
            ] + force_flag,
        )
        if rc != 0:
            errors.append("02_prepare_lowres.py failed")

    # ------------------------------------------------------------------ #
    # STEP 4: Train grayscale model                                        #
    # ------------------------------------------------------------------ #
    if not skip_training:
        gs_weight = os.path.join(weights_dir, VARIANT_WEIGHT_NAMES[VARIANT_GRAYSCALE])
        if os.path.isfile(gs_weight) and not force:
            print(f"\n[SKIP] Grayscale model already exists: {gs_weight}")
            print("       Use --force to retrain.")
        else:
            rc = run_step(
                "Train YOLOv8m — Grayscale",
                [
                    os.path.join(SCRIPT_DIR, "03_train.py"),
                    "--variant",      VARIANT_GRAYSCALE,
                    "--dataset_root", dataset_root,
                    "--eval_root",    eval_root,
                ],
            )
            if rc != 0:
                errors.append("03_train.py --variant grayscale failed")

    # ------------------------------------------------------------------ #
    # STEP 5: Train low-res model                                          #
    # ------------------------------------------------------------------ #
    if not skip_training:
        lr_weight = os.path.join(weights_dir, VARIANT_WEIGHT_NAMES[VARIANT_LOWRES])
        if os.path.isfile(lr_weight) and not force:
            print(f"\n[SKIP] Low-res model already exists: {lr_weight}")
            print("       Use --force to retrain.")
        else:
            rc = run_step(
                "Train YOLOv8m — Low-Resolution",
                [
                    os.path.join(SCRIPT_DIR, "03_train.py"),
                    "--variant",      VARIANT_LOWRES,
                    "--dataset_root", dataset_root,
                    "--eval_root",    eval_root,
                ],
            )
            if rc != 0:
                errors.append("03_train.py --variant lowres failed")

    # ------------------------------------------------------------------ #
    # STEP 6: Evaluate original RGB model                                  #
    # ------------------------------------------------------------------ #
    if not skip_eval:
        rgb_csv = os.path.join(metrics_dir, "metrics_rgb.json")
        if os.path.isfile(rgb_csv) and not force:
            print(f"\n[SKIP] RGB evaluation metrics already exist: {rgb_csv}")
        else:
            rc = run_step(
                "Evaluate — Original RGB",
                [
                    os.path.join(SCRIPT_DIR, "04_evaluate.py"),
                    "--model",    original_model,
                    "--data",     get_yaml_path(eval_root, VARIANT_RGB),
                    "--label",    VARIANT_RGB,
                    "--eval_root", eval_root,
                    "--split",    eval_split,
                ],
            )
            if rc != 0:
                errors.append("04_evaluate.py --label rgb failed")

    # ------------------------------------------------------------------ #
    # STEP 7: Evaluate grayscale model                                     #
    # ------------------------------------------------------------------ #
    if not skip_eval:
        gs_csv = os.path.join(metrics_dir, "metrics_grayscale.json")
        gs_model = os.path.join(weights_dir, VARIANT_WEIGHT_NAMES[VARIANT_GRAYSCALE])
        if os.path.isfile(gs_csv) and not force:
            print(f"\n[SKIP] Grayscale evaluation metrics already exist: {gs_csv}")
        elif not os.path.isfile(gs_model):
            print(f"\n[WARN] Grayscale model not found: {gs_model}")
            print("       Run training first (or use --skip_eval to skip this step).")
            errors.append("Grayscale model missing for evaluation")
        else:
            rc = run_step(
                "Evaluate — Grayscale",
                [
                    os.path.join(SCRIPT_DIR, "04_evaluate.py"),
                    "--model",    gs_model,
                    "--data",     get_yaml_path(eval_root, VARIANT_GRAYSCALE),
                    "--label",    VARIANT_GRAYSCALE,
                    "--eval_root", eval_root,
                    "--split",    eval_split,
                ],
            )
            if rc != 0:
                errors.append("04_evaluate.py --label grayscale failed")

    # ------------------------------------------------------------------ #
    # STEP 8: Evaluate low-res model                                       #
    # ------------------------------------------------------------------ #
    if not skip_eval:
        lr_csv   = os.path.join(metrics_dir, "metrics_lowres.json")
        lr_model = os.path.join(weights_dir, VARIANT_WEIGHT_NAMES[VARIANT_LOWRES])
        if os.path.isfile(lr_csv) and not force:
            print(f"\n[SKIP] Low-res evaluation metrics already exist: {lr_csv}")
        elif not os.path.isfile(lr_model):
            print(f"\n[WARN] Low-res model not found: {lr_model}")
            print("       Run training first (or use --skip_eval to skip this step).")
            errors.append("Low-res model missing for evaluation")
        else:
            rc = run_step(
                "Evaluate — Low-Resolution",
                [
                    os.path.join(SCRIPT_DIR, "04_evaluate.py"),
                    "--model",    lr_model,
                    "--data",     get_yaml_path(eval_root, VARIANT_LOWRES),
                    "--label",    VARIANT_LOWRES,
                    "--eval_root", eval_root,
                    "--split",    eval_split,
                ],
            )
            if rc != 0:
                errors.append("04_evaluate.py --label lowres failed")

    # ------------------------------------------------------------------ #
    # STEP 9: Compare and report                                           #
    # ------------------------------------------------------------------ #
    rc = run_step(
        "Compare and Generate Reports",
        [
            os.path.join(SCRIPT_DIR, "05_compare.py"),
            "--eval_root", eval_root,
        ],
    )
    if rc != 0:
        errors.append("05_compare.py failed")

    # ------------------------------------------------------------------ #
    # Final summary                                                        #
    # ------------------------------------------------------------------ #
    print(f"\n{'#'*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'#'*60}")

    if errors:
        print(f"\n  [WARN] {len(errors)} step(s) encountered errors:")
        for e in errors:
            print(f"    ✗ {e}")
    else:
        print("\n  All steps completed successfully.")

    print(f"\n  Results directory: {eval_root}/results/")
    print(f"    Weights  : results/weights/")
    print(f"    Metrics  : results/metrics/")
    print(f"    Charts   : results/plots/")
    print(f"    Training : results/runs/")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete robustness evaluation pipeline."
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the original YOLO dataset root (contains images/ and labels/).",
    )
    parser.add_argument(
        "--original_model",
        required=True,
        help="Path to the pre-trained original RGB model (best.pt).",
    )
    parser.add_argument(
        "--eval_root",
        required=True,
        help="Path to this robustness_eval/ folder.",
    )
    parser.add_argument(
        "--eval_split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split used for evaluation (default: test).",
    )
    parser.add_argument(
        "--skip_prepare",
        action="store_true",
        help="Skip dataset preparation steps (assumes datasets already exist).",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip model training steps (assumes trained models already exist).",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip model evaluation steps (assumes metrics CSVs already exist).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force-overwrite existing datasets, models, and metrics.",
    )
    args = parser.parse_args()

    # ---- Validate ----------------------------------------------------------
    if not os.path.isdir(args.dataset_root):
        print(f"[ERROR] dataset_root does not exist: {args.dataset_root}")
        sys.exit(1)
    if not os.path.isfile(args.original_model):
        print(f"[ERROR] original_model not found: {args.original_model}")
        sys.exit(1)
    if not os.path.isdir(args.eval_root):
        print(f"[ERROR] eval_root does not exist: {args.eval_root}")
        sys.exit(1)

    run_pipeline(
        dataset_root   = args.dataset_root,
        original_model = args.original_model,
        eval_root      = args.eval_root,
        skip_prepare   = args.skip_prepare,
        skip_training  = args.skip_training,
        skip_eval      = args.skip_eval,
        eval_split     = args.eval_split,
        force          = args.force,
    )


if __name__ == "__main__":
    main()
