"""
robustness_eval/04_evaluate.py
================================
Evaluates a YOLOv8 model on a dataset split and exports metrics to CSV.

What it does:
  - Runs model.val() on the specified dataset and split (default: test)
  - Extracts per-class and overall metrics:
      Precision, Recall, mAP@50, mAP@50-95, F1-score
  - Saves results to results/metrics/metrics_<label>.csv
  - Also exports the confusion matrix as a PNG

Usage (evaluate original RGB model on its original test set):
  python 04_evaluate.py \
      --model    PATH/TO/best.pt \
      --data     PATH/TO/data_rgb.yaml \
      --label    rgb \
      --eval_root PATH/TO/robustness_eval

Usage (evaluate grayscale model on grayscale test set):
  python 04_evaluate.py \
      --model    results/weights/best_grayscale.pt \
      --data     results/yamls/data_grayscale.yaml \
      --label    grayscale \
      --eval_root PATH/TO/robustness_eval

Arguments:
  --model     : Path to the .pt model weights file
  --data      : Path to the data YAML for the dataset to evaluate on
  --label     : Short name used for the output CSV (e.g. rgb, grayscale, lowres)
  --eval_root : Path to the robustness_eval/ folder
  --split     : Dataset split to evaluate on (default: test; options: train, val, test)
"""

import argparse
import os
import sys
import csv
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import (
    CLASS_NAMES,
    ensure_dirs,
    get_device,
    get_metrics_dir,
    get_plots_dir,
    get_runs_dir,
)

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics is not installed. Run: pip install ultralytics>=8.2.0")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_float(value) -> float:
    """Converts a value to float safely; returns 0.0 on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def extract_metrics(results, class_names: list) -> dict:
    """
    Extracts a flat metrics dictionary from an Ultralytics validation Results object.

    Returns a dict with keys:
      overall_precision, overall_recall, overall_map50, overall_map5095, overall_f1
      <class>_precision, <class>_recall, <class>_map50, <class>_map5095
    """
    metrics = {}

    # ---- Overall metrics ---------------------------------------------------
    box = results.box  # DetMetrics object

    # Ultralytics stores per-class arrays in box.ap_class_index, box.p, etc.
    # The mean() values give overall (micro-averaged) metrics.
    overall_p     = safe_float(box.mp)       # mean precision
    overall_r     = safe_float(box.mr)       # mean recall
    overall_map50 = safe_float(box.map50)    # mAP@0.50
    overall_map   = safe_float(box.map)      # mAP@0.50:0.95
    # F1 = harmonic mean of precision and recall
    overall_f1    = (
        2 * overall_p * overall_r / (overall_p + overall_r)
        if (overall_p + overall_r) > 0 else 0.0
    )

    metrics["overall_precision"]  = round(overall_p,     4)
    metrics["overall_recall"]     = round(overall_r,     4)
    metrics["overall_map50"]      = round(overall_map50, 4)
    metrics["overall_map5095"]    = round(overall_map,   4)
    metrics["overall_f1"]         = round(overall_f1,    4)

    # ---- Per-class metrics -------------------------------------------------
    # box.ap_class_index: list of class indices that were present in eval
    # box.p, box.r, box.ap50, box.ap: per-class arrays (same order)
    nc = len(class_names)
    per_class_p    = list(box.p)    if hasattr(box, "p")    else []
    per_class_r    = list(box.r)    if hasattr(box, "r")    else []
    per_class_ap50 = list(box.ap50) if hasattr(box, "ap50") else []
    per_class_ap   = list(box.ap)   if hasattr(box, "ap")   else []
    class_indices  = list(box.ap_class_index) if hasattr(box, "ap_class_index") else []

    for i, cls_idx in enumerate(class_indices):
        if cls_idx < nc:
            cls_name = class_names[cls_idx]
            p    = safe_float(per_class_p[i])    if i < len(per_class_p)    else 0.0
            r    = safe_float(per_class_r[i])    if i < len(per_class_r)    else 0.0
            ap50 = safe_float(per_class_ap50[i]) if i < len(per_class_ap50) else 0.0
            ap   = safe_float(per_class_ap[i])   if i < len(per_class_ap)   else 0.0

            metrics[f"{cls_name}_precision"]  = round(p,    4)
            metrics[f"{cls_name}_recall"]     = round(r,    4)
            metrics[f"{cls_name}_map50"]      = round(ap50, 4)
            metrics[f"{cls_name}_map5095"]    = round(ap,   4)

    return metrics


def save_metrics_csv(metrics: dict, label: str, metrics_dir: str) -> str:
    """
    Saves a metrics dict to a CSV file.
    Also saves a JSON copy for easy programmatic access.
    Returns the path to the CSV file.
    """
    ensure_dirs(metrics_dir)

    csv_path  = os.path.join(metrics_dir, f"metrics_{label}.csv")
    json_path = os.path.join(metrics_dir, f"metrics_{label}.json")

    # ---- CSV ---------------------------------------------------------------
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])

    # ---- JSON (for 05_compare.py) -----------------------------------------
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"[OK] Metrics saved → {csv_path}")
    return csv_path


def print_metrics_table(metrics: dict, label: str) -> None:
    """Prints a pretty console summary of the overall metrics."""
    print(f"\n{'='*60}")
    print(f"  Evaluation Results — {label.upper()}")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Value':>10}")
    print(f"  {'-'*30}")
    keys_of_interest = [
        ("Precision",  "overall_precision"),
        ("Recall",     "overall_recall"),
        ("mAP@50",     "overall_map50"),
        ("mAP@50-95",  "overall_map5095"),
        ("F1-Score",   "overall_f1"),
    ]
    for display, key in keys_of_interest:
        val = metrics.get(key, "N/A")
        print(f"  {display:<20} {val:>10}")
    print(f"{'='*60}")

    # Per-class summary
    print(f"\n  Per-class mAP@50:")
    print(f"  {'-'*40}")
    for cls in CLASS_NAMES:
        key = f"{cls}_map50"
        val = metrics.get(key, "N/A")
        print(f"    {cls:<18} {val}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLOv8 model on a dataset and export metrics to CSV."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the YOLOv8 .pt weights file.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the YOLO data YAML for the dataset to evaluate on.",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Short name for this evaluation (used in output filenames), e.g. 'rgb'.",
    )
    parser.add_argument(
        "--eval_root",
        required=True,
        help="Path to the robustness_eval/ folder.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: test).",
    )
    args = parser.parse_args()

    # ---- Validate inputs ---------------------------------------------------
    if not os.path.isfile(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        sys.exit(1)
    if not os.path.isfile(args.data):
        print(f"[ERROR] Data YAML not found: {args.data}")
        sys.exit(1)
    if not os.path.isdir(args.eval_root):
        print(f"[ERROR] eval_root does not exist: {args.eval_root}")
        sys.exit(1)

    eval_root   = os.path.abspath(args.eval_root)
    metrics_dir = get_metrics_dir(eval_root)
    plots_dir   = get_plots_dir(eval_root)
    runs_dir    = get_runs_dir(eval_root)
    device      = get_device()

    ensure_dirs(metrics_dir, plots_dir, runs_dir)

    print(f"\n{'='*60}")
    print(f"  Evaluating: {args.label.upper()}")
    print(f"{'='*60}")
    print(f"  Model  : {args.model}")
    print(f"  Data   : {args.data}")
    print(f"  Split  : {args.split}")
    print(f"  Device : {device.upper()}")
    print(f"{'='*60}\n")

    # ---- Load model and run validation ------------------------------------
    model = YOLO(args.model)

    val_results = model.val(
        data    = args.data,
        split   = args.split,
        device  = device,
        verbose = True,
        plots   = True,
        project = runs_dir,
        name    = f"eval_{args.label}_{args.split}",
        exist_ok= True,
        save_json = False,
    )

    # ---- Extract and save metrics -----------------------------------------
    metrics = extract_metrics(val_results, CLASS_NAMES)
    print_metrics_table(metrics, args.label)
    save_metrics_csv(metrics, args.label, metrics_dir)

    print(f"\n[DONE] Evaluation complete for: {args.label}")
    print(f"       Metrics directory: {metrics_dir}\n")


if __name__ == "__main__":
    main()
