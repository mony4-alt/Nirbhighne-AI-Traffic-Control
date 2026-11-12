"""
robustness_eval/config.py
=========================
Central configuration for the YOLOv8m robustness evaluation pipeline.

All paths are derived from a user-supplied DATASET_ROOT at runtime.
Do NOT hardcode any file paths here — only defaults and constants.
"""

import os
import torch

# ---------------------------------------------------------------------------
# Training hyperparameters
# These are kept identical across all three experiments (RGB, grayscale, lowres)
# to ensure fair comparison.
# ---------------------------------------------------------------------------
TRAIN_CONFIG = {
    "epochs":     100,
    "imgsz":      640,
    "batch":      16,
    "optimizer":  "auto",
    "patience":   20,          # Early stopping patience (epochs without improvement)
    "workers":    8,           # DataLoader workers
    "seed":       42,          # Reproducibility seed
    "verbose":    True,
    "exist_ok":   True,        # Allow re-running without clearing old runs
    "val":        True,        # Run validation at end of each epoch
    "save":       True,        # Save best.pt and last.pt checkpoints
    "plots":      True,        # Generate training plots
}

# ---------------------------------------------------------------------------
# Dataset variant names
# ---------------------------------------------------------------------------
VARIANT_RGB       = "rgb"
VARIANT_GRAYSCALE = "grayscale"
VARIANT_LOWRES    = "lowres"

VARIANT_DATASET_DIRS = {
    VARIANT_RGB:       None,          # Uses original dataset root directly
    VARIANT_GRAYSCALE: "dataset_grayscale",
    VARIANT_LOWRES:    "dataset_lowres",
}

VARIANT_WEIGHT_NAMES = {
    VARIANT_RGB:       "best.pt",
    VARIANT_GRAYSCALE: "best_grayscale.pt",
    VARIANT_LOWRES:    "best_lowres.pt",
}

VARIANT_YAML_NAMES = {
    VARIANT_RGB:       "data_rgb.yaml",
    VARIANT_GRAYSCALE: "data_grayscale.yaml",
    VARIANT_LOWRES:    "data_lowres.yaml",
}

# ---------------------------------------------------------------------------
# Class names (11 classes from original training — order must match label indices)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "Ambulance",
    "Auto_ricksaw",
    "Bike",
    "Bus",
    "Fire_truck",
    "Leguna",
    "Pickup",
    "Police_car",
    "Private_car",
    "Truck",
    "cng",
]

# Categorization for analysis (used in 05_compare.py)
SMALL_VEHICLES  = {"Bike", "cng", "Auto_ricksaw"}
MEDIUM_VEHICLES = {"Private_car", "Police_car", "Pickup", "Ambulance", "Leguna"}
LARGE_VEHICLES  = {"Bus", "Truck", "Fire_truck"}

# ---------------------------------------------------------------------------
# Low-resolution scale factor
# ---------------------------------------------------------------------------
LOWRES_SCALE = 0.75  # 75% of original dimensions (approx 25% resolution reduction)

# ---------------------------------------------------------------------------
# Supported image extensions
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def get_device() -> str:
    """Returns 'cuda' if a CUDA-capable GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Path helpers (all derived from a runtime-supplied dataset_root)
# ---------------------------------------------------------------------------
def get_results_dir(eval_root: str) -> str:
    """Returns the results/ directory inside robustness_eval/."""
    return os.path.join(eval_root, "results")


def get_dataset_dir(dataset_root: str, variant: str) -> str:
    """
    Returns the path to the dataset directory for a given variant.

    For 'rgb', this is the original dataset_root itself.
    For 'grayscale'/'lowres', it is a sibling folder next to dataset_root.
    """
    if variant == VARIANT_RGB:
        return dataset_root
    parent = os.path.dirname(dataset_root)
    return os.path.join(parent, VARIANT_DATASET_DIRS[variant])


def get_yaml_path(eval_root: str, variant: str) -> str:
    """Returns the path to the generated data YAML for a variant."""
    return os.path.join(get_results_dir(eval_root), "yamls", VARIANT_YAML_NAMES[variant])


def get_weights_dir(eval_root: str) -> str:
    return os.path.join(get_results_dir(eval_root), "weights")


def get_metrics_dir(eval_root: str) -> str:
    return os.path.join(get_results_dir(eval_root), "metrics")


def get_plots_dir(eval_root: str) -> str:
    return os.path.join(get_results_dir(eval_root), "plots")


def get_runs_dir(eval_root: str) -> str:
    return os.path.join(get_results_dir(eval_root), "runs")


def ensure_dirs(*paths: str) -> None:
    """Creates all given directories (and their parents) if they don't exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)
