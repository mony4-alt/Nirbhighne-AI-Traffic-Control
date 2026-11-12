"""
evaluator.py — Model Evaluation Engine
=======================================
Runs Ultralytics YOLOv8 validation (model.val()) and saves all results.

This module is completely independent of detector.py and does not affect
any existing detection or video-processing functionality.

Public API
----------
run_evaluation(model_path, yaml_path, output_dir, device, progress_cb)
    Runs validation and returns an EvaluationResult dataclass.

EvaluationResult
    Flat struct holding overall + per-class metrics and paths to saved files.
"""

import json
import os
import shutil
import time
import cv2
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Callable

import config
from utils import logger, fix_pyinstaller_streams
from evaluation_dataset import prepare_validation_dataset

fix_pyinstaller_streams()


# ─────────────────────────────────────────────────────────────────────────────
# Result data-class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """All metrics and file paths produced by a single evaluation run."""

    # Overall metrics (0.0 if unavailable)
    precision:   float = 0.0
    recall:      float = 0.0
    map50:       float = 0.0
    map5095:     float = 0.0
    f1:          float = 0.0

    # Per-class metrics: {class_name: {"p": float, "r": float, "ap50": float, "ap": float}}
    per_class: dict = field(default_factory=dict)

    # Paths of saved artifacts
    metrics_txt:  str = ""
    metrics_json: str = ""
    plots_dir:    str = ""  # Folder that Ultralytics wrote plots into

    # Timing
    elapsed_sec: float = 0.0

    # Raw error message (empty = success)
    error: str = ""

    @property
    def success(self) -> bool:
        return not self.error


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(value) -> float:
    """Converts any numeric-like value to float; returns 0.0 on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_curve_values(raw_values: str) -> list[float]:
    """Parse comma/space-separated curve values and validate their range."""
    tokens = [token for token in raw_values.replace(";", ",").split(",")]
    values = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"Invalid curve value: {token!r}") from exc
        if not 0.0 <= value <= 1.0:
            raise ValueError("Curve values must be between 0 and 1.")
        values.append(value)

    if len(values) < 2:
        raise ValueError("Enter at least two curve values.")
    return values


def _parse_confusion_matrix(raw_matrix: str) -> list[list[float]]:
    """Parse a semicolon/newline-separated numeric matrix."""
    rows = []
    for raw_row in raw_matrix.replace(";", "\n").splitlines():
        raw_row = raw_row.strip()
        if not raw_row:
            continue
        row = []
        for token in raw_row.replace(",", " ").split():
            try:
                value = float(token)
            except ValueError as exc:
                raise ValueError(f"Invalid matrix value: {token!r}") from exc
            if value < 0:
                raise ValueError("Matrix values cannot be negative.")
            row.append(value)
        if not row:
            raise ValueError("Every matrix row must contain a value.")
        rows.append(row)

    if not rows:
        raise ValueError("Enter at least one matrix row.")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("All matrix rows must contain the same number of values.")
    if len(rows) != width:
        raise ValueError("The confusion matrix must be square.")
    return rows


def _prepare_chart_labels(class_labels: str, matrix_size: int) -> list[str]:
    """Normalise labels for confusion-matrix plots."""
    labels = [label.strip() for label in class_labels.split(",") if label.strip()]
    if labels and len(labels) != matrix_size:
        raise ValueError(
            f"Enter exactly {matrix_size} class labels, separated by commas."
        )
    if not labels:
        labels = [f"Class {index}" for index in range(1, matrix_size + 1)]
    return labels


def generate_custom_evaluation_charts(
    curve_values: str,
    matrix_values: str,
    output_dir: str,
    class_labels: str = "",
) -> dict[str, str]:
    """Create PNG curve and confusion-matrix charts from user-entered data.

    ``curve_values`` is a comma-separated list of values from 0 to 1.
    ``matrix_values`` uses semicolons or newlines between rows, for example
    ``80,10,5; 8,85,7; 4,6,90``.
    """
    curve = _parse_curve_values(curve_values)
    matrix = _parse_confusion_matrix(matrix_values)
    labels = _prepare_chart_labels(class_labels, len(matrix))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Chart generation requires matplotlib. Install the project requirements."
        ) from exc

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    curve_path = os.path.join(output_dir, "custom_metric_curve.png")
    x_values = np.linspace(0.0, 1.0, len(curve))
    curve_figure, curve_axis = plt.subplots(figsize=(8, 5), dpi=150)
    curve_axis.plot(
        x_values,
        curve,
        marker="o",
        linewidth=2.2,
        color="#22c55e",
    )
    curve_axis.set_title("Custom Evaluation Curve")
    curve_axis.set_xlabel("Evaluation point")
    curve_axis.set_ylabel("Metric value")
    curve_axis.set_ylim(0.0, 1.0)
    curve_axis.grid(True, alpha=0.25)
    curve_figure.tight_layout()
    curve_figure.savefig(curve_path, format="png", bbox_inches="tight")
    plt.close(curve_figure)

    matrix_path = os.path.join(output_dir, "custom_confusion_matrix.png")
    matrix_array = np.asarray(matrix, dtype=float)
    matrix_figure, matrix_axis = plt.subplots(
        figsize=(max(5.5, len(matrix) * 1.2), max(4.8, len(matrix) * 1.1)),
        dpi=150,
    )
    image = matrix_axis.imshow(matrix_array, cmap="Blues")
    matrix_figure.colorbar(image, ax=matrix_axis, fraction=0.046, pad=0.04)
    matrix_axis.set(
        title="Custom Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    threshold = matrix_array.max() / 2.0 if matrix_array.size else 0.0
    for row_index in range(matrix_array.shape[0]):
        for column_index in range(matrix_array.shape[1]):
            value = matrix_array[row_index, column_index]
            text_color = "white" if value > threshold else "black"
            matrix_axis.text(
                column_index,
                row_index,
                f"{value:g}",
                ha="center",
                va="center",
                color=text_color,
            )
    matrix_figure.tight_layout()
    matrix_figure.savefig(matrix_path, format="png", bbox_inches="tight")
    plt.close(matrix_figure)

    return {"curve": curve_path, "matrix": matrix_path}


def _parse_yaml_data(yaml_path: str) -> dict:
    """Parses data.yaml to extract dataset config.

    Raises ValueError with a descriptive message if the file cannot be parsed.
    """
    data = {}
    parse_error = None
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception as exc:
        parse_error = exc
        # Fallback: manual line-by-line parse
        logger.warning(f"yaml.safe_load failed ({exc}), falling back to manual parse.")
        try:
            with open(yaml_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("val:"):
                        data["val"] = line.split(":", 1)[1].strip()
                    elif line.startswith("train:"):
                        data["train"] = line.split(":", 1)[1].strip()
                    elif line.startswith("path:"):
                        data["path"] = line.split(":", 1)[1].strip()
                    elif line.startswith("nc:"):
                        try:
                            data["nc"] = int(line.split(":", 1)[1].strip())
                        except ValueError:
                            pass
        except OSError as io_exc:
            raise ValueError(f"Cannot read YAML file: {io_exc}") from io_exc

    if not data.get("val"):
        raise ValueError(
            f"No 'val:' entry found in data.yaml.\n"
            f"File: {yaml_path}\n"
            "Make sure the file contains a 'val:' key pointing to validation images."
        )
    return data


def _resolve_val_images_dir(yaml_path: str, data: dict) -> str:
    """Resolves absolute path to validation images from data.yaml."""
    val = data.get("val", "")
    if not val:
        raise ValueError(f"No 'val' path found in data.yaml:\n{yaml_path}")

    path_root = data.get("path", "")
    if path_root:
        if not os.path.isabs(path_root):
            path_root = os.path.join(os.path.dirname(yaml_path), path_root)
        val_dir = val if os.path.isabs(val) else os.path.join(path_root, val)
    else:
        val_dir = val if os.path.isabs(val) else os.path.join(os.path.dirname(yaml_path), val)

    return os.path.abspath(val_dir)


def _find_val_labels_dir(val_images_dir: str) -> str:
    """Locates matching labels directory for validation images."""
    norm_path = val_images_dir.replace("\\", "/")
    parts = norm_path.split("/")
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        cand = os.path.normpath("/".join(parts))
        if os.path.isdir(cand):
            return cand

    parent = os.path.dirname(val_images_dir)
    cand = os.path.join(parent, "labels")
    if os.path.isdir(cand):
        return cand

    split = os.path.basename(val_images_dir)
    cand = os.path.join(os.path.dirname(parent), "labels", split)
    if os.path.isdir(cand):
        return cand

    return os.path.join(parent, "labels")


def _create_and_verify_temp_grayscale(
    yaml_path: str,
    temp_dir: str,
    progress_cb: Callable[[str], None] | None = None
) -> str:
    """
    Creates temporary 3-channel grayscale dataset in robustness_eval/temp/grayscale/.
    Verifies that images are actually grayscale before returning the temporary data.yaml path.
    """
    def _log(msg: str):
        logger.info(f"[Grayscale Prep] {msg}")
        if progress_cb:
            progress_cb(msg)

    _log("Reading validation YAML configuration…")
    data = _parse_yaml_data(yaml_path)
    val_images_dir = _resolve_val_images_dir(yaml_path, data)

    if not os.path.isdir(val_images_dir):
        raise ValueError(f"Validation images directory not found:\n{val_images_dir}")

    val_labels_dir = _find_val_labels_dir(val_images_dir)

    temp_images_dir = os.path.join(temp_dir, "images", "val")
    temp_labels_dir = os.path.join(temp_dir, "labels", "val")

    # Clean existing temporary directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    os.makedirs(temp_images_dir, exist_ok=True)
    os.makedirs(temp_labels_dir, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
    image_files = [f for f in os.listdir(val_images_dir) if os.path.splitext(f)[1].lower() in img_exts]

    if not image_files:
        raise ValueError(f"No validation images found in:\n{val_images_dir}")

    _log(f"Converting {len(image_files)} validation images to 3-channel grayscale…")

    for i, fname in enumerate(image_files, 1):
        src_img_path = os.path.join(val_images_dir, fname)
        dst_img_path = os.path.join(temp_images_dir, fname)

        img = cv2.imread(src_img_path)
        if img is None:
            continue

        # Convert to grayscale, then back to 3-channel BGR for YOLOv8
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(dst_img_path, gray_3ch)

        # Copy label file if available
        stem = os.path.splitext(fname)[0]
        label_fname = stem + ".txt"
        src_lbl_path = os.path.join(val_labels_dir, label_fname)
        if os.path.isfile(src_lbl_path):
            shutil.copy2(src_lbl_path, os.path.join(temp_labels_dir, label_fname))

        if i % 50 == 0 or i == len(image_files):
            _log(f"Grayscale conversion: {i}/{len(image_files)} images processed")

    # Requirement 9: Verify that images are actually grayscale before evaluation
    _log("Verifying images are actually grayscale before evaluation…")
    temp_img_files = [f for f in os.listdir(temp_images_dir) if os.path.splitext(f)[1].lower() in img_exts]
    if not temp_img_files:
        raise ValueError("Grayscale verification failed: No converted images exist in temporary dataset.")

    for fname in temp_img_files:
        img_path = os.path.join(temp_images_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Grayscale verification error: Could not read image:\n{img_path}")
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError(f"Grayscale verification error: Image {fname} does not have 3 channels (shape: {img.shape})")

        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if not (np.array_equal(b, g) and np.array_equal(g, r)):
            raise ValueError(f"Grayscale verification error: Image {fname} is not grayscale (channels B, G, R are not identical)")

    _log("Verification successful: Images confirmed as 3-channel grayscale [OK]")

    # Generate temporary YAML
    temp_yaml_path = os.path.join(temp_dir, "data_temp_grayscale.yaml")
    base_posix = os.path.abspath(temp_dir).replace("\\", "/")

    nc = data.get("nc", 11)
    names = data.get("names", [])

    yaml_content = (
        f"# Auto-generated temporary grayscale dataset YAML\n"
        f"path: {base_posix}\n"
        f"train: images/val\n"
        f"val: images/val\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )

    with open(temp_yaml_path, "w", encoding="utf-8") as fh:
        fh.write(yaml_content)

    return temp_yaml_path


def _extract_metrics(val_results, class_names: list) -> dict:
    """
    Pulls precision, recall, mAP@50, mAP@50-95, and per-class values out of
    an Ultralytics DetMetrics / Results object.
    """
    box = val_results.box  # ultralytics.utils.metrics.DetMetrics

    overall_p    = _safe_float(box.mp)
    overall_r    = _safe_float(box.mr)
    overall_ap50 = _safe_float(box.map50)
    overall_ap   = _safe_float(box.map)
    denom        = overall_p + overall_r
    overall_f1   = (2 * overall_p * overall_r / denom) if denom > 0 else 0.0

    per_class = {}
    nc = len(class_names)
    indices = list(box.ap_class_index) if hasattr(box, "ap_class_index") else []
    p_arr   = list(box.p)    if hasattr(box, "p")    else []
    r_arr   = list(box.r)    if hasattr(box, "r")    else []
    ap50_arr= list(box.ap50) if hasattr(box, "ap50") else []
    ap_arr  = list(box.ap)   if hasattr(box, "ap")   else []

    for i, cls_idx in enumerate(indices):
        if cls_idx < nc:
            name = class_names[cls_idx]
            per_class[name] = {
                "p":    round(_safe_float(p_arr[i])    if i < len(p_arr)    else 0.0, 4),
                "r":    round(_safe_float(r_arr[i])    if i < len(r_arr)    else 0.0, 4),
                "ap50": round(_safe_float(ap50_arr[i]) if i < len(ap50_arr) else 0.0, 4),
                "ap":   round(_safe_float(ap_arr[i])   if i < len(ap_arr)   else 0.0, 4),
            }

    return {
        "precision": round(overall_p,    4),
        "recall":    round(overall_r,    4),
        "map50":     round(overall_ap50, 4),
        "map5095":   round(overall_ap,   4),
        "f1":        round(overall_f1,   4),
        "per_class": per_class,
    }


def _save_metrics_txt(metrics: dict, output_dir: str, title: str = "Model Evaluation Report") -> str:
    """
    Writes a human-readable metrics.txt report.
    Returns the file path.
    """
    path = os.path.join(output_dir, "metrics.txt")
    lines = [
        "=" * 52,
        f"  YOLOv8  —  {title}",
        "=" * 52,
        "",
        f"  {'Precision':<20} {metrics['precision']:.4f}",
        f"  {'Recall':<20} {metrics['recall']:.4f}",
        f"  {'mAP@0.50':<20} {metrics['map50']:.4f}",
        f"  {'mAP@0.50:0.95':<20} {metrics['map5095']:.4f}",
        f"  {'F1-Score':<20} {metrics['f1']:.4f}",
        "",
        "-" * 52,
        "  Per-class  mAP@0.50",
        "-" * 52,
    ]
    for cls, vals in metrics["per_class"].items():
        lines.append(
            f"  {cls:<20} P={vals['p']:.4f}  R={vals['r']:.4f}  "
            f"AP50={vals['ap50']:.4f}  AP={vals['ap']:.4f}"
        )
    lines += ["", "=" * 52]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    return path


def _save_metrics_json(metrics: dict, output_dir: str) -> str:
    """
    Writes metrics.json. Returns the file path.
    """
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return path


def _copy_ultralytics_plots(val_run_dir: str, output_dir: str) -> str:
    """
    Copies generated plots to output_dir and output_dir/plots/.
    """
    plots_dst = os.path.join(output_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dst, exist_ok=True)

    if not val_run_dir or not os.path.isdir(val_run_dir):
        return plots_dst

    exts = {".png", ".jpg", ".jpeg"}
    for fname in os.listdir(val_run_dir):
        if os.path.splitext(fname)[1].lower() in exts:
            src = os.path.join(val_run_dir, fname)
            dst_root = os.path.join(output_dir, fname)
            dst_plots = os.path.join(plots_dst, fname)
            try:
                shutil.copy2(src, dst_root)
                shutil.copy2(src, dst_plots)
            except Exception:
                pass

    return plots_dst


# ─────────────────────────────────────────────────────────────────────────────
# Internal: model loader helper
# ─────────────────────────────────────────────────────────────────────────────

def _load_yolo_model(model_path: str, device: str):
    """Load a YOLO model using the chdir workaround, then move to the given device.

    Fixes BUG-06: model was loaded without specifying device, making the
    device argument passed to model.val() inconsistent on multi-GPU machines.
    """
    from ultralytics import YOLO
    old_cwd    = os.getcwd()
    model_dir  = os.path.dirname(os.path.abspath(model_path))
    model_name = os.path.basename(model_path)
    try:
        os.chdir(model_dir)
        model = YOLO(model_name)
    finally:
        os.chdir(old_cwd)
    if device == "cuda":
        model.to(device)
    return model


def _normalise_eval_device(device: str | None) -> str:
    """Use a device that Ultralytics' PyTorch validator can actually use."""
    requested = (device or "").lower()
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        logger.warning("Evaluator requested CUDA, but CUDA is unavailable; using CPU.")
    elif requested not in ("", "cpu"):
        logger.info("Evaluator device '%s' is not a PyTorch validator device; using CPU.", device)
    # The live detector may use OpenVINO/DirectML for iGPUs, but model.val()
    # here is PyTorch-only, so make the fallback explicit and predictable.
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Internal: absolute-path YAML writer
# ─────────────────────────────────────────────────────────────────────────────

def _write_absolute_yaml(
    yaml_path: str,
    data: dict,
    temp_yaml_path: str,
    override_val_dir: str | None = None,
) -> str:
    """
    Writes a new data.yaml with all paths fully resolved to absolute paths.

    Fixes BUG-07: data.yaml with relative paths like '../valid/images' is resolved
    from the model's directory after os.chdir(), not from the YAML's own directory.
    Writing a temp YAML with absolute paths bypasses the issue entirely.
    """
    yaml_dir  = os.path.dirname(yaml_path)
    path_root = data.get("path", "")
    if path_root and not os.path.isabs(path_root):
        path_root = os.path.abspath(os.path.join(yaml_dir, path_root))

    def _abs(rel_path: str) -> str:
        if not rel_path:
            return ""
        if os.path.isabs(rel_path):
            return os.path.normpath(rel_path)
        if path_root:
            return os.path.normpath(os.path.join(path_root, rel_path))
        return os.path.normpath(os.path.join(yaml_dir, rel_path))

    abs_val   = override_val_dir if override_val_dir else _abs(data.get("val", ""))
    abs_train = override_val_dir if override_val_dir else _abs(data.get("train", abs_val))
    abs_test  = _abs(data.get("test", ""))
    nc    = data.get("nc", 1)
    names = data.get("names", [])

    fwd = lambda p: p.replace("\\", "/")  # noqa: E731
    lines = [
        "# Auto-generated absolute-path YAML — do not edit manually",
        f"path: {fwd(os.path.dirname(abs_val))}",
        f"train: {fwd(abs_train)}",
        f"val:   {fwd(abs_val)}",
    ]
    if abs_test:
        lines.append(f"test:  {fwd(abs_test)}")
    lines += [f"nc: {nc}", f"names: {names}"]

    os.makedirs(os.path.dirname(temp_yaml_path) or ".", exist_ok=True)
    with open(temp_yaml_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    logger.info(f"Absolute-path YAML written -> {temp_yaml_path}")
    logger.info(f"  val   -> {abs_val}")
    logger.info(f"  train -> {abs_train}")
    return temp_yaml_path


def run_evaluation(
    model_path:   str,
    yaml_path:    str,
    output_dir:   str | None = None,
    device:       str | None = None,
    progress_cb:  Callable[[str], None] | None = None,
) -> EvaluationResult:
    """
    Runs YOLOv8 validation on a temporary grayscale dataset and saves results to *output_dir*.

    Parameters
    ----------
    model_path  : Absolute path to the .pt weights file.
    yaml_path   : Absolute path to a YOLO-format data.yaml file for original dataset.
    output_dir  : Directory where metrics and plots are saved (defaults to robustness_eval/results/grayscale/).
    device      : 'cuda' or 'cpu'. Auto-detected if None.
    progress_cb : Optional callback(message: str) called from the worker thread.

    Returns
    -------
    EvaluationResult — always returned (check .success / .error).
    """

    def _cb(msg: str):
        logger.info(f"[Evaluator] {msg}")
        if progress_cb:
            progress_cb(msg)

    result = EvaluationResult()
    t_start = time.time()
    fix_pyinstaller_streams()

    if output_dir is None or output_dir == config.EVAL_RESULTS_DIR:
        output_dir = config.ROBUSTNESS_RESULTS_GRAYSCALE_DIR
    output_dir = os.path.abspath(output_dir)

    # ── Pre-flight checks ──────────────────────────────────────────────────────
    if not os.path.isfile(model_path):
        result.error = f"Model file not found:\n{model_path}"
        return result

    if not os.path.isfile(yaml_path):
        result.error = f"data.yaml not found:\n{yaml_path}"
        return result

    os.makedirs(output_dir, exist_ok=True)

    # ── Prepare & Verify Temporary Grayscale Dataset ───────────────────────────
    try:
        prepared_dir = os.path.join(config.ROBUSTNESS_EVAL_DIR, "temp", "prepared_validation")
        prepared_yaml_path = prepare_validation_dataset(yaml_path, prepared_dir)
        temp_dir = config.ROBUSTNESS_TEMP_GRAYSCALE_DIR
        eval_yaml_path = _create_and_verify_temp_grayscale(prepared_yaml_path, temp_dir, progress_cb=_cb)
    except Exception as exc:
        logger.exception("Evaluator: temporary grayscale dataset creation/verification failed.")
        result.error = f"Grayscale dataset error:\n{exc}"
        return result

    # ── Device selection ──────────────────────────────────────────────────────
    device = _normalise_eval_device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    _cb(f"Device: {device.upper()}")

    # ── Load YOLO model ────────────────────────────────────────────────
    try:
        _cb(f"Loading model: {os.path.basename(model_path)}")
        model = _load_yolo_model(model_path, device)
        logger.info("Evaluator: model loaded.")
    except Exception as exc:
        logger.exception("Evaluator: failed to load model.")
        result.error = f"Failed to load model:\n{exc}"
        return result

    # ── Run validation ────────────────────────────────────────────────────────
    try:
        _cb("Running YOLOv8 validation on grayscale dataset…")

        val_project  = os.path.join(output_dir, "_ultralytics_run")
        val_run_name = "eval"

        val_results = model.val(
            data     = eval_yaml_path,
            device   = device,
            verbose  = False,
            plots    = True,
            project  = val_project,
            name     = val_run_name,
            exist_ok = True,
            save_json= False,
            # workers=0 prevents Windows multiprocessing deadlocks in frozen/GUI apps
            workers  = 0,
            # Conservative settings are reliable on 4 GB cards. This model/
            # runtime combination can silently report zero metrics with FP16.
            batch    = config.EVAL_BATCH_SIZE if device == "cuda" else 1,
            half     = config.EVAL_USE_HALF and device == "cuda",
            # Standard eval imgsz; larger wastes time without meaningful accuracy gain
            imgsz    = 640,
        )

        _cb("Validation finished. Extracting metrics…")

    except Exception as exc:
        logger.exception("Evaluator: validation failed.")
        result.error = f"Validation error:\n{exc}"
        return result

    # ── Extract metrics ───────────────────────────────────────────────────────
    try:
        class_names = list(model.names.values()) if hasattr(model, "names") else []
        metrics     = _extract_metrics(val_results, class_names)
    except Exception as exc:
        logger.exception("Evaluator: metrics extraction failed.")
        result.error = f"Could not extract metrics:\n{exc}"
        return result

    # ── Save artifacts ────────────────────────────────────────────────────────
    try:
        txt_path  = _save_metrics_txt(metrics, output_dir, title="Grayscale Robustness Evaluation")
        json_path = _save_metrics_json(metrics, output_dir)

        # Copy Ultralytics-generated plots (PR curve, confusion matrix, …)
        val_run_dir = os.path.join(val_project, val_run_name)
        plots_dir   = _copy_ultralytics_plots(val_run_dir, output_dir)

        _cb("Results saved to robustness_eval/results/grayscale/")
        logger.info(f"Evaluator: results written to {output_dir}")
    except Exception as exc:
        logger.exception("Evaluator: failed to save results.")
        result.error = f"Could not save results:\n{exc}"
        return result

    # ── Build result ──────────────────────────────────────────────────────────
    result.precision   = metrics["precision"]
    result.recall      = metrics["recall"]
    result.map50       = metrics["map50"]
    result.map5095     = metrics["map5095"]
    result.f1          = metrics["f1"]
    result.per_class   = metrics["per_class"]
    result.metrics_txt = txt_path
    result.metrics_json= json_path
    result.plots_dir   = plots_dir
    result.elapsed_sec = time.time() - t_start

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API  (standard evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def run_standard_evaluation(
    model_path:   str,
    yaml_path:    str,
    output_dir:   str | None = None,
    device:       str | None = None,
    progress_cb:  Callable[[str], None] | None = None,
) -> EvaluationResult:
    """
    Runs YOLOv8 validation directly on the original dataset and saves results.

    This is the standard (non-robustness) evaluation path called by the GUI's
    'Evaluate Model' button.  Writes a temp YAML with absolute paths to
    work around Ultralytics' CWD-relative path resolution (BUG-07).

    Parameters
    ----------
    model_path  : Absolute path to the .pt weights file.
    yaml_path   : Absolute path to a YOLO-format data.yaml file.
    output_dir  : Directory where metrics and plots are saved
                  (defaults to evaluation_results/).
    device      : 'cuda' or 'cpu'. Auto-detected if None.
    progress_cb : Optional callback(message: str) called from the worker thread.

    Returns
    -------
    EvaluationResult — always returned (check .success / .error).
    """

    def _cb(msg: str):
        logger.info(f"[Evaluator] {msg}")
        if progress_cb:
            progress_cb(msg)

    result  = EvaluationResult()
    t_start = time.time()

    if output_dir is None:
        output_dir = config.EVAL_RESULTS_DIR
    output_dir = os.path.abspath(output_dir)

    # ── Pre-flight checks ──────────────────────────────────────────────────────
    if not os.path.isfile(model_path):
        result.error = f"Model file not found:\n{model_path}"
        return result

    if not os.path.isfile(yaml_path):
        result.error = f"data.yaml not found:\n{yaml_path}"
        return result

    os.makedirs(output_dir, exist_ok=True)

    # ── Parse YAML and write absolute-path version (fixes BUG-07) ─────────────
    try:
        _cb("Parsing data.yaml…")
        data = _parse_yaml_data(yaml_path)
    except Exception as exc:
        logger.exception("Evaluator: failed to parse data.yaml.")
        result.error = f"data.yaml parse error:\n{exc}"
        return result

    try:
        prepared_dir = os.path.join(output_dir, "_prepared_validation")
        prepared_yaml_path = prepare_validation_dataset(yaml_path, prepared_dir)
        prepared_data = _parse_yaml_data(prepared_yaml_path)
        temp_yaml_path = os.path.join(output_dir, "_temp_eval_data.yaml")
        _write_absolute_yaml(prepared_yaml_path, prepared_data, temp_yaml_path)
        _cb("Resolved dataset paths to absolute form.")
    except Exception as exc:
        logger.exception("Evaluator: failed to write absolute-path YAML.")
        result.error = f"YAML path resolution error:\n{exc}"
        return result

    # ── Device selection ──────────────────────────────────────────────────────
    device = _normalise_eval_device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _cb(f"Device: {device.upper()}")

    # ── Load YOLO model ───────────────────────────────────────────────────────
    try:
        _cb(f"Loading model: {os.path.basename(model_path)}")
        model = _load_yolo_model(model_path, device)
        logger.info("Evaluator: model loaded.")
    except Exception as exc:
        logger.exception("Evaluator: failed to load model.")
        result.error = f"Failed to load model:\n{exc}"
        return result

    # ── Run validation ────────────────────────────────────────────────────────
    try:
        _cb("Running YOLOv8 validation on original dataset…")

        val_project  = os.path.join(output_dir, "_ultralytics_run")
        val_run_name = "eval"

        val_results = model.val(
            data     = temp_yaml_path,
            device   = device,
            verbose  = False,
            plots    = True,
            project  = val_project,
            name     = val_run_name,
            exist_ok = True,
            save_json= False,
            # workers=0 prevents Windows multiprocessing deadlocks in frozen/GUI apps
            workers  = 0,
            # Conservative settings are reliable on 4 GB cards. This model/
            # runtime combination can silently report zero metrics with FP16.
            batch    = config.EVAL_BATCH_SIZE if device == "cuda" else 1,
            half     = config.EVAL_USE_HALF and device == "cuda",
            # Standard eval imgsz
            imgsz    = 640,
        )

        _cb("Validation finished. Extracting metrics…")

    except Exception as exc:
        logger.exception("Evaluator: validation failed.")
        result.error = f"Validation error:\n{exc}"
        return result

    # ── Extract metrics ───────────────────────────────────────────────────────
    try:
        class_names = list(model.names.values()) if hasattr(model, "names") else []
        metrics     = _extract_metrics(val_results, class_names)
    except Exception as exc:
        logger.exception("Evaluator: metrics extraction failed.")
        result.error = f"Could not extract metrics:\n{exc}"
        return result

    # ── Save artifacts ────────────────────────────────────────────────────────
    try:
        txt_path  = _save_metrics_txt(metrics, output_dir, title="Model Evaluation Report")
        json_path = _save_metrics_json(metrics, output_dir)

        val_run_dir = os.path.join(val_project, val_run_name)
        plots_dir   = _copy_ultralytics_plots(val_run_dir, output_dir)

        _cb(f"Results saved to {output_dir}")
        logger.info(f"Evaluator: standard evaluation results written to {output_dir}")
    except Exception as exc:
        logger.exception("Evaluator: failed to save results.")
        result.error = f"Could not save results:\n{exc}"
        return result

    # ── Build result ──────────────────────────────────────────────────────────
    result.precision   = metrics["precision"]
    result.recall      = metrics["recall"]
    result.map50       = metrics["map50"]
    result.map5095     = metrics["map5095"]
    result.f1          = metrics["f1"]
    result.per_class   = metrics["per_class"]
    result.metrics_txt = txt_path
    result.metrics_json= json_path
    result.plots_dir   = plots_dir
    result.elapsed_sec = time.time() - t_start

    return result
