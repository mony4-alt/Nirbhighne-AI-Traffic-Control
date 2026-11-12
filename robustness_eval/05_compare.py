"""
robustness_eval/05_compare.py
==============================
Loads evaluation metrics from all three experiments and produces:
  1. A final comparison table (console + CSV)
  2. Bar charts for Precision, Recall, mAP@50, mAP@50-95
  3. Per-class mAP@50 grouped bar chart
  4. A data-driven analysis report (text file)

This script ONLY reads pre-computed metrics from JSON files.
It does NOT run any model or invent any numbers.

Usage:
  python 05_compare.py --eval_root PATH/TO/robustness_eval

Options:
  --eval_root : Path to the robustness_eval/ folder
  --labels    : Space-separated list of labels to compare
                (default: rgb grayscale lowres)
"""

import argparse
import json
import os
import sys
import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import (
    CLASS_NAMES,
    SMALL_VEHICLES,
    MEDIUM_VEHICLES,
    LARGE_VEHICLES,
    ensure_dirs,
    get_metrics_dir,
    get_plots_dir,
)

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend (no display needed)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib or numpy not installed — skipping chart generation.")
    print("       To install: pip install matplotlib numpy")


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
PALETTE = {
    "rgb":       "#2196F3",   # Blue  — original
    "grayscale": "#9C27B0",   # Purple — grayscale
    "lowres":    "#FF5722",   # Deep orange — low-res
}

DEFAULT_LABELS = ["rgb", "grayscale", "lowres"]

DISPLAY_NAMES = {
    "rgb":       "Original RGB",
    "grayscale": "Grayscale",
    "lowres":    "Reduced Resolution",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics(metrics_dir: str, labels: list) -> dict:
    """
    Loads all metrics JSON files. Returns a dict keyed by label.
    Warns (but does not crash) if a file is missing.
    """
    data = {}
    for label in labels:
        path = os.path.join(metrics_dir, f"metrics_{label}.json")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                data[label] = json.load(fh)
            print(f"[OK] Loaded metrics for: {label}")
        else:
            print(f"[WARN] Metrics file not found: {path}")
            print(f"       Run 04_evaluate.py --label {label} first.")
    return data


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

OVERALL_KEYS = [
    ("Precision",  "overall_precision"),
    ("Recall",     "overall_recall"),
    ("mAP@50",     "overall_map50"),
    ("mAP@50-95",  "overall_map5095"),
    ("F1-Score",   "overall_f1"),
]


def print_comparison_table(all_metrics: dict, labels: list) -> None:
    """Prints a formatted comparison table to stdout."""
    col_w = 18
    header = f"{'Metric':<20}" + "".join(
        f"{DISPLAY_NAMES.get(l, l):>{col_w}}" for l in labels
    )
    sep = "-" * len(header)
    print(f"\n{'='*60}")
    print("  FINAL COMPARISON TABLE")
    print(f"{'='*60}")
    print(header)
    print(sep)

    for display, key in OVERALL_KEYS:
        row = f"{display:<20}"
        for label in labels:
            val = all_metrics.get(label, {}).get(key, "N/A")
            row += f"{str(val):>{col_w}}"
        print(row)

    print(sep)
    print()


def save_comparison_csv(all_metrics: dict, labels: list, metrics_dir: str) -> str:
    """Saves the comparison table as metrics_summary.csv."""
    csv_path = os.path.join(metrics_dir, "metrics_summary.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Metric"] + [DISPLAY_NAMES.get(l, l) for l in labels])

        for display, key in OVERALL_KEYS:
            row = [display]
            for label in labels:
                row.append(all_metrics.get(label, {}).get(key, ""))
            writer.writerow(row)

    print(f"[OK] Summary CSV → {csv_path}")
    return csv_path


def save_perclass_csv(all_metrics: dict, labels: list, metrics_dir: str) -> str:
    """Saves per-class mAP@50 comparison as metrics_perclass.csv."""
    csv_path = os.path.join(metrics_dir, "metrics_perclass.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Class"] + [DISPLAY_NAMES.get(l, l) for l in labels])

        for cls in CLASS_NAMES:
            row = [cls]
            for label in labels:
                key = f"{cls}_map50"
                row.append(all_metrics.get(label, {}).get(key, ""))
            writer.writerow(row)

    print(f"[OK] Per-class CSV → {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _bar_chart(
    labels: list,
    values: list,
    title: str,
    ylabel: str,
    save_path: str,
    ylim: tuple = (0, 1),
) -> None:
    """Saves a single grouped bar chart comparing metrics across experiments."""
    fig, ax = plt.subplots(figsize=(8, 5))
    display = [DISPLAY_NAMES.get(l, l) for l in labels]
    colors  = [PALETTE.get(l, "#607D8B") for l in labels]
    bars    = ax.bar(display, values, color=colors, width=0.5, edgecolor="white", linewidth=0.8)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(ylim)
    ax.set_xlabel("Experiment", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    # Value labels on bars
    for bar, val in zip(bars, values):
        if isinstance(val, float):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Chart → {save_path}")


def plot_overall_charts(all_metrics: dict, labels: list, plots_dir: str) -> None:
    """Generates one bar chart per overall metric."""
    if not HAS_MPL:
        return

    chart_defs = [
        ("overall_precision",  "Precision Comparison",       "Precision",  "chart_precision.png"),
        ("overall_recall",     "Recall Comparison",          "Recall",     "chart_recall.png"),
        ("overall_map50",      "mAP@50 Comparison",          "mAP@50",     "chart_map50.png"),
        ("overall_map5095",    "mAP@50-95 Comparison",       "mAP@50-95",  "chart_map5095.png"),
        ("overall_f1",         "F1-Score Comparison",        "F1-Score",   "chart_f1.png"),
    ]

    for key, title, ylabel, filename in chart_defs:
        values = []
        valid_labels = []
        for label in labels:
            val = all_metrics.get(label, {}).get(key)
            if val is not None:
                values.append(float(val))
                valid_labels.append(label)

        if not values:
            print(f"  [SKIP] No data for chart: {title}")
            continue

        save_path = os.path.join(plots_dir, filename)
        _bar_chart(
            labels    = valid_labels,
            values    = values,
            title     = title,
            ylabel    = ylabel,
            save_path = save_path,
        )


def plot_perclass_chart(all_metrics: dict, labels: list, plots_dir: str) -> None:
    """Generates a grouped bar chart of per-class mAP@50 for each experiment."""
    if not HAS_MPL:
        return

    nc      = len(CLASS_NAMES)
    x       = np.arange(nc)
    n       = len(labels)
    width   = 0.8 / n

    fig, ax = plt.subplots(figsize=(max(14, nc * 1.5), 6))

    for i, label in enumerate(labels):
        values = []
        for cls in CLASS_NAMES:
            val = all_metrics.get(label, {}).get(f"{cls}_map50", 0.0)
            values.append(float(val) if val != "" else 0.0)

        offset = (i - n / 2 + 0.5) * width
        bars   = ax.bar(
            x + offset, values, width,
            label  = DISPLAY_NAMES.get(label, label),
            color  = PALETTE.get(label, "#607D8B"),
            edgecolor = "white", linewidth = 0.6,
        )

    ax.set_title("Per-Class mAP@50 Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("mAP@50", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(plots_dir, "chart_perclass_map50.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Chart → {save_path}")


def plot_radar_chart(all_metrics: dict, labels: list, plots_dir: str) -> None:
    """Generates a radar chart overlaying all experiments on 4 metrics."""
    if not HAS_MPL:
        return

    categories = ["Precision", "Recall", "mAP@50", "mAP@50-95"]
    keys       = ["overall_precision", "overall_recall", "overall_map50", "overall_map5095"]
    N          = len(categories)
    angles     = [n / float(N) * 2 * np.pi for n in range(N)]
    angles    += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)

    for label in labels:
        values = [float(all_metrics.get(label, {}).get(k, 0.0)) for k in keys]
        values += values[:1]
        color  = PALETTE.get(label, "#607D8B")
        ax.plot(angles, values, linewidth=2, linestyle="solid", color=color,
                label=DISPLAY_NAMES.get(label, label))
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)
    ax.set_title("Metric Radar Chart", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    save_path = os.path.join(plots_dir, "chart_radar.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Chart → {save_path}")


# ---------------------------------------------------------------------------
# Analysis report
# ---------------------------------------------------------------------------

def compute_delta(base_val, new_val) -> float:
    """Returns the absolute delta between two values."""
    try:
        return round(float(new_val) - float(base_val), 4)
    except (TypeError, ValueError):
        return 0.0


def classify_change(delta: float, threshold: float = 0.02) -> str:
    """Returns a short descriptor based on delta magnitude."""
    if abs(delta) < threshold:
        return "minimal change"
    if delta > 0:
        return f"improvement (+{delta:.4f})"
    return f"degradation ({delta:.4f})"


def generate_analysis(all_metrics: dict, labels: list, plots_dir: str) -> str:
    """
    Generates a factual, data-driven analysis report.
    All conclusions are derived directly from the loaded metrics.
    Returns the report text.
    """
    rgb_metrics = all_metrics.get("rgb", {})
    lines = []

    lines.append("=" * 70)
    lines.append("  ROBUSTNESS EVALUATION — ANALYSIS REPORT")
    lines.append("  Generated automatically from actual evaluation metrics")
    lines.append("=" * 70)
    lines.append("")

    # ---- Section 1: Experiment 1 (Grayscale) ------------------------------
    lines.append("EXPERIMENT 1: GRAYSCALE DATASET EVALUATION")
    lines.append("-" * 50)

    if "grayscale" in all_metrics:
        gs = all_metrics["grayscale"]
        for display, key in OVERALL_KEYS:
            base = rgb_metrics.get(key, "N/A")
            new  = gs.get(key, "N/A")
            delta = compute_delta(base, new)
            desc = classify_change(delta) if base != "N/A" and new != "N/A" else "N/A"
            lines.append(f"  {display:<18}: RGB={base}  Grayscale={new}  → {desc}")

        lines.append("")
        lines.append("  Per-class mAP@50 (Grayscale vs RGB):")
        for cls in CLASS_NAMES:
            base = rgb_metrics.get(f"{cls}_map50", "N/A")
            new  = gs.get(f"{cls}_map50", "N/A")
            if base != "N/A" and new != "N/A":
                delta = compute_delta(base, new)
                desc  = classify_change(delta, threshold=0.03)
                lines.append(f"    {cls:<18}: RGB={base:.4f}  GS={new:.4f}  → {desc}")

        # Overall inference about color information
        gs_map50 = float(gs.get("overall_map50", 0))
        rgb_map50 = float(rgb_metrics.get("overall_map50", 0)) if rgb_metrics else 0
        delta_map = compute_delta(rgb_map50, gs_map50)
        lines.append("")
        lines.append("  Color Importance Assessment:")
        if abs(delta_map) < 0.02:
            lines.append(
                f"  The grayscale model achieved mAP@50={gs_map50:.4f} vs RGB mAP@50={rgb_map50:.4f} "
                f"(delta={delta_map:+.4f}). This negligible difference suggests the model relies "
                f"primarily on structural/shape features rather than color information."
            )
        elif delta_map < 0:
            lines.append(
                f"  The grayscale model achieved mAP@50={gs_map50:.4f} vs RGB mAP@50={rgb_map50:.4f} "
                f"(delta={delta_map:+.4f}). This drop indicates that color information does contribute "
                f"meaningfully to the original model's detection accuracy."
            )
        else:
            lines.append(
                f"  Surprisingly, the grayscale model achieved mAP@50={gs_map50:.4f} vs RGB "
                f"mAP@50={rgb_map50:.4f} (delta={delta_map:+.4f}). This suggests the model can "
                f"match or exceed performance even without color, possibly due to reduced overfitting."
            )
    else:
        lines.append("  [DATA UNAVAILABLE] Grayscale metrics not found.")
        lines.append("  Run 04_evaluate.py --label grayscale to generate them.")

    lines.append("")

    # ---- Section 2: Experiment 2 (Low-Res) --------------------------------
    lines.append("EXPERIMENT 2: REDUCED RESOLUTION DATASET EVALUATION")
    lines.append("-" * 50)

    if "lowres" in all_metrics:
        lr = all_metrics["lowres"]
        for display, key in OVERALL_KEYS:
            base = rgb_metrics.get(key, "N/A")
            new  = lr.get(key, "N/A")
            delta = compute_delta(base, new)
            desc = classify_change(delta) if base != "N/A" and new != "N/A" else "N/A"
            lines.append(f"  {display:<18}: RGB={base}  LowRes={new}  → {desc}")

        lines.append("")
        lines.append("  Vehicle size category analysis (mAP@50 delta vs RGB):")

        small_deltas  = []
        medium_deltas = []
        large_deltas  = []

        for cls in CLASS_NAMES:
            base = rgb_metrics.get(f"{cls}_map50", None)
            new  = lr.get(f"{cls}_map50", None)
            if base is not None and new is not None:
                delta = compute_delta(base, new)
                cat = (
                    "small"  if cls in SMALL_VEHICLES  else
                    "large"  if cls in LARGE_VEHICLES   else
                    "medium"
                )
                line = f"    {cls:<18}: RGB={float(base):.4f}  LR={float(new):.4f}  Δ={delta:+.4f}  [{cat}]"
                lines.append(line)
                if cat == "small":
                    small_deltas.append(delta)
                elif cat == "large":
                    large_deltas.append(delta)
                else:
                    medium_deltas.append(delta)

        lines.append("")
        lines.append("  Resolution Impact Assessment:")

        def avg(lst):
            return round(sum(lst) / len(lst), 4) if lst else 0.0

        avg_small  = avg(small_deltas)
        avg_large  = avg(large_deltas)
        avg_medium = avg(medium_deltas)

        lines.append(f"    Average mAP@50 delta — Small vehicles  : {avg_small:+.4f}")
        lines.append(f"    Average mAP@50 delta — Medium vehicles : {avg_medium:+.4f}")
        lines.append(f"    Average mAP@50 delta — Large vehicles  : {avg_large:+.4f}")
        lines.append("")

        if avg_small < avg_large:
            lines.append(
                f"  Small vehicles (Bike, cng, Auto_ricksaw) suffer greater accuracy loss "
                f"(avg Δ={avg_small:+.4f}) than large vehicles (avg Δ={avg_large:+.4f}) under "
                f"reduced resolution. This is expected, as small objects rely on fine-grained "
                f"spatial details that are lost in downscaled images."
            )
        elif avg_small > avg_large:
            lines.append(
                f"  Unexpectedly, large vehicles show greater mAP@50 degradation (avg Δ={avg_large:+.4f}) "
                f"compared to small vehicles (avg Δ={avg_small:+.4f}). This may indicate that large "
                f"vehicle classes were already well-represented and the reduced resolution specifically "
                f"affects their distinctive features."
            )
        else:
            lines.append(
                f"  Small and large vehicles show similar mAP@50 changes under reduced resolution, "
                f"suggesting resolution reduction affects all classes roughly equally."
            )

        lr_map50   = float(lr.get("overall_map50", 0))
        rgb_map50  = float(rgb_metrics.get("overall_map50", 0)) if rgb_metrics else 0
        delta_map2 = compute_delta(rgb_map50, lr_map50)
        lines.append("")
        lines.append(
            f"  Overall: Reduced resolution model achieved mAP@50={lr_map50:.4f} vs "
            f"RGB mAP@50={rgb_map50:.4f} (delta={delta_map2:+.4f})."
        )
    else:
        lines.append("  [DATA UNAVAILABLE] Low-res metrics not found.")
        lines.append("  Run 04_evaluate.py --label lowres to generate them.")

    lines.append("")
    lines.append("=" * 70)
    lines.append("  END OF REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


def save_analysis(report: str, plots_dir: str) -> str:
    """Saves the analysis report to a text file."""
    path = os.path.join(plots_dir, "analysis_report.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"[OK] Analysis report → {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare metrics from all robustness experiments and generate reports."
    )
    parser.add_argument(
        "--eval_root",
        required=True,
        help="Path to the robustness_eval/ folder.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=DEFAULT_LABELS,
        help="Labels to compare (default: rgb grayscale lowres).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.eval_root):
        print(f"[ERROR] eval_root does not exist: {args.eval_root}")
        sys.exit(1)

    eval_root   = os.path.abspath(args.eval_root)
    metrics_dir = get_metrics_dir(eval_root)
    plots_dir   = get_plots_dir(eval_root)
    ensure_dirs(plots_dir)

    # ---- Load metrics -------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Loading Metrics")
    print(f"{'='*60}")
    all_metrics = load_metrics(metrics_dir, args.labels)

    if not all_metrics:
        print("\n[ERROR] No metrics files found. Run 04_evaluate.py for each variant first.")
        sys.exit(1)

    available_labels = [l for l in args.labels if l in all_metrics]

    # ---- Comparison table --------------------------------------------------
    print_comparison_table(all_metrics, available_labels)
    save_comparison_csv(all_metrics, available_labels, metrics_dir)
    save_perclass_csv(all_metrics, available_labels, metrics_dir)

    # ---- Charts ------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Generating Charts")
    print(f"{'='*60}")
    plot_overall_charts(all_metrics, available_labels, plots_dir)
    plot_perclass_chart(all_metrics, available_labels, plots_dir)
    plot_radar_chart(all_metrics, available_labels, plots_dir)

    # ---- Analysis ----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Generating Analysis Report")
    print(f"{'='*60}")
    report = generate_analysis(all_metrics, available_labels, plots_dir)
    print("\n" + report)
    save_analysis(report, plots_dir)

    print(f"\n[DONE] All outputs saved to:")
    print(f"         Metrics : {metrics_dir}")
    print(f"         Charts  : {plots_dir}")
    print()


if __name__ == "__main__":
    main()
