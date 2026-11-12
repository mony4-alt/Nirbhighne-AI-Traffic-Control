# Robustness Evaluation Pipeline
## YOLOv8m Vehicle Detector — Experiment Suite

This folder contains a complete, reproducible pipeline for evaluating the robustness of the trained **YOLOv8m** vehicle detection model under two challenging image conditions:

| Experiment | Description |
|---|---|
| **Grayscale** | RGB images converted to 3-channel grayscale |
| **Low-Resolution** | Images downscaled to 75% of original dimensions |

All reported metrics come **exclusively from actual model training and evaluation**. No values are fabricated.

---

## Prerequisites

### Python packages
Make sure these are installed in your environment:

```bash
pip install ultralytics>=8.2.0 opencv-python>=4.8.0 torch>=2.0.0 matplotlib numpy
```

> **GPU acceleration**: If you have an NVIDIA GPU, the scripts automatically use CUDA. Ensure PyTorch is installed with CUDA support:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### Dataset
Your dataset must be downloaded locally and structured as follows:

```
<dataset_root>/
├── images/
│   ├── train/      ← training images
│   ├── val/        ← validation images
│   └── test/       ← test images
└── labels/
    ├── train/      ← YOLO .txt annotation files
    ├── val/
    └── test/
```

The `test/` split is optional but recommended for evaluation.

---

## File Structure

```
robustness_eval/
├── config.py                  # Central configuration (hyperparameters, paths, class names)
├── generate_yamls.py          # Generates YOLO data YAML files for all variants
├── 01_prepare_grayscale.py    # Converts RGB dataset → 3-channel grayscale dataset
├── 02_prepare_lowres.py       # Resizes dataset to 75% of original dimensions
├── 03_train.py                # Trains YOLOv8m on a specified variant
├── 04_evaluate.py             # Evaluates a model; exports CSV + JSON metrics
├── 05_compare.py              # Builds comparison tables, charts, and analysis report
├── 06_run_all.py              # Master orchestrator — runs all steps in sequence
└── README.md                  # This file

results/                       # Auto-created by the pipeline
├── yamls/                     # Generated YOLO data YAML files
├── datasets/
│   ├── dataset_grayscale/     # Created by Step 1
│   └── dataset_lowres/        # Created by Step 2
├── weights/
│   ├── best_grayscale.pt      # Trained grayscale model
│   └── best_lowres.pt         # Trained low-res model
├── metrics/
│   ├── metrics_rgb.csv        # Evaluation metrics (original model)
│   ├── metrics_rgb.json
│   ├── metrics_grayscale.csv
│   ├── metrics_grayscale.json
│   ├── metrics_lowres.csv
│   ├── metrics_lowres.json
│   ├── metrics_summary.csv    # Final comparison table
│   └── metrics_perclass.csv   # Per-class mAP@50 table
├── plots/
│   ├── chart_precision.png
│   ├── chart_recall.png
│   ├── chart_map50.png
│   ├── chart_map5095.png
│   ├── chart_f1.png
│   ├── chart_perclass_map50.png
│   ├── chart_radar.png
│   └── analysis_report.txt    # Data-driven analysis report
└── runs/                      # YOLO training run outputs (logs, curves, etc.)
```

---

## Option A: Full Pipeline (recommended)

Run everything in one command:

```bash
python robustness_eval/06_run_all.py \
    --dataset_root   "C:/path/to/your/dataset" \
    --original_model "C:/path/to/VehicleDetector/weights/best.pt" \
    --eval_root      "C:/path/to/VehicleDetector/robustness_eval"
```

This runs all 9 steps sequentially:
1. Generate YAML files
2. Create grayscale dataset
3. Create low-res dataset
4. Train grayscale model (~100 epochs)
5. Train low-res model (~100 epochs)
6. Evaluate original RGB model
7. Evaluate grayscale model
8. Evaluate low-res model
9. Generate comparison tables + charts

### Skip flags (for resuming or re-running partial steps)

```bash
# Skip dataset preparation (datasets already exist)
python 06_run_all.py ... --skip_prepare

# Skip training (models already exist)
python 06_run_all.py ... --skip_training

# Only regenerate comparison outputs (all evaluations already done)
python 06_run_all.py ... --skip_prepare --skip_training --skip_eval

# Force-overwrite everything
python 06_run_all.py ... --force
```

---

## Option B: Run Steps Individually

### Step 1 — Prepare Grayscale Dataset
```bash
python robustness_eval/01_prepare_grayscale.py \
    --dataset_root "C:/path/to/dataset" \
    --eval_root    "C:/path/to/robustness_eval"
```

### Step 2 — Prepare Low-Resolution Dataset
```bash
python robustness_eval/02_prepare_lowres.py \
    --dataset_root "C:/path/to/dataset" \
    --eval_root    "C:/path/to/robustness_eval"
```

### Step 3 — Train on Grayscale Dataset
```bash
python robustness_eval/03_train.py \
    --variant      grayscale \
    --dataset_root "C:/path/to/dataset" \
    --eval_root    "C:/path/to/robustness_eval"
```

### Step 4 — Train on Low-Resolution Dataset
```bash
python robustness_eval/03_train.py \
    --variant      lowres \
    --dataset_root "C:/path/to/dataset" \
    --eval_root    "C:/path/to/robustness_eval"
```

### Step 5 — Evaluate Original Model
```bash
python robustness_eval/04_evaluate.py \
    --model    "C:/path/to/weights/best.pt" \
    --data     "C:/path/to/robustness_eval/results/yamls/data_rgb.yaml" \
    --label    rgb \
    --eval_root "C:/path/to/robustness_eval"
```

### Step 6 — Evaluate Grayscale Model
```bash
python robustness_eval/04_evaluate.py \
    --model    "C:/path/to/robustness_eval/results/weights/best_grayscale.pt" \
    --data     "C:/path/to/robustness_eval/results/yamls/data_grayscale.yaml" \
    --label    grayscale \
    --eval_root "C:/path/to/robustness_eval"
```

### Step 7 — Evaluate Low-Resolution Model
```bash
python robustness_eval/04_evaluate.py \
    --model    "C:/path/to/robustness_eval/results/weights/best_lowres.pt" \
    --data     "C:/path/to/robustness_eval/results/yamls/data_lowres.yaml" \
    --label    lowres \
    --eval_root "C:/path/to/robustness_eval"
```

### Step 8 — Generate Comparison Report
```bash
python robustness_eval/05_compare.py \
    --eval_root "C:/path/to/robustness_eval"
```

---

## Training Configuration

| Parameter  | Value |
|---|---|
| Model      | YOLOv8m (pretrained backbone) |
| Epochs     | 100 |
| Image size | 640 |
| Batch size | 16 |
| Optimizer  | auto |
| Patience   | 20 (early stopping) |
| Seed       | 42 |
| Device     | Auto (CUDA if available) |

These settings are **identical across all three experiments** to ensure fair comparison.

---

## Classes (11 total)

```
0: Ambulance
1: Auto_ricksaw
2: Bike
3: Bus
4: Fire_truck
5: Leguna
6: Pickup
7: Police_car
8: Private_car
9: Truck
10: cng
```

---

## Expected Outputs

After a complete run, the `results/` folder will contain:

| File | Description |
|---|---|
| `metrics/metrics_summary.csv` | Final comparison table (all 3 experiments) |
| `metrics/metrics_perclass.csv` | Per-class mAP@50 for each experiment |
| `plots/chart_map50.png` | Bar chart: mAP@50 comparison |
| `plots/chart_map5095.png` | Bar chart: mAP@50-95 comparison |
| `plots/chart_perclass_map50.png` | Grouped bar chart by class |
| `plots/chart_radar.png` | Radar chart: all 4 metrics overlaid |
| `plots/analysis_report.txt` | Data-driven analysis (no invented conclusions) |
| `weights/best_grayscale.pt` | Trained grayscale model |
| `weights/best_lowres.pt` | Trained low-res model |

---

## Notes

- **Original model is never overwritten.** The pipeline only reads `best.pt` for evaluation; it does not modify it.
- **Dataset preparation is idempotent.** Running the prep scripts again without `--force` will skip conversion if the output folder already exists.
- **Evaluation uses the `test` split by default.** Change with `--eval_split val` if your dataset has no test split.
- **The analysis report is purely data-driven.** All conclusions are drawn directly from the actual metric values — no text is pre-written or invented.
