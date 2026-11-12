# Model Card: Nirbhighne Vehicle Detector

## Model identity

| Item | Value |
| --- | --- |
| Weight file | Preferred: `weights/merged_v8m_from_s.pt`; fallbacks are selected by `config.MODEL_PATH` |
| Task | Object detection |
| Architecture | YOLOv8m-style `DetectionModel` (depth multiplier 0.67, width multiplier 0.75) |
| Input channels | 3 (RGB/BGR image data) |
| Detection classes | 11 |
| Runtime | Ultralytics YOLO with NVIDIA CUDA, Intel OpenVINO, Windows DirectML/ONNX, or CPU fallback |

## Intended use

Use this model for a classroom/capstone visualisation of vehicle detection,
tracking, and traffic-priority recommendations on recorded traffic video.

It must not be used to directly control physical traffic signals, make safety
decisions, or replace a trained traffic operator. The application is a
decision-support prototype only.

## Classes

1. Ambulance
2. Auto_ricksaw
3. Bike
4. Bus
5. Fire_truck
6. Leguna
7. Pickup
8. Police_car
9. Private_car
10. Truck
11. cng

## Dataset record

| Split | Images currently present | Source manifest |
| --- | ---: | --- |
| Train | 1,719 | `data.yml & utility file/data.yaml` |
| Validation | 64 | `data.yml & utility file/data.yaml` |
| Test | 46 | `data.yml & utility file/data.yaml` |

The dataset manifest uses YOLO text labels, where each annotation row normally
contains `class_id x_center y_center width height`, with coordinates normalised
to the image width and height. Class IDs must match the class order above.

## Training provenance -- complete before a formal submission

The repository contains the trained weights and dataset manifest, but the
original training record was not stored with the project. Fill in the following
from the training notebook/run logs rather than guessing:

| Required record | Value |
| --- | --- |
| Dataset source/version/export date | Not recorded in repository |
| Annotation guidelines and reviewer | Not recorded in repository |
| Training date | Not recorded in repository |
| Base checkpoint | Not recorded in repository |
| Image size / epochs / batch size | Not recorded in repository |
| Optimizer / learning rate / augmentation | Not recorded in repository |
| Hardware / CUDA / Python environment | Not recorded in repository |
| Training command or notebook cell | Not recorded in repository |

For future retraining, save the exact command, dataset export/version, random
seed, package versions, output `results.csv`, `args.yaml`, `best.pt`, and a
copy of the validation metrics in a dated experiment folder.

## Evaluation procedure

Run **Evaluate Model -> Standard validation** in the application, or call
`run_standard_evaluation()` with the active `config.MODEL_PATH` and the dataset
YAML. It makes a portable validation copy, converts HEIC/HEIF inputs to JPEG
when necessary, and writes `metrics.txt`, `metrics.json`, and Ultralytics plots.

Record the following results from the actual standard-validation run before
claiming performance in a report:

| Metric | Latest standard-validation result |
| --- | --- |
| Precision | 0.6188 |
| Recall | 0.7057 |
| mAP@0.50 | 0.6490 |
| mAP@0.50:0.95 | 0.4551 |
| F1 | 0.6594 |

These are the latest recorded standard-validation results in
`evaluation_results/metrics.json` for the repository's current model and
validation manifest. Re-run the evaluation after changing weights, labels, or
the dataset before using the values as a new performance claim.

Also attach the per-class table and confusion matrix. Overall mAP alone can
hide weak ambulance or fire-truck performance.

## Known limitations

- The model can miss small, occluded, unusual, night-time, or out-of-domain vehicles.
- Confidence is not a guarantee of correctness.
- Tracking IDs can change after occlusion, so cumulative counts are estimates.
- Ambulance priority uses a fixed high 0.60 confidence threshold plus three
  processed-frame confirmation; it can still create false alarms and is not
  safety certified.
- The current 1/2/4 lane modes represent independent video feeds, not calibrated
  geometric zones inside one camera frame.
