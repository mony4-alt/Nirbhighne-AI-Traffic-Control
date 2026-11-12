import os

import sys

# Base Directories
if getattr(sys, 'frozen', False):
    # Running inside PyInstaller bundle
    BASE_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    # Running as script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WEIGHTS_DIR      = os.path.join(BASE_DIR, "weights")
OUTPUT_DIR       = os.path.join(BASE_DIR, "output")
LOGS_DIR         = os.path.join(BASE_DIR, "logs")
ASSETS_DIR       = os.path.join(BASE_DIR, "assets")
ICONS_DIR        = os.path.join(BASE_DIR, "icons")
EVAL_RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")

ROBUSTNESS_EVAL_DIR              = os.path.join(BASE_DIR, "robustness_eval")
ROBUSTNESS_TEMP_GRAYSCALE_DIR     = os.path.join(ROBUSTNESS_EVAL_DIR, "temp", "grayscale")
ROBUSTNESS_RESULTS_GRAYSCALE_DIR  = os.path.join(ROBUSTNESS_EVAL_DIR, "results", "grayscale")

# Ensure required directories exist
for path in [WEIGHTS_DIR, OUTPUT_DIR, LOGS_DIR, ASSETS_DIR, ICONS_DIR, EVAL_RESULTS_DIR, ROBUSTNESS_EVAL_DIR, ROBUSTNESS_TEMP_GRAYSCALE_DIR, ROBUSTNESS_RESULTS_GRAYSCALE_DIR]:
    os.makedirs(path, exist_ok=True)

# Persists the last-selected data.yaml path across sessions
EVAL_YAML_CACHE = os.path.join(LOGS_DIR, "eval_yaml_path.txt")
_bundled_eval_yaml = os.path.join(BASE_DIR, "evaluation_data", "data.yaml")
_project_eval_yaml = os.path.join(BASE_DIR, "data.yml & utility file", "portable_eval.yaml")
DEFAULT_EVAL_YAML = next(
    (path for path in (_bundled_eval_yaml, _project_eval_yaml) if os.path.isfile(path)),
    ""
)

# Files — priority: optimized merged_v8m_from_s.pt (alpha=0.02, highest Recall & mAP50-95), fallback best.pt
_candidates = [
    os.path.join(WEIGHTS_DIR, "merged_v8m_from_s.pt"),  # Optimized merged model (Best Recall & mAP50-95)
    os.path.join(WEIGHTS_DIR, "best.pt"),               # Original v8m
    os.path.join(WEIGHTS_DIR, "final_best_v8m.pt"),
    os.path.join(BASE_DIR,    "final_best_v8m.pt"),
]
MODEL_PATH = next((p for p in _candidates if os.path.isfile(p)), os.path.join(WEIGHTS_DIR, "best.pt"))



LOG_PATH   = os.path.join(LOGS_DIR, "detector.log")
APP_ICON_ICO = os.path.join(ICONS_DIR, "app_icon.ico")
APP_ICON_PNG = os.path.join(ICONS_DIR, "app_icon.png")


def sync_app_icons():
    """Automatically keep app_icon.ico synced if user replaces app_icon.png."""
    if os.path.isfile(APP_ICON_PNG):
        try:
            if not os.path.isfile(APP_ICON_ICO) or os.path.getmtime(APP_ICON_PNG) > os.path.getmtime(APP_ICON_ICO):
                from PIL import Image
                img = Image.open(APP_ICON_PNG)
                sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
                img.save(APP_ICON_ICO, format="ICO", sizes=sizes)
                assets_ico = os.path.join(ASSETS_DIR, "app_icon.ico")
                assets_png = os.path.join(ASSETS_DIR, "app_icon.png")
                img.save(assets_ico, format="ICO", sizes=sizes)
                img.save(assets_png)
        except Exception:
            pass

sync_app_icons()



# Application Settings
APP_TITLE = "Nirbhighne: AI-Driven Dynamic Traffic Signal Controlling System"
APP_SUBTITLE = ""
DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 820

# YOLO Detector Settings// GPU,CPU 
DEFAULT_CONFIDENCE = 0.30          # Better recall for small/partly occluded vehicles
# Ambulance is the deliberate confidence exception: it always needs a strong
# 60% model score, regardless of the normal-vehicle GUI slider. This reduces
# false emergency priorities; three-frame confirmation remains an extra guard.
AMBULANCE_CONFIDENCE = 0.60
DEFAULT_IOU = 0.45
INFERENCE_IMGSZ = 1280             # Larger input improves far-row car detection
# The v8m model is expensive at 1280px.  A 4 GB card such as the GTX 1650
# uses a smaller input and one lane at a time in 4-lane mode to avoid silent
# CUDA memory pressure while preserving more detail than a 640px-only setup.
GPU_SMALL_VRAM_GB = 6.0
# A 3 GB-or-less card should remain on one image at a time.  The GTX 1650
# profile below has enough headroom for two 640px v8m images, which reduces
# repeated per-lane scheduling overhead without requiring a large GPU.
GPU_TINY_VRAM_GB = 3.5
GPU_SMALL_INFERENCE_IMGSZ_SINGLE = 960
GPU_SMALL_INFERENCE_IMGSZ_MULTI = 640
GPU_TINY_MULTI_BATCH = 1
GPU_SMALL_MULTI_BATCH = 2
GPU_LARGE_INFERENCE_IMGSZ_MULTI = 960
GPU_LARGE_MULTI_BATCH = 4
# Intel OpenVINO uses the integrated/discrete Intel GPU for model inference.
# The CPU continues decoding, tracking, drawing, and writing video frames.
# Keep this fixed because the exported OpenVINO graph is shape-specialised.
INTEL_OPENVINO_ENABLED = True
INTEL_OPENVINO_INFERENCE_IMGSZ = 640
# DirectML provides the Windows fallback accelerator for AMD Radeon GPUs and
# DirectX 12-capable Intel GPUs when OpenVINO is unavailable.
DIRECTML_ENABLED = True
DIRECTML_INFERENCE_IMGSZ = 640
# CPU mode keeps the original video for display/output, but uses these smaller
# internal model inputs so CPU-only machines can analyse every frame faster.
# The multi-feed value is lower because each feed is shown in a smaller grid.
CPU_INFERENCE_IMGSZ_SINGLE = 768
CPU_INFERENCE_IMGSZ_MULTI  = 640

# CPU execution profile.  Four PyTorch workers is a good balance for YOLO on
# typical 4- to 8-logical-core PCs; using every logical core can be slower due
# to thread contention.  OpenCV is kept single-threaded while YOLO runs so it
# does not compete with those workers.  These settings are applied only when
# the CPU device is selected.
CPU_TORCH_THREADS = 4
CPU_OPENCV_THREADS = 1
# CUDA/OpenVINO/DirectML do the model work on the accelerator; keeping OpenCV
# serial prevents four feed readers from oversubscribing the host CPU.
ACCELERATOR_OPENCV_THREADS = 1

# CPU hardware profiles. With CPU_AUTO_PROFILE enabled, the detector chooses
# one profile at model load from logical CPU cores and physical memory. The
# low profile prioritises continuous processing on modest hardware; the high
# profile spends the available CPU capacity on small/distant-object recall.
CPU_AUTO_PROFILE = True
CPU_LOW_MAX_CORES = 4
CPU_LOW_MAX_RAM_GB = 8
CPU_HIGH_MIN_CORES = 12
CPU_HIGH_MIN_RAM_GB = 16
CPU_LOW_TORCH_THREADS = 2
CPU_HIGH_TORCH_THREADS = 6
CPU_INFERENCE_IMGSZ_LOW_SINGLE = 640
CPU_INFERENCE_IMGSZ_LOW_MULTI = 512
CPU_INFERENCE_IMGSZ_HIGH_SINGLE = 960
CPU_INFERENCE_IMGSZ_HIGH_MULTI = 768

# A 2/4-camera view is displayed in a small grid, so building a full-resolution
# composite wastes CPU and memory only to have Tk immediately shrink it again.
# This cap affects the live multi-camera preview only--not YOLO inference or
# saved per-lane videos.  INTER_AREA downscaling keeps the preview smooth.
CPU_MULTI_DISPLAY_MAX_WIDTH = 1600
CPU_MULTI_DISPLAY_MAX_HEIGHT = 900
CPU_LOW_MULTI_DISPLAY_MAX_WIDTH = 1280
CPU_LOW_MULTI_DISPLAY_MAX_HEIGHT = 720
CPU_HIGH_MULTI_DISPLAY_MAX_WIDTH = 1920
CPU_HIGH_MULTI_DISPLAY_MAX_HEIGHT = 1080
MAX_DETECTIONS = 300
# Detection overlays are drawn after inference, so these settings affect both
# CUDA and CPU output equally.  The thicker line remains visible after a 4-lane
# preview is downscaled into a 2x2 grid.
DETECTION_BOX_THICKNESS = 5
DETECTION_AMBULANCE_BOX_THICKNESS = 7
DETECTION_LABEL_FONT_SCALE = 0.55
DETECTION_LABEL_THICKNESS = 2

# Validation is deliberately conservative.  On older 4 GB NVIDIA cards,
# half-precision batched validation can return empty metrics without raising an
# error; full precision with a small batch is reliable and still fast enough.
EVAL_BATCH_SIZE = 4
EVAL_USE_HALF = False
AMBULANCE_CONFIRMATION_FRAMES = 3  # Filters one-frame false emergency alerts
AMBULANCE_CONFIRMATION_CONFIDENCE = 0.60
AMBULANCE_CONFIRMATION_MIN_AVG_CONFIDENCE = 0.60
AMBULANCE_CONFIRMATION_HISTORY = 5
AMBULANCE_LOST_GRACE_FRAMES = 1
OUTPUT_WRITER_QUEUE_SIZE = 32

# Frame-skipping — process every Nth frame (1 = no skip, 2 = every other, etc.)
# Increase to trade latency for throughput on CPU or slow machines.
FRAME_SKIP_INTERVAL = 1

# Optional CPU-only adaptive skipping. It is OFF by default so CPU mode keeps
# every source frame for the best possible counting continuity. Set a positive
# target only when faster processing matters more than examining every frame.
CPU_ADAPTIVE_FPS_TARGET = 0
CPU_MAX_FRAME_SKIP = 4

# Ambulance EMA confidence smoothing (0 < alpha <= 1).
# Lower alpha = smoother but slower to react; higher = more reactive.
AMBULANCE_EMA_ALPHA = 0.35

# Signal hysteresis — minimum consecutive frames a lane must stay 'busiest'
# before it is promoted to GREEN.  Prevents rapid flipping between lanes.
SIGNAL_HOLD_MIN_FRAMES = 8

# Congestion classification thresholds (vehicles in a lane)
CONGESTION_LOW      = 5    # < 5   → Low
CONGESTION_MEDIUM   = 15   # 5-14  → Medium
CONGESTION_HIGH     = 30   # 15-29 → High
                            # >= 30 → Critical

# GUI canvas max refresh rate (frames per second)
GUI_RENDER_FPS_CAP = 30

# Live camera capture settings. These are requests rather than guarantees:
# OpenCV keeps the camera's supported mode when a request is rejected.
LIVE_CAMERA_WIDTH = 1280
LIVE_CAMERA_HEIGHT = 720
LIVE_CAMERA_FPS = 30
LIVE_CAMERA_BUFFER_DRAIN = 2  # discard stale frames when inference is slower

# Ambulance emergency overlay settings
AMBULANCE_OVERLAY_BORDER_THICKNESS = 18   # px — flashing red border on video frame
AMBULANCE_OVERLAY_ALPHA = 0.55            # Blend alpha for tinted overlay
AMBULANCE_DIRECTION_MIN_FRAMES = 4        # Min frames of consistent motion to infer direction
AMBULANCE_DIRECTION_HISTORY = 12          # Frames of centroid history kept per lane

# Traffic signal timer (seconds)
MIN_SIGNAL_GREEN_SEC = 8           # Minimum green time regardless of density
MAX_SIGNAL_GREEN_SEC = 90          # Maximum green time for the busiest lane
SIGNAL_BASE_GREEN_SEC = 20         # Base green time when densities are equal

# Sound alert (Windows only — set False to silence)
AMBULANCE_SOUND_ALERT = True
AMBULANCE_BEEP_HZ = 1100
AMBULANCE_BEEP_MS = 320

# Target vehicle classes mapping (lower case key to display name value)
VEHICLE_CLASSES = {
    'car': 'Car',
    'private_car': 'Car',
    'bus': 'Bus',
    'truck': 'Truck',
    'motorcycle': 'Motorcycle',
    'bicycle': 'Bicycle',
    # Common alternate annotations
    'bike': 'Motorcycle',
    'auto_ricksaw': 'Auto Ricksaw',
    'cng': 'CNG Ricksaw',
    'ambulance': 'Ambulance',
    'fire_truck': 'Fire Truck',
    'police_car': 'Police Car',
    'pickup': 'Pickup Truck',
    'leguna': 'Leguna',
}
