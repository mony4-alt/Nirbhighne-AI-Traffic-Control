"""detector.py — VehicleDetector: multi-lane, multi-threaded YOLO inference.

Key design
──────────
* Multi-lane camera feeds are read on their own reader threads
  (ThreadPoolExecutor); a single feed is read directly to avoid scheduling
  overhead.
* Lane frames are read in parallel, then sent through a hardware-aware batch
  (or safe per-lane chunks on small GPUs and fixed-shape accelerators).
* Ambulance detection triggers a full-frame emergency overlay (flashing border
  + directional arrow) drawn directly on the OpenCV frame with cv2 primitives.
* Per-lane signal timers are computed from vehicle density.
"""

from __future__ import annotations

import os
import threading
import queue
import time
import collections
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import config
from traffic_logic import (
    normalise_lane_count,
    calculate_weighted_demand,
    traffic_signal_advice,
    build_lane_signal_states,
    compute_green_duration,
    AmbulanceDirectionTracker,
    LaneWaitingTimeTracker,
    TrafficDemandSmoother,
)
from utils import logger


# Keep OpenCV's native optimised code paths enabled for colour conversion,
# resizing, drawing, and video I/O. This does not alter the model, confidence
# threshold, or source image resolution.
cv2.setUseOptimized(True)


# ─────────────────────────────────────────────────────────────────────────────
# Stats dataclass
# ─────────────────────────────────────────────────────────────────────────────

class DetectionStats:
    """Struct to hold performance and counting statistics for the GUI."""
    def __init__(self):
        self.fps = 0.0
        self.inference_time_ms = 0.0
        self.frame_number = 0
        self.total_frames = 0
        self.elapsed_time = 0.0
        self.progress_percent = 0.0
        self.resolution = "0x0"
        self.current_count = 0
        self.total_detected = 0
        self.class_counts_current: dict = {}
        self.class_counts_total: dict = {}
        self.lane_count = 1
        self.lane_vehicle_counts: dict[int, int] = {}
        self.ambulance_lanes: list[int] = []
        self.ambulance_counts_by_lane: dict[int, int] = {}
        self.ambulance_proximity_by_lane: dict[int, float] = {}
        self.priority_ambulance_lane: int | None = None
        self.ambulance_directions: dict[int, str] = {}   # lane → 'left'/'right'/'unknown'
        self.signal_lane: int | None = None
        self.signal_advice = "Select a lane mode to receive traffic advice"
        self.emergency_advice = "No ambulance detected"
        self.lane_traffic_demand: dict[int, float] = {}
        self.lane_waiting_times: dict[int, float] = {}
        self.green_durations: dict[int, int] = {}        # lane → seconds
        self.lane_signal_states: dict[int, str] = {}     # lane → 'GREEN'/'RED'/'YIELD'/'EMERGENCY'


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rgb_to_bgr(rgb):
    r, g, b = rgb
    return (b, g, r)


# ─────────────────────────────────────────────────────────────────────────────
# VehicleDetector
# ─────────────────────────────────────────────────────────────────────────────

class VehicleDetector:
    """Manages background YOLOv8 inference for one or more independent lane feeds.

    Architecture
    ────────────
    • `start()` launches a single worker thread (_run_inference_loop).
    • Inside that loop, a ThreadPoolExecutor reads frames from all lanes in
      parallel, then the results are batched into one GPU call.
    • A separate output-writer thread per lane handles disk I/O so encoding
      never blocks inference.
    """

    def __init__(self, model_path=config.MODEL_PATH):
        self.model_path = model_path
        self.device = "cpu"
        self.model: YOLO | None = None
        self.stats = DetectionStats()

        # Keep the application's normal library thread settings so a later
        # switch back to CUDA is not affected by the CPU execution profile.
        self._default_torch_threads = torch.get_num_threads()
        self._default_opencv_threads = cv2.getNumThreads()
        self._cpu_runtime_configured = False
        self._cpu_profile_name = "balanced"
        self._cpu_profile: dict[str, int] = {}

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._worker_thread: threading.Thread | None = None
        self.current_state = "Idle"

        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._frame_sequence = 0

        self.event_queue: queue.Queue = queue.Queue()
        self._stats_event_lock = threading.Lock()
        self._stats_event_pending = False

        self.grayscale_mode: bool = False
        self.lane_count: int = 1
        self.confidence: float = config.DEFAULT_CONFIDENCE
        self.ambulance_confidence: float = config.AMBULANCE_CONFIDENCE

        # Ambulance streaks (lane → consecutive-frame count)
        self._ambulance_lane_streaks: dict[int, int] = {}
        self._ambulance_confidence_history: dict[int, collections.deque[float]] = {}
        self._ambulance_missed_frames: dict[int, int] = {}
        # Retain the last confirmed details during the short lost-detection
        # grace period so one missed frame cannot change emergency priority.
        self._ambulance_counts_by_lane: dict[int, int] = {}
        self._ambulance_proximity_by_lane: dict[int, float] = {}
        # Direction tracker
        self._dir_tracker = AmbulanceDirectionTracker()
        self._demand_smoother = TrafficDemandSmoother()
        self._waiting_tracker = LaneWaitingTimeTracker()

        # Overlay flash state
        self._overlay_flash_state: bool = False
        self._overlay_flash_counter: int = 0

        self._stable_signal_lane: int | None = None
        self._candidate_signal_lane: int | None = None
        self._candidate_signal_frames: int = 0

        # Class colours (RGB) — converted to BGR when needed
        self._class_colors_rgb = {
            'car':          (16,  185, 129),
            'bus':          (59,  130, 246),
            'truck':        (245, 158,  11),
            'motorcycle':   (6,   182, 212),
            'bike':         (6,   182, 212),
            'bicycle':      (236,  72, 153),
            'ambulance':    (239,  68,  68),
            'police_car':   (59,  130, 246),
            'fire_truck':   (239,  68,  68),
            'auto_ricksaw': (250, 204,  21),
            'cng':          (16,  185, 129),
            'pickup':       (168,  85, 247),
            'leguna':       (14,  165, 233),
            'private_car':  (16,  185, 129),
        }
        self._default_color_rgb = (156, 163, 175)

    # ─────────────────────────────────────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _system_memory_gb() -> float:
        """Return installed physical memory in GiB without adding a dependency."""
        if os.name == "nt":
            try:
                import ctypes

                class MemoryStatus(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                status = MemoryStatus()
                status.dwLength = ctypes.sizeof(status)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                    return status.ullTotalPhys / (1024 ** 3)
            except Exception:
                pass
        return 0.0

    def _select_cpu_profile(self) -> dict[str, int]:
        """Choose a compact CPU profile without changing the inference pipeline."""
        cores = os.cpu_count() or 1
        ram_gb = self._system_memory_gb()
        low_end = (
            cores <= config.CPU_LOW_MAX_CORES
            or (ram_gb > 0 and ram_gb < config.CPU_LOW_MAX_RAM_GB)
        )
        high_end = (
            cores >= config.CPU_HIGH_MIN_CORES
            and (ram_gb == 0 or ram_gb >= config.CPU_HIGH_MIN_RAM_GB)
        )

        if config.CPU_AUTO_PROFILE and low_end:
            self._cpu_profile_name = "low"
            profile = {
                "threads": config.CPU_LOW_TORCH_THREADS,
                "single_imgsz": config.CPU_INFERENCE_IMGSZ_LOW_SINGLE,
                "multi_imgsz": config.CPU_INFERENCE_IMGSZ_LOW_MULTI,
                "preview_width": config.CPU_LOW_MULTI_DISPLAY_MAX_WIDTH,
                "preview_height": config.CPU_LOW_MULTI_DISPLAY_MAX_HEIGHT,
            }
        elif config.CPU_AUTO_PROFILE and high_end:
            self._cpu_profile_name = "high"
            profile = {
                "threads": config.CPU_HIGH_TORCH_THREADS,
                "single_imgsz": config.CPU_INFERENCE_IMGSZ_HIGH_SINGLE,
                "multi_imgsz": config.CPU_INFERENCE_IMGSZ_HIGH_MULTI,
                "preview_width": config.CPU_HIGH_MULTI_DISPLAY_MAX_WIDTH,
                "preview_height": config.CPU_HIGH_MULTI_DISPLAY_MAX_HEIGHT,
            }
        else:
            self._cpu_profile_name = "balanced"
            profile = {
                "threads": config.CPU_TORCH_THREADS,
                "single_imgsz": config.CPU_INFERENCE_IMGSZ_SINGLE,
                "multi_imgsz": config.CPU_INFERENCE_IMGSZ_MULTI,
                "preview_width": config.CPU_MULTI_DISPLAY_MAX_WIDTH,
                "preview_height": config.CPU_MULTI_DISPLAY_MAX_HEIGHT,
            }

        self._cpu_profile = profile
        return profile

    @staticmethod
    def intel_gpu_available() -> bool:
        """Return whether OpenVINO can execute on an Intel GPU."""
        if not config.INTEL_OPENVINO_ENABLED:
            return False
        try:
            import openvino as ov

            core = ov.Core()
            return any(
                device.startswith("GPU")
                and "intel" in str(core.get_property(device, "FULL_DEVICE_NAME")).lower()
                for device in core.available_devices
            )
        except Exception:
            return False

    @staticmethod
    def directml_available() -> bool:
        """Return whether Windows DirectML can use a compatible GPU."""
        if not config.DIRECTML_ENABLED or os.name != "nt":
            return False
        try:
            import onnxruntime

            return "DmlExecutionProvider" in onnxruntime.get_available_providers()
        except Exception:
            return False

    @classmethod
    def preferred_accelerator(cls) -> str | None:
        """Prefer NVIDIA CUDA, Intel OpenVINO, then Windows DirectML."""
        if torch.cuda.is_available():
            return "cuda"
        if cls.intel_gpu_available():
            return "intel:gpu"
        return "directml" if cls.directml_available() else None

    @classmethod
    def preferred_device(cls) -> str:
        return cls.preferred_accelerator() or "cpu"

    def _openvino_model_dir(self) -> str:
        stem, _ = os.path.splitext(os.path.abspath(self.model_path))
        return f"{stem}_openvino_model"

    def _ensure_openvino_model(self) -> str:
        """Export once, then reuse the OpenVINO graph until weights change."""
        model_dir = self._openvino_model_dir()
        xml_files = [
            os.path.join(model_dir, name)
            for name in os.listdir(model_dir)
            if name.endswith(".xml")
        ] if os.path.isdir(model_dir) else []
        if xml_files and max(os.path.getmtime(path) for path in xml_files) >= os.path.getmtime(self.model_path):
            return model_dir

        logger.info("Exporting %s for Intel OpenVINO GPU inference.", os.path.basename(self.model_path))
        source_model = YOLO(self.model_path)
        exported = source_model.export(
            format="openvino",
            imgsz=config.INTEL_OPENVINO_INFERENCE_IMGSZ,
            dynamic=False,
            nms=False,
        )
        return str(exported)

    def _onnx_model_path(self) -> str:
        stem, _ = os.path.splitext(os.path.abspath(self.model_path))
        return f"{stem}.onnx"

    def _ensure_onnx_model(self) -> str:
        """Export a fixed-shape ONNX graph for DirectML and reuse it."""
        onnx_path = self._onnx_model_path()
        if os.path.isfile(onnx_path) and os.path.getmtime(onnx_path) >= os.path.getmtime(self.model_path):
            return onnx_path

        logger.info("Exporting %s for DirectML inference.", os.path.basename(self.model_path))
        source_model = YOLO(self.model_path)
        return str(source_model.export(
            format="onnx",
            imgsz=config.DIRECTML_INFERENCE_IMGSZ,
            dynamic=False,
            simplify=False,
        ))

    @staticmethod
    def _enable_directml_onnx_backend():
        """Teach Ultralytics' ONNX backend to use DirectML for this session."""
        from ultralytics.nn.backends.onnx import ONNXBackend

        if getattr(ONNXBackend, "_vehicle_detector_directml_patch", False):
            ONNXBackend._vehicle_detector_directml_enabled = True
            return

        original_load_model = ONNXBackend.load_model

        def load_model_with_directml(self, weight):
            if not getattr(ONNXBackend, "_vehicle_detector_directml_enabled", False):
                return original_load_model(self, weight)

            import onnxruntime

            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            self.session = onnxruntime.InferenceSession(str(weight), providers=providers)
            self.output_names = [output.name for output in self.session.get_outputs()]
            metadata_map = self.session.get_modelmeta().custom_metadata_map
            if metadata_map:
                self.apply_metadata(dict(metadata_map))
            self.dynamic = isinstance(self.session.get_outputs()[0].shape[0], str)
            self.fp16 = "float16" in self.session.get_inputs()[0].type
            # DirectML does not support ONNX Runtime's parallel session mode or
            # concurrent calls on one session, so keep this detector's single
            # inference worker and standard tensor-to-NumPy handoff.
            self.use_io_binding = False

        ONNXBackend.load_model = load_model_with_directml
        ONNXBackend._vehicle_detector_directml_patch = True
        ONNXBackend._vehicle_detector_directml_enabled = True

    def _configure_runtime_for_device(self, log_profile: bool = True):
        """Apply the small, CPU-only runtime profile used during inference."""
        if self.device == "cpu":
            profile = self._select_cpu_profile()
            torch_threads = max(1, min(profile["threads"], os.cpu_count() or 1))
            torch.set_num_threads(torch_threads)
            # One OpenCV worker avoids oversubscribing modest CPUs while Torch
            # is running the model.  Video decoding still happens in the lane
            # reader threads.
            cv2.setNumThreads(max(1, config.CPU_OPENCV_THREADS))
            torch.backends.mkldnn.enabled = True
            self._cpu_runtime_configured = True
            if log_profile:
                logger.info(
                    "CPU %s profile: %s logical core(s), %.1f GiB RAM, %s Torch thread(s), %s OpenCV thread(s), MKL-DNN enabled.",
                    self._cpu_profile_name, os.cpu_count() or 1, self._system_memory_gb(),
                    torch_threads, max(1, config.CPU_OPENCV_THREADS),
                )
        elif self._cpu_runtime_configured:
            torch.set_num_threads(self._default_torch_threads)
            cv2.setNumThreads(self._default_opencv_threads)
            self._cpu_runtime_configured = False
        if self.device != "cpu":
            # Accelerator backends still decode, resize, and draw on the
            # host. Keep that work predictable instead of starting an OpenCV
            # thread pool for every lane.
            cv2.setNumThreads(max(1, config.ACCELERATOR_OPENCV_THREADS))

    def load_model(self, device: str | None = None):
        """Load YOLO model onto the requested device with device-specific optimisations."""
        if not os.path.exists(self.model_path):
            logger.error(f"Model not found: {self.model_path}")
            raise FileNotFoundError(self.model_path)

        old_cwd = os.getcwd()
        model_dir = os.path.dirname(os.path.abspath(self.model_path))
        model_name = os.path.basename(self.model_path)

        requested_device = device or self.preferred_device()
        if requested_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU.")
            requested_device = "cpu"
        if requested_device == "intel:gpu" and not self.intel_gpu_available():
            logger.warning("Intel OpenVINO GPU requested but unavailable; falling back to CPU.")
            requested_device = "cpu"
        if requested_device == "directml" and not self.directml_available():
            logger.warning("DirectML requested but unavailable; falling back to CPU.")
            requested_device = "cpu"
        self.device = requested_device

        logger.info(f"Loading YOLO from {model_name} on {self.device}")
        try:
            self._configure_runtime_for_device()
            if self.device == "intel:gpu":
                openvino_model_dir = self._ensure_openvino_model()
                self.model = YOLO(openvino_model_dir, task="detect")
                self.model.predict(
                    np.zeros((640, 640, 3), dtype=np.uint8),
                    device="intel:gpu",
                    imgsz=self._inference_imgsz(1),
                    verbose=False,
                )
                self.current_state = "Idle"
                logger.info("YOLO loaded on Intel OpenVINO GPU.")
                return
            if self.device == "directml":
                onnx_model_path = self._ensure_onnx_model()
                self._enable_directml_onnx_backend()
                self.model = YOLO(onnx_model_path, task="detect")
                self.model.predict(
                    np.zeros((640, 640, 3), dtype=np.uint8),
                    device="cpu",
                    imgsz=self._inference_imgsz(1),
                    verbose=False,
                )
                self.current_state = "Idle"
                logger.info("YOLO loaded on Windows DirectML GPU.")
                return
            if self.device == "cuda":
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                # TF32 gives ~3x throughput on Ampere+ GPUs (safe for detection)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                # Prefer the cuDNN autotuner to find the fastest conv algorithm
                torch.backends.cudnn.benchmark_limit = 0
                torch.cuda.empty_cache()
                # Set CUDA stream priority: high-priority stream for inference
                try:
                    torch.cuda.set_per_process_memory_fraction(0.85)
                except Exception:
                    pass
            os.chdir(model_dir)
            self.model = YOLO(model_name)
            self.model.to(self.device)
            # Fuse convolution + batch-normalisation layers for both CUDA and
            # CPU.  The latter is especially valuable on CPU-only installs.
            if hasattr(self.model.model, "fuse"):
                self.model.model.fuse()
            # Warm-up so the first real frame doesn't pay compile/cache cost
            warmup = np.zeros((640, 640, 3), dtype=np.uint8)
            warmup_kwargs = {
                "device": self.device,
                "imgsz": self._inference_imgsz(1),
                "verbose": False,
            }
            # Do not pass half=False on CPU: newer Ultralytics versions warn
            # about that deprecated argument.  CUDA keeps its existing FP16
            # acceleration unchanged.
            if self.device == "cuda":
                warmup_kwargs["half"] = True
            self.model.predict(warmup, **warmup_kwargs)
            # Ultralytics configures its own predictor during the first call
            # and can replace Torch's CPU thread count.  Set our CPU profile
            # again afterwards so it remains in effect for every real frame.
            if self.device == "cpu":
                self._configure_runtime_for_device(log_profile=False)
            self.current_state = "Idle"
            logger.info(f"YOLO loaded on {self.device}.")
        except Exception as exc:
            if self.device == "cuda":
                logger.warning("CUDA load failed (%s); trying CPU fallback.", exc)
                try:
                    torch.cuda.empty_cache()
                    self.model = YOLO(model_name)
                    self.model.to("cpu")
                    self.device = "cpu"
                    self._configure_runtime_for_device()
                    if hasattr(self.model.model, "fuse"):
                        self.model.model.fuse()
                    self.model.predict(
                        np.zeros((640, 640, 3), dtype=np.uint8),
                        device="cpu",
                        imgsz=self._inference_imgsz(1),
                        verbose=False,
                    )
                    self._configure_runtime_for_device(log_profile=False)
                    self.current_state = "Idle"
                    logger.info("YOLO loaded on CPU fallback.")
                    return
                except Exception:
                    logger.exception("CPU fallback also failed")
            elif self.device == "intel:gpu":
                logger.warning("Intel OpenVINO GPU load failed (%s); trying CPU fallback.", exc)
                try:
                    self.device = "cpu"
                    self._configure_runtime_for_device()
                    os.chdir(model_dir)
                    self.model = YOLO(model_name)
                    self.model.to("cpu")
                    if hasattr(self.model.model, "fuse"):
                        self.model.model.fuse()
                    self.model.predict(
                        np.zeros((640, 640, 3), dtype=np.uint8),
                        device="cpu",
                        imgsz=self._inference_imgsz(1),
                        verbose=False,
                    )
                    self._configure_runtime_for_device(log_profile=False)
                    self.current_state = "Idle"
                    logger.info("YOLO loaded on CPU fallback.")
                    return
                except Exception:
                    logger.exception("CPU fallback also failed")
            elif self.device == "directml":
                logger.warning("DirectML load failed (%s); trying CPU fallback.", exc)
                try:
                    self.device = "cpu"
                    self._configure_runtime_for_device()
                    os.chdir(model_dir)
                    self.model = YOLO(model_name)
                    self.model.to("cpu")
                    if hasattr(self.model.model, "fuse"):
                        self.model.model.fuse()
                    self.model.predict(
                        np.zeros((640, 640, 3), dtype=np.uint8),
                        device="cpu",
                        imgsz=self._inference_imgsz(1),
                        verbose=False,
                    )
                    self._configure_runtime_for_device(log_profile=False)
                    self.current_state = "Idle"
                    logger.info("YOLO loaded on CPU fallback.")
                    return
                except Exception:
                    logger.exception("CPU fallback also failed")
            logger.exception("Failed to load YOLO model")
            raise
        finally:
            os.chdir(old_cwd)

    # ─────────────────────────────────────────────────────────────────────────
    # Public control API
    # ─────────────────────────────────────────────────────────────────────────

    def set_lane_count(self, lane_count: int | str):
        self.lane_count = normalise_lane_count(lane_count)
        logger.info("Lane mode: %s feed(s).", self.lane_count)

    @staticmethod
    def _cuda_vram_gb() -> float:
        """Return CUDA device memory in GiB, or zero when CUDA is unavailable."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _cuda_free_vram_gb() -> float:
        """Return currently free CUDA memory, or zero if it cannot be read."""
        try:
            if torch.cuda.is_available():
                free_bytes, _ = torch.cuda.mem_get_info()
                return free_bytes / (1024 ** 3)
        except Exception:
            pass
        return 0.0

    def _inference_batch_size(self, lane_count: int) -> int:
        """Choose a safe per-tick batch for the selected accelerator."""
        if lane_count <= 1:
            return 1
        if self.device != "cuda":
            return 1
        vram_gb = self._cuda_vram_gb()
        free_gb = self._cuda_free_vram_gb()
        # A user may have another CUDA application open. Fall back to one
        # image if free memory is already tight, even when total VRAM is large.
        if vram_gb < config.GPU_TINY_VRAM_GB or (free_gb > 0 and free_gb < 0.75):
            return config.GPU_TINY_MULTI_BATCH
        if vram_gb < config.GPU_SMALL_VRAM_GB:
            return config.GPU_SMALL_MULTI_BATCH
        return min(lane_count, config.GPU_LARGE_MULTI_BATCH)

    def _inference_imgsz(self, lane_count: int) -> int:
        """Choose detail/throughput settings without lowering display quality."""
        if self.device == "intel:gpu":
            return config.INTEL_OPENVINO_INFERENCE_IMGSZ
        if self.device == "directml":
            return config.DIRECTML_INFERENCE_IMGSZ
        if self.device != "cpu":
            if self._cuda_vram_gb() < config.GPU_SMALL_VRAM_GB:
                return (
                    config.GPU_SMALL_INFERENCE_IMGSZ_SINGLE
                    if lane_count == 1 else config.GPU_SMALL_INFERENCE_IMGSZ_MULTI
                )
            return (
                config.INFERENCE_IMGSZ
                if lane_count == 1 else config.GPU_LARGE_INFERENCE_IMGSZ_MULTI
            )
        profile = self._cpu_profile or self._select_cpu_profile()
        return profile["single_imgsz"] if lane_count == 1 else profile["multi_imgsz"]

    def get_latest_frame(self):
        """Return the newest published RGB frame without an extra full copy.

        The worker never mutates a frame after publishing it; it replaces the
        reference with a newly-created frame on the next iteration. Consumers
        must treat the returned array as read-only. Avoiding a copy saves a
        full-resolution memory transfer on every GUI refresh.
        """
        with self._frame_lock:
            return self._latest_frame

    def get_latest_frame_if_new(self, previous_sequence: int):
        """Return ``(sequence, frame)`` for a new immutable published frame."""
        with self._frame_lock:
            if self._latest_frame is None or self._frame_sequence == previous_sequence:
                return self._frame_sequence, None
            return self._frame_sequence, self._latest_frame

    def mark_stats_event_processed(self):
        with self._stats_event_lock:
            self._stats_event_pending = False

    def _reset_tracker_state(self):
        """Reset Ultralytics tracker state before a new detection session."""
        # Calls within one session use persist=True for stable IDs. Ultralytics
        # retains those trackers between calls, so reset them when a new video
        # run starts rather than carrying old tracks into the new source.
        predictor = getattr(self.model, "predictor", None)
        if predictor is None:
            return
        for tracker in getattr(predictor, "trackers", ()):
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()
        if hasattr(predictor, "vid_path"):
            predictor.vid_path = [None] * len(getattr(predictor, "vid_path", ()))

    def _queue_stats_update(self):
        with self._stats_event_lock:
            if self._stats_event_pending:
                return
            self._stats_event_pending = True
        self.event_queue.put(("stats_update", None))

    def start(self, source_paths, save_output=True,
              conf=config.DEFAULT_CONFIDENCE, iou=config.DEFAULT_IOU,
              is_camera: bool = False):
        """Launch inference on a list of lane camera paths or camera indices.

        Parameters
        ----------
        source_paths : list[str | int]
            File paths for video files, or integer camera indices (e.g. 0, 1)
            for live webcam/CCTV feeds.
        is_camera : bool
            When True the source is treated as a live feed — no output is
            written and progress is not computed.
        """
        if isinstance(source_paths, (str, int)):
            source_paths = [source_paths]

        if self._worker_thread and self._worker_thread.is_alive():
            logger.warning("Inference thread already running.")
            return
        if self.model is None:
            self.event_queue.put(("error", "Model is not loaded yet."))
            return

        self._stop_event.clear()
        self._pause_event.set()
        self.stats = DetectionStats()
        self._reset_tracker_state()
        self._latest_frame = None
        self._frame_sequence = 0
        self._ambulance_lane_streaks = {}
        self._ambulance_confidence_history = {}
        self._ambulance_missed_frames = {}
        self._ambulance_counts_by_lane = {}
        self._ambulance_proximity_by_lane = {}
        self._dir_tracker.clear_all()
        self._demand_smoother.clear_all()
        self._waiting_tracker.clear_all()
        self._overlay_flash_counter = 0
        self._stable_signal_lane = None
        self._candidate_signal_lane = None
        self._candidate_signal_frames = 0
        with self._stats_event_lock:
            self._stats_event_pending = False

        # Drain leftover events from previous detection sessions
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
                self.event_queue.task_done()
            except queue.Empty:
                break

        self.lane_count = len(source_paths)
        self._is_live_camera = is_camera
        self._worker_thread = threading.Thread(
            target=self._run_inference_loop,
            args=(source_paths, save_output, conf, iou),
            name="InferenceWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def stop(self):
        self._stop_event.set()
        self._pause_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
            logger.info("Inference worker stopped.")
        self.current_state = "Idle"

    def pause(self):
        self._pause_event.clear()
        self.current_state = "Paused"

    def resume(self):
        self._pause_event.set()
        self.current_state = "Processing"

    @property
    def is_busy(self) -> bool:
        return self.current_state in ("Processing", "Paused")

    # ─────────────────────────────────────────────────────────────────────────
    # Colour helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _color_rgb(self, class_name: str) -> tuple[int, int, int]:
        return self._class_colors_rgb.get(class_name.lower(), self._default_color_rgb)

    def _color_bgr(self, class_name: str) -> tuple[int, int, int]:
        return _rgb_to_bgr(self._color_rgb(class_name))

    def color_hex(self, class_name: str) -> str:
        r, g, b = self._color_rgb(class_name)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _passes_confidence(self, class_name: str, conf_val: float, default_conf: float) -> bool:
        thresh = getattr(self, "confidence", default_conf)
        if class_name.lower() == "ambulance":
            amb_thresh = getattr(self, "ambulance_confidence", config.AMBULANCE_CONFIDENCE)
            # Ambulance is intentionally independent of the normal vehicle
            # slider. Emergency priority requires a strong fixed score rather
            # than allowing a low general threshold to create false alarms.
            return conf_val >= amb_thresh
        return conf_val >= thresh

    @staticmethod
    def _ambulance_proximity_score(
        bbox: tuple[int, int, int, int], frame_shape: tuple[int, ...]
    ) -> float:
        """Estimate ambulance closeness from bounding-box perspective cues.

        Without camera calibration, the lower edge of the box and its height
        are useful cues: closer vehicles normally appear lower and larger.
        The normalized result is used only to rank ambulances in a camera
        view, not as a physical distance measurement.
        """
        x1, y1, x2, y2 = bbox
        frame_height, frame_width = frame_shape[:2]
        if frame_height <= 0 or frame_width <= 0:
            return 0.0

        bottom_score = max(0.0, min(1.0, y2 / frame_height))
        box_height = max(0, y2 - y1)
        size_score = max(0.0, min(1.0, box_height / (0.35 * frame_height)))
        return round(0.60 * bottom_score + 0.40 * size_score, 6)

    # ─────────────────────────────────────────────────────────────────────────
    # OpenCV drawing helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_accent_box(self, img, pt1, pt2, color, thickness=2, r=8, d=10):
        x1, y1 = pt1
        x2, y2 = pt2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        h, w = y2 - y1, x2 - x1
        r_adj = min(r, h // 2, w // 2)
        d_adj = min(d, h // 2 - r_adj, w // 2 - r_adj)
        if r_adj <= 0 or d_adj <= 0:
            return
        lw = cv2.LINE_AA
        cv2.line(img, (x1+r_adj, y1), (x1+r_adj+d_adj, y1), color, thickness, lineType=lw)
        cv2.line(img, (x1, y1+r_adj), (x1, y1+r_adj+d_adj), color, thickness, lineType=lw)
        cv2.ellipse(img, (x1+r_adj, y1+r_adj), (r_adj, r_adj), 180, 0, 90, color, thickness, lineType=lw)
        cv2.line(img, (x2-r_adj-d_adj, y1), (x2-r_adj, y1), color, thickness, lineType=lw)
        cv2.line(img, (x2, y1+r_adj), (x2, y1+r_adj+d_adj), color, thickness, lineType=lw)
        cv2.ellipse(img, (x2-r_adj, y1+r_adj), (r_adj, r_adj), 270, 0, 90, color, thickness, lineType=lw)
        cv2.line(img, (x1+r_adj, y2), (x1+r_adj+d_adj, y2), color, thickness, lineType=lw)
        cv2.line(img, (x1, y2-r_adj-d_adj), (x1, y2-r_adj), color, thickness, lineType=lw)
        cv2.ellipse(img, (x1+r_adj, y2-r_adj), (r_adj, r_adj), 90, 0, 90, color, thickness, lineType=lw)
        cv2.line(img, (x2-r_adj-d_adj, y2), (x2-r_adj, y2), color, thickness, lineType=lw)
        cv2.line(img, (x2, y2-r_adj-d_adj), (x2, y2-r_adj), color, thickness, lineType=lw)
        cv2.ellipse(img, (x2-r_adj, y2-r_adj), (r_adj, r_adj), 0, 0, 90, color, thickness, lineType=lw)

    def _draw_lane_badge(self, frame, lane_number, vehicle_count,
                         ambulance=False, signal=False, signal_state="RED",
                         green_secs: int = 0, ambulance_count: int = 0,
                         priority: bool = False):
        """Draw a lane header badge on an independent camera feed."""
        height, width = frame.shape[:2]
        if ambulance:
            color = (0, 30, 200)          # deep red-blue in BGR
            text  = f"LANE {lane_number}  ·  🚨 AMBULANCE  ·  {vehicle_count} vehicles"
            if priority:
                text = f"LANE {lane_number}  |  TOP PRIORITY  |  {max(1, ambulance_count)} ambulance(s)  |  {vehicle_count} vehicles"
        elif signal_state == "GREEN":
            color = (34, 139, 34)         # forest green
            text  = f"LANE {lane_number}  ·  GREEN [{green_secs}s]  ·  {vehicle_count} vehicles"
        elif signal_state == "YIELD":
            color = (0, 140, 255)         # amber-orange
            text  = f"LANE {lane_number}  ·  YIELD  ·  {vehicle_count} vehicles"
        else:
            color = (80, 30, 200)         # dark red
            text  = f"LANE {lane_number}  ·  RED  ·  {vehicle_count} vehicles"

        cv2.rectangle(frame, (0, 0), (width, 34), color, -1)
        cv2.putText(frame, text, (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                    (255, 255, 255), 2, lineType=cv2.LINE_AA)

    def _draw_ambulance_emergency_overlay(
        self,
        frame: np.ndarray,
        lane_number: int,
        direction: str,
        flash: bool,
        priority: bool = False,
        ambulance_count: int = 1,
    ):
        """Draw a prominent emergency overlay on a lane frame.

        • Flashing red border (thick)
        • Semi-transparent red tint when flash=True
        • Large "AMBULANCE DETECTED" text
        • Directional arrow telling other cars which way to move
        """
        h, w = frame.shape[:2]
        t = config.AMBULANCE_OVERLAY_BORDER_THICKNESS

        # ── Flashing border ──────────────────────────────────────────────────
        border_color = (0, 0, 255) if flash else (0, 80, 200)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, t)

        # ── Semi-transparent red tint on flash frames ─────────────────────
        if flash:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

        # ── Central emergency text block ─────────────────────────────────
        cx, cy = w // 2, h // 2

        # Background pill
        box_w, box_h = min(w - 40, 480), 110
        bx1, by1 = cx - box_w // 2, cy - box_h // 2
        bx2, by2 = cx + box_w // 2, cy + box_h // 2
        pill = frame.copy()
        cv2.rectangle(pill, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
        cv2.addWeighted(pill, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)

        # Line 1 — title
        font = cv2.FONT_HERSHEY_DUPLEX
        title = "!! AMBULANCE DETECTED !!"
        ts = cv2.getTextSize(title, font, 0.70, 2)[0]
        cv2.putText(frame, title,
                    (cx - ts[0] // 2, cy - 22),
                    font, 0.70, (0, 0, 255) if flash else (0, 100, 255),
                    2, lineType=cv2.LINE_AA)

        # Line 2 — direction instruction
        if direction == "left":
            dir_text = "<<  MOVE RIGHT — CLEAR LEFT LANE"
            arrow_pts = np.array([
                [bx1 + 20, cy + 22],
                [bx1 + 55, cy + 10],
                [bx1 + 55, cy + 34],
            ], np.int32)
        elif direction == "right":
            dir_text = "CLEAR RIGHT LANE — MOVE LEFT  >>"
            arrow_pts = np.array([
                [bx2 - 20, cy + 22],
                [bx2 - 55, cy + 10],
                [bx2 - 55, cy + 34],
            ], np.int32)
        else:
            dir_text = "YIELD — CLEAR LANE FOR AMBULANCE"
            arrow_pts = None

        ds = cv2.getTextSize(dir_text, font, 0.50, 1)[0]
        cv2.putText(frame, dir_text,
                    (cx - ds[0] // 2, cy + 26),
                    font, 0.50, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        if arrow_pts is not None:
            cv2.fillPoly(frame, [arrow_pts], (0, 80, 255))

        # Bottom strip — lane number
        priority_text = " - TOP PRIORITY" if priority else ""
        strip_text = (
            f"LANE {lane_number} EMERGENCY{priority_text}"
            f" ({max(1, ambulance_count)} ambulance(s))"
        )
        cv2.rectangle(frame, (0, h - 36), (w, h), (0, 0, 200), -1)
        st = cv2.getTextSize(strip_text, font, 0.58, 2)[0]
        cv2.putText(frame, strip_text,
                    (cx - st[0] // 2, h - 12),
                    font, 0.58, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    @staticmethod
    def _draw_traffic_signal_hud(
        frame: np.ndarray,
        signal_state: str,
        green_secs: int,
        vehicle_count: int,
    ):
        """Draw a small traffic-light HUD in the bottom-right corner."""
        h, w = frame.shape[:2]
        pad = 10
        box_w, box_h = 130, 50
        x1 = w - box_w - pad
        y1 = h - box_h - 40   # above the ambulance strip if present
        x2, y2 = x1 + box_w, y1 + box_h

        if signal_state == "EMERGENCY":
            bg, fg, label = (0, 0, 160), (255, 255, 255), "EMERGENCY"
        elif signal_state == "GREEN":
            bg, fg, label = (0, 120, 0), (255, 255, 255), f"GREEN {green_secs}s"
        elif signal_state == "YIELD":
            bg, fg, label = (0, 120, 200), (255, 255, 255), "YIELD"
        else:
            bg, fg, label = (0, 0, 120), (200, 200, 200), "RED"

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), fg, 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        ls = cv2.getTextSize(label, font, 0.45, 1)[0]
        cv2.putText(frame, label,
                    (x1 + (box_w - ls[0]) // 2, y1 + 18),
                    font, 0.45, fg, 1, lineType=cv2.LINE_AA)
        cnt_text = f"{vehicle_count} vehicles"
        cs = cv2.getTextSize(cnt_text, font, 0.38, 1)[0]
        cv2.putText(frame, cnt_text,
                    (x1 + (box_w - cs[0]) // 2, y1 + 36),
                    font, 0.38, fg, 1, lineType=cv2.LINE_AA)

    @staticmethod
    def _compose_grid(
        frames_bgr: list[np.ndarray],
        lane_count: int,
        max_width: int = 0,
        max_height: int = 0,
    ) -> np.ndarray:
        """Build the GUI preview grid, optionally capping its output size.

        The cap is deliberately applied after detection/drawing.  It therefore
        never changes the image supplied to YOLO or the full-resolution frame
        sent to an output writer.
        """
        if lane_count == 1:
            return frames_bgr[0]
        target_h = min(f.shape[0] for f in frames_bgr)
        target_w = min(f.shape[1] for f in frames_bgr)
        columns = 2
        rows = 1 if lane_count == 2 else 2
        full_w, full_h = target_w * columns, target_h * rows
        scale = 1.0
        if max_width > 0:
            scale = min(scale, max_width / full_w)
        if max_height > 0:
            scale = min(scale, max_height / full_h)
        target_w = max(1, int(target_w * scale))
        target_h = max(1, int(target_h * scale))

        cells = []
        for frame in frames_bgr:
            if frame.shape[1] == target_w and frame.shape[0] == target_h:
                cells.append(frame)
            else:
                # INTER_AREA is smooth and efficient for the preview's
                # downscaling; use the former default for an upscaled cell.
                interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                cells.append(cv2.resize(frame, (target_w, target_h), interpolation=interpolation))
        if lane_count == 2:
            return np.hstack(cells)
        top    = np.hstack(cells[:2])
        bottom = np.hstack(cells[2:4])
        return np.vstack([top, bottom])

    # ─────────────────────────────────────────────────────────────────────────
    # Per-lane result processing
    # ─────────────────────────────────────────────────────────────────────────

    def _process_lane_result(
        self, frame, result, lane_number, conf,
        unique_tracks, track_key_offset,
    ):
        """Extract detections, draw bboxes, update ambulance direction tracker."""
        boxes = result.boxes
        detection_rows = boxes.data.detach().cpu().tolist()
        is_tracked = bool(boxes.is_track)

        lane_vehicle_count = 0
        ambulance_seen = False
        ambulance_confidence = 0.0
        ambulance_proximity_scores: list[float] = []
        current_class_counts: dict[str, int] = {}
        ambulance_cx: float | None = None
        nearest_ambulance_proximity = -1.0

        for row in detection_rows:
            x1, y1, x2, y2 = (int(v) for v in row[:4])
            if is_tracked:
                track_id = int(row[4])
                conf_val  = float(row[5])
                cls_id    = int(row[6])
            else:
                track_id = None
                conf_val  = float(row[4])
                cls_id    = int(row[5])

            class_name = self.model.names[cls_id]
            if not self._passes_confidence(class_name, conf_val, conf):
                continue

            display_class = config.VEHICLE_CLASSES.get(class_name.lower(), class_name)
            current_class_counts[display_class] = current_class_counts.get(display_class, 0) + 1
            lane_vehicle_count += 1

            is_ambulance = class_name.lower() == "ambulance"
            if is_ambulance:
                ambulance_seen = True
                ambulance_confidence = max(ambulance_confidence, conf_val)
                proximity = self._ambulance_proximity_score(
                    (x1, y1, x2, y2), frame.shape
                )
                ambulance_proximity_scores.append(proximity)
                # Track the nearest ambulance in this lane for direction
                # inference as well as for the lane-level priority score.
                if proximity >= nearest_ambulance_proximity:
                    nearest_ambulance_proximity = proximity
                    ambulance_cx = (x1 + x2) / 2.0

            if track_id is not None:
                unique_tracks[track_key_offset + track_id] = display_class

            color = self._color_bgr(display_class)
            # CPU previews are commonly scaled down (especially multi-lane),
            # so use a slightly stronger base stroke to keep boxes visible.
            thickness = (
                config.DETECTION_AMBULANCE_BOX_THICKNESS
                if is_ambulance else config.DETECTION_BOX_THICKNESS
            )
            self._draw_accent_box(frame, (x1, y1), (x2, y2), color, thickness=thickness)

            if is_ambulance:
                # Extra bright red outline
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), (0, 0, 255),
                    config.DETECTION_AMBULANCE_BOX_THICKNESS,
                    lineType=cv2.LINE_AA,
                )

            label = (
                f"!! {display_class} #{track_id} {conf_val*100:.0f}% !!"
                if is_ambulance and track_id is not None
                else (
                    f"!! {display_class} {conf_val*100:.0f}% !!"
                    if is_ambulance
                    else (
                        f"{display_class} #{track_id} {conf_val*100:.0f}%"
                        if track_id is not None
                        else f"{display_class} {conf_val*100:.0f}%"
                    )
                )
            )
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX,
                config.DETECTION_LABEL_FONT_SCALE,
                config.DETECTION_LABEL_THICKNESS,
            )
            cv2.rectangle(frame,
                          (x1, max(0, y1 - lh - 6)),
                          (x1 + lw + 6, y1),
                          color, -1)
            cv2.putText(frame, label,
                        (x1 + 3, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        config.DETECTION_LABEL_FONT_SCALE,
                        (255, 255, 255), config.DETECTION_LABEL_THICKNESS,
                        lineType=cv2.LINE_AA)

        # Update direction tracker with ambulance centroid
        if ambulance_seen and ambulance_cx is not None:
            self._dir_tracker.update(lane_number, ambulance_cx)

        return (
            lane_vehicle_count,
            ambulance_seen,
            ambulance_confidence,
            current_class_counts,
            len(detection_rows),
            ambulance_proximity_scores,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Sound alert (Windows-only, non-blocking)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _play_ambulance_beep():
        """Fire a short beep on Windows without blocking inference."""
        if not config.AMBULANCE_SOUND_ALERT:
            return
        try:
            import winsound
            winsound.Beep(config.AMBULANCE_BEEP_HZ, config.AMBULANCE_BEEP_MS)
        except Exception:
            pass   # Non-Windows or winsound unavailable

    # ─────────────────────────────────────────────────────────────────────────
    # Per-lane frame reader (runs inside ThreadPoolExecutor)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _read_frame(cap: cv2.VideoCapture, grayscale: bool, live: bool = False):
        """Read a frame, optionally preferring the newest live-camera frame."""
        if live:
            # Some camera backends ignore CAP_PROP_BUFFERSIZE. Discard a
            # couple of queued frames so slow inference does not accumulate
            # seconds of latency behind the real scene.
            ret = False
            for _ in range(max(1, config.LIVE_CAMERA_BUFFER_DRAIN)):
                ret = cap.grab()
                if not ret:
                    return False, None
            ret, frame = cap.retrieve()
        else:
            ret, frame = cap.read()
        if not ret:
            return False, None
        if grayscale:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray,  cv2.COLOR_GRAY2BGR)
        return True, frame

    # ─────────────────────────────────────────────────────────────────────────
    # Main inference loop
    # ─────────────────────────────────────────────────────────────────────────

    def _run_inference_loop(self, source_paths, save_output, conf, iou):
        self.current_state = "Processing"
        self.event_queue.put(("state_change", "Processing"))

        lane_count = len(source_paths)
        # Pass the lowest threshold that can still survive the final Python
        # filter. This preserves the slider's recall behavior while avoiding
        # unnecessary low-confidence candidates when the user chooses a high
        # threshold, improving FPS on every backend.
        requested_conf = getattr(self, "confidence", conf)
        ambulance_conf = getattr(self, "ambulance_confidence", config.AMBULANCE_CONFIDENCE)
        infer_conf = min(requested_conf, ambulance_conf)
        skip       = max(1, config.FRAME_SKIP_INTERVAL)
        max_skip   = config.CPU_MAX_FRAME_SKIP
        inference_imgsz = self._inference_imgsz(lane_count)
        if self.device == "cpu":
            logger.info(
                "CPU profile: model input %spx for %s feed(s); display/output remain original resolution.",
                inference_imgsz, lane_count,
            )

        # ── Open captures ────────────────────────────────────────────────────
        is_live = getattr(self, "_is_live_camera", False)
        caps: list[cv2.VideoCapture] = []
        for path in source_paths:
            # path can be an int (camera index) or a file path string
            if isinstance(path, int) or (isinstance(path, str) and path.isdigit()):
                cam_idx = int(path)
                # On Windows, cv2.CAP_DSHOW provides fast, low-latency camera initialization
                backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
                cap = cv2.VideoCapture(cam_idx, backend)
                if not cap.isOpened() and backend != cv2.CAP_ANY:
                    cap = cv2.VideoCapture(cam_idx, cv2.CAP_ANY)
                if cap.isOpened():
                    # MJPG keeps USB camera bandwidth and CPU decode overhead
                    # manageable at 720p. Unsupported cameras simply ignore
                    # the request and retain their native format.
                    if os.name == "nt":
                        cap.set(
                            cv2.CAP_PROP_FOURCC,
                            cv2.VideoWriter_fourcc(*"MJPG"),
                        )
                    # Request a practical capture mode and a one-frame buffer.
                    # Backends may reject any of these requests gracefully.
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.LIVE_CAMERA_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.LIVE_CAMERA_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, config.LIVE_CAMERA_FPS)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                cap = cv2.VideoCapture(path)

            if not cap.isOpened():
                for c in caps: c.release()
                self.current_state = "Error"
                label = f"Camera {path}" if isinstance(path, int) else path
                self.event_queue.put(("error", f"Could not open source:\n{label}"))
                return
            caps.append(cap)

        fps_values  = [cap.get(cv2.CAP_PROP_FPS) for cap in caps]
        fps         = next((f for f in fps_values if f and f > 0 and not np.isnan(f)), 30.0)
        # Live cameras report total_frames as 0 or -1; treat that as live (infinite).
        total_frames = 0 if is_live else min(
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) for cap in caps
        )
        widths  = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  for cap in caps]
        heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps]
        self.stats.total_frames = total_frames
        self.stats.resolution   = " + ".join(f"{w}x{h}" for w, h in zip(widths, heights))

        # ── Output writers (one per lane, each on its own thread) ────────────
        writer_queues:  list[queue.Queue] = []
        writer_threads: list[threading.Thread] = []
        writer_failed   = threading.Event()
        writer_errors:  list[Exception] = []

        if save_output and not is_live:
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            for li, path in enumerate(source_paths, start=1):
                # Use a safe string label for the filename
                raw_name = str(path)
                name_part = "camera" if isinstance(path, int) or raw_name.isdigit() else os.path.splitext(os.path.basename(raw_name))[0]
                suffix = "" if lane_count == 1 else f"_lane{li}"
                out_fp = os.path.join(
                    config.OUTPUT_DIR, f"{name_part}{suffix}_{run_id}_processed.mp4"
                )
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    out_fp, fourcc, fps, (widths[li - 1], heights[li - 1])
                )
                if not writer.isOpened():
                    for c in caps: c.release()
                    self.current_state = "Error"
                    self.event_queue.put(("error", f"Could not create output:\n{out_fp}"))
                    return
                logger.info(f"Lane {li} output → {out_fp}")

                wq: queue.Queue = queue.Queue(maxsize=config.OUTPUT_WRITER_QUEUE_SIZE)

                def _write_loop(out=writer, q=wq, failed=writer_failed, errs=writer_errors):
                    try:
                        while True:
                            frm = q.get()
                            try:
                                if frm is None:
                                    return
                                out.write(frm)
                            finally:
                                q.task_done()
                    except Exception as exc:
                        errs.append(exc)
                        failed.set()
                        logger.exception("Output writer failed")
                    finally:
                        out.release()

                wt = threading.Thread(
                    target=_write_loop, name=f"OutWriter-L{li}", daemon=True
                )
                wt.start()
                writer_queues.append(wq)
                writer_threads.append(wt)

        def _enqueue_output(li_1based: int, frm: np.ndarray):
            if not writer_queues:
                return
            wq = writer_queues[li_1based - 1]
            while not self._stop_event.is_set():
                if writer_failed.is_set():
                    raise RuntimeError(f"Writer failed: {writer_errors[0]}")
                try:
                    wq.put(frm, timeout=0.1)
                    return
                except queue.Full:
                    continue

        # ── Main loop ────────────────────────────────────────────────────────
        unique_tracks: dict[int, str] = {}
        frame_idx       = 0
        raw_frame_idx   = 0           # every-frame counter used for skip logic
        start_time      = time.time()
        fps_window:  list[float] = []
        _prev_ambulance_confirmed: set[int] = set()
        _beep_thread: threading.Thread | None = None
        # A one-feed run has no parallel work to schedule. Reading directly
        # avoids creating/submitting to a worker pool on every frame. Multi-feed
        # runs retain parallel capture so one slow source does not serialize
        # the others.
        reader_pool = (
            ThreadPoolExecutor(max_workers=lane_count)
            if lane_count > 1 else None
        )
        signal_lane: int | None = self._stable_signal_lane

        try:
            while not self._stop_event.is_set():
                # ── Pause handling ───────────────────────────────────────────
                if not self._pause_event.is_set():
                    self.current_state = "Paused"
                    self.event_queue.put(("state_change", "Paused"))
                    while not self._pause_event.is_set() and not self._stop_event.is_set():
                        time.sleep(0.05)
                    if self._stop_event.is_set():
                        break
                    self.current_state = "Processing"
                    self.event_queue.put(("state_change", "Processing"))

                frame_start = time.perf_counter()

                # ── Parallel frame reading via ThreadPoolExecutor ────────────
                if lane_count == 1:
                    ret, frm = self._read_frame(
                        caps[0], self.grayscale_mode, is_live
                    )
                    ordered = {0: frm if ret else None}
                else:
                    futures = {
                        reader_pool.submit(
                            self._read_frame, cap, self.grayscale_mode, is_live
                        ): idx
                        for idx, cap in enumerate(caps)
                    }
                    ordered = {}
                    for fut in as_completed(futures):
                        idx = futures[fut]
                        ret, frm = fut.result()
                        ordered[idx] = frm if ret else None

                raw_frame_idx += 1

                # Collect frames in lane order; stop if any lane ended
                frames: list[np.ndarray] = []
                for i in range(lane_count):
                    frm = ordered.get(i)
                    if frm is None:
                        frames = []
                        break
                    frames.append(frm)

                if not frames:
                    break

                # ── Frame-skip (still advances caps, just skips inference) ──
                if skip > 1 and (raw_frame_idx % skip != 0):
                    continue

                frame_idx += 1

                # ── Batched GPU inference ────────────────────────────────────
                inference_device = "cpu" if self.device == "directml" else self.device
                track_kwargs = {
                    # Ultralytics accepts an ndarray directly. Avoiding a
                    # one-item list removes small per-frame source-normalising
                    # overhead while leaving multi-feed batching unchanged.
                    "source": frames[0] if len(frames) == 1 else frames,
                    "persist": True,
                    "device": inference_device,
                    "conf": infer_conf,
                    "iou": iou,
                    "imgsz": inference_imgsz,
                    "max_det": config.MAX_DETECTIONS,
                    "verbose": False,
                }
                if self.device == "cuda":
                    track_kwargs["half"] = True

                # Keep the v8m workload within the accelerator's memory. A
                # GTX 1650 has 4 GB, so it uses two 640px images per call;
                # very small cards remain at one image, and larger cards can
                # batch more. Chunking also protects OpenVINO/DirectML
                # providers that use fixed batch shapes.
                batch_size = self._inference_batch_size(len(frames))
                if len(frames) > batch_size:
                    results = []
                    for start in range(0, len(frames), batch_size):
                        chunk = frames[start:start + batch_size]
                        chunk_kwargs = dict(track_kwargs)
                        chunk_kwargs["source"] = chunk[0] if len(chunk) == 1 else chunk
                        res = self.model.track(**chunk_kwargs)
                        if isinstance(res, list):
                            results.extend(res)
                        else:
                            results.append(res)
                else:
                    results = self.model.track(**track_kwargs)
                    if not isinstance(results, list):
                        results = [results]

                inf_time = max(
                    (r.speed.get("inference", 0.0) for r in results),
                    default=0.0,
                )

                # ── Per-lane processing ───────────────────────────────────────
                lane_vehicle_counts: dict[int, int] = {l: 0 for l in range(1, lane_count + 1)}
                ambulance_lanes:     set[int]       = set()
                ambulance_confidences: dict[int, float] = {l: 0.0 for l in range(1, lane_count + 1)}
                ambulance_counts_by_lane: dict[int, int] = {l: 0 for l in range(1, lane_count + 1)}
                ambulance_proximity_by_lane: dict[int, float] = {l: 0.0 for l in range(1, lane_count + 1)}
                current_class_counts: dict[str, int] = {}
                vehicle_types_by_lane: dict[int, list[str]] = {
                    lane: [] for lane in range(1, lane_count + 1)
                }
                raw_det_total = 0
                processed_frames: list[np.ndarray] = []

                for li, (frm, result) in enumerate(zip(frames, results), start=1):
                    offset = li * 100_000
                    count, amb_seen, amb_conf, lane_cls, raw_cnt, amb_proximity_scores = self._process_lane_result(
                        frm, result, li, conf, unique_tracks, offset
                    )
                    lane_vehicle_counts[li] = count
                    for cls_name, cls_count in lane_cls.items():
                        vehicle_types_by_lane[li].extend(
                            [cls_name] * max(0, int(cls_count))
                        )
                    raw_det_total += raw_cnt
                    for cls, n in lane_cls.items():
                        current_class_counts[cls] = current_class_counts.get(cls, 0) + n
                    if amb_seen:
                        ambulance_lanes.add(li)
                        ambulance_confidences[li] = amb_conf
                        ambulance_counts_by_lane[li] = len(amb_proximity_scores)
                        ambulance_proximity_by_lane[li] = max(amb_proximity_scores, default=0.0)
                    processed_frames.append(frm)

                # ── Ambulance streak counting ────────────────────────────────
                new_streaks: dict[int, int] = {}
                new_misses: dict[int, int] = {}
                new_ambulance_counts: dict[int, int] = {}
                new_ambulance_proximity: dict[int, float] = {}
                confirmed_amb_lanes: set[int] = set()
                for lane in range(1, lane_count + 1):
                    candidate_conf = ambulance_confidences[lane]
                    if candidate_conf >= config.AMBULANCE_CONFIDENCE:
                        history = self._ambulance_confidence_history.setdefault(
                            lane,
                            collections.deque(maxlen=config.AMBULANCE_CONFIRMATION_HISTORY),
                        )
                        history.append(candidate_conf)
                        new_streaks[lane] = self._ambulance_lane_streaks.get(lane, 0) + 1
                        new_misses[lane] = 0
                        new_ambulance_counts[lane] = ambulance_counts_by_lane[lane]
                        new_ambulance_proximity[lane] = ambulance_proximity_by_lane[lane]
                    else:
                        prior_misses = self._ambulance_missed_frames.get(lane, 0) + 1
                        if (
                            lane in self._ambulance_lane_streaks
                            and prior_misses <= config.AMBULANCE_LOST_GRACE_FRAMES
                        ):
                            new_streaks[lane] = self._ambulance_lane_streaks[lane]
                            new_misses[lane] = prior_misses
                            new_ambulance_counts[lane] = self._ambulance_counts_by_lane.get(lane, 0)
                            new_ambulance_proximity[lane] = self._ambulance_proximity_by_lane.get(lane, 0.0)
                        else:
                            self._ambulance_confidence_history.pop(lane, None)
                            continue

                    history = self._ambulance_confidence_history[lane]
                    if (
                        new_streaks[lane] >= config.AMBULANCE_CONFIRMATION_FRAMES
                        and max(history) >= config.AMBULANCE_CONFIRMATION_CONFIDENCE
                        and sum(history) / len(history) >= config.AMBULANCE_CONFIRMATION_MIN_AVG_CONFIDENCE
                    ):
                        confirmed_amb_lanes.add(lane)

                self._ambulance_lane_streaks = new_streaks
                self._ambulance_missed_frames = new_misses
                self._ambulance_counts_by_lane = new_ambulance_counts
                self._ambulance_proximity_by_lane = new_ambulance_proximity

                # ── Sound alert on new ambulance confirmation ────────────────
                newly_confirmed = confirmed_amb_lanes - _prev_ambulance_confirmed
                if newly_confirmed and (
                    _beep_thread is None or not _beep_thread.is_alive()
                ):
                    _beep_thread = threading.Thread(
                        target=self._play_ambulance_beep, daemon=True
                    )
                    _beep_thread.start()
                _prev_ambulance_confirmed = set(confirmed_amb_lanes)

                # ── Direction inference ──────────────────────────────────────
                amb_directions: dict[int, str] = {}
                for lane in confirmed_amb_lanes:
                    amb_directions[lane] = self._dir_tracker.direction(lane)
                # Clear direction history for lanes that no longer have ambulance
                for lane in list(self._dir_tracker._history.keys()):
                    if lane not in confirmed_amb_lanes:
                        self._dir_tracker.clear_lane(lane)

                # ── Signal timers ────────────────────────────────────────────
                # Weight larger vehicles more heavily, then smooth the demand
                # so one noisy frame cannot immediately change the advice.
                traffic_demand = self._demand_smoother.update(
                    calculate_weighted_demand(vehicle_types_by_lane)
                )
                waiting_times = self._waiting_tracker.update(
                    lane_vehicle_counts,
                    signal_lane,
                    elapsed_seconds=skip / max(fps, 1.0),
                )
                green_durations = compute_green_duration(
                    lane_vehicle_counts,
                    traffic_demand=traffic_demand,
                )
                requested_signal_lane, signal_advice = traffic_signal_advice(
                    lane_vehicle_counts,
                    confirmed_amb_lanes,
                    green_durations,
                    traffic_demand=traffic_demand,
                    waiting_times=waiting_times,
                    ambulance_counts=self._ambulance_counts_by_lane,
                    ambulance_proximity=self._ambulance_proximity_by_lane,
                )

                # Emergency priority takes effect immediately. Normal lane
                # changes require a sustained busiest-lane signal to stop the
                # recommendation flickering between similar traffic counts.
                if confirmed_amb_lanes or requested_signal_lane is None:
                    signal_lane = requested_signal_lane
                    self._candidate_signal_lane = None
                    self._candidate_signal_frames = 0
                elif self._stable_signal_lane is None:
                    signal_lane = requested_signal_lane
                    self._stable_signal_lane = signal_lane
                elif requested_signal_lane == self._stable_signal_lane:
                    signal_lane = self._stable_signal_lane
                    self._candidate_signal_lane = None
                    self._candidate_signal_frames = 0
                else:
                    if requested_signal_lane == self._candidate_signal_lane:
                        self._candidate_signal_frames += 1
                    else:
                        self._candidate_signal_lane = requested_signal_lane
                        self._candidate_signal_frames = 1
                    if self._candidate_signal_frames >= config.SIGNAL_HOLD_MIN_FRAMES:
                        self._stable_signal_lane = requested_signal_lane
                        self._candidate_signal_lane = None
                        self._candidate_signal_frames = 0
                    signal_lane = self._stable_signal_lane
                    if signal_lane != requested_signal_lane:
                        signal_advice = (
                            f"Suggested GREEN: Lane {signal_lane} held stable "
                            f"while Lane {requested_signal_lane} is confirmed"
                        )
                signal_states = build_lane_signal_states(
                    lane_vehicle_counts, signal_lane, confirmed_amb_lanes
                )
                priority_ambulance_lane = (
                    signal_lane if confirmed_amb_lanes else None
                )

                emergency_advice = (
                    f"🚨 EMERGENCY: Clear Lane{'s' if len(confirmed_amb_lanes) > 1 else ''} "
                    f"{', '.join(map(str, sorted(confirmed_amb_lanes)))} — Ambulance approaching"
                    if confirmed_amb_lanes else "No ambulance detected"
                )

                # ── Flash toggle ─────────────────────────────────────────────
                if confirmed_amb_lanes:
                    priority_count = self._ambulance_counts_by_lane.get(
                        priority_ambulance_lane, 0
                    )
                    emergency_advice = (
                        f"EMERGENCY: TOP PRIORITY Lane {priority_ambulance_lane}; "
                        f"{priority_count} ambulance(s) in priority lane; "
                        f"clear lanes {', '.join(map(str, sorted(confirmed_amb_lanes)))}"
                    )

                self._overlay_flash_counter += 1
                flash_on = (self._overlay_flash_counter % 6) < 3   # ~2.5 Hz at 15 fps

                # ── Draw overlays on each lane frame ─────────────────────────
                for li, frm in enumerate(processed_frames, start=1):
                    state       = signal_states.get(li, "RED")
                    g_dur       = green_durations.get(li, 0)
                    is_amb_lane = li in confirmed_amb_lanes
                    is_priority_amb_lane = li == priority_ambulance_lane
                    ambulance_count = self._ambulance_counts_by_lane.get(li, 0)

                    if is_amb_lane:
                        direction = amb_directions.get(li, "unknown")
                        self._draw_ambulance_emergency_overlay(
                            frm, li, direction, flash_on,
                            priority=is_priority_amb_lane,
                            ambulance_count=ambulance_count,
                        )

                    self._draw_lane_badge(
                        frm, li, lane_vehicle_counts.get(li, 0),
                        ambulance=is_amb_lane,
                        ambulance_count=ambulance_count,
                        priority=is_priority_amb_lane,
                        signal=state == "GREEN",
                        signal_state=state,
                        green_secs=g_dur,
                    )

                    if not is_amb_lane:
                        self._draw_traffic_signal_hud(frm, state, g_dur, lane_vehicle_counts.get(li, 0))

                    if save_output:
                        _enqueue_output(li, frm.copy())

                # ── Compose grid and push to GUI ─────────────────────────────
                preview_max_width = 0
                preview_max_height = 0
                if self.device == "cpu" and lane_count > 1:
                    profile = self._cpu_profile or self._select_cpu_profile()
                    preview_max_width = profile["preview_width"]
                    preview_max_height = profile["preview_height"]
                composite = self._compose_grid(
                    processed_frames,
                    lane_count,
                    max_width=preview_max_width,
                    max_height=preview_max_height,
                )
                rgb_frame = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)

                with self._frame_lock:
                    self._latest_frame   = rgb_frame
                    self._frame_sequence += 1

                # ── FPS and stats ────────────────────────────────────────────
                frame_end = time.perf_counter()
                fps_window.append(1.0 / max(frame_end - frame_start, 1e-6))
                if len(fps_window) > 30:
                    fps_window.pop(0)

                # When enabled, adapt skipping gently toward the requested
                # processed-frame rate. A fixed interval remains the floor.
                current_fps = sum(fps_window) / len(fps_window)
                target_fps = config.CPU_ADAPTIVE_FPS_TARGET if self.device == "cpu" else 0
                if target_fps > 0 and len(fps_window) >= 10:
                    if current_fps < target_fps * 0.85 and skip < max_skip:
                        skip += 1
                    elif current_fps > target_fps * 1.20 and skip > config.FRAME_SKIP_INTERVAL:
                        skip -= 1

                cumulative_class_counts: dict[str, int] = {}
                for cls_name in unique_tracks.values():
                    cumulative_class_counts[cls_name] = cumulative_class_counts.get(cls_name, 0) + 1

                s = self.stats
                s.fps                  = current_fps
                s.inference_time_ms    = inf_time
                s.frame_number         = frame_idx
                s.elapsed_time         = time.time() - start_time
                # Live camera feeds have no total frame count; keep progress at 0
                s.progress_percent     = (raw_frame_idx / total_frames * 100) if (total_frames > 0 and not is_live) else 0.0
                s.current_count        = sum(lane_vehicle_counts.values())
                s.total_detected       = len(unique_tracks)
                s.class_counts_current = current_class_counts
                s.class_counts_total   = cumulative_class_counts
                s.lane_count           = lane_count
                s.lane_vehicle_counts  = lane_vehicle_counts
                s.ambulance_lanes      = sorted(confirmed_amb_lanes)
                s.ambulance_counts_by_lane = {
                    lane: self._ambulance_counts_by_lane.get(lane, 0)
                    for lane in confirmed_amb_lanes
                }
                s.ambulance_proximity_by_lane = {
                    lane: self._ambulance_proximity_by_lane.get(lane, 0.0)
                    for lane in confirmed_amb_lanes
                }
                s.priority_ambulance_lane = priority_ambulance_lane
                s.ambulance_directions = amb_directions
                s.signal_lane          = signal_lane
                s.signal_advice        = signal_advice
                s.emergency_advice     = emergency_advice
                s.green_durations      = green_durations
                s.lane_signal_states   = signal_states
                s.lane_traffic_demand  = traffic_demand
                s.lane_waiting_times    = waiting_times
                self._queue_stats_update()

        except Exception as exc:
            logger.error("Inference exception: %s", exc, exc_info=True)
            self.current_state = "Error"
            self.event_queue.put(("error", f"Processing exception:\n{exc}"))
        finally:
            try:
                if reader_pool is not None:
                    reader_pool.shutdown(wait=True)
            except Exception:
                pass

            for cap in caps:
                try:
                    cap.release()
                except Exception:
                    pass
            for wq, wt in zip(writer_queues, writer_threads):
                if wt.is_alive() and not writer_failed.is_set():
                    while wt.is_alive():
                        try:
                            wq.put(None, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                wt.join(timeout=10.0)

            if self.current_state != "Error":
                if is_live:
                    self.current_state = "Camera Stopped"
                    self.stats.progress_percent = 0.0
                    self.event_queue.put(("state_change", "Camera Stopped"))
                    logger.info("Camera feed ended.")
                else:
                    self.current_state = "Finished"
                    self.stats.progress_percent = 100.0
                    self.event_queue.put(("state_change", "Finished"))
                    logger.info("Video processing completed successfully.")
