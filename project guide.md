# Nirbhighne Project Guide

This is the short quick-start guide. `PROJECT_GUIDE.txt` contains the deeper
technical explanation, including the top-level folder/file responsibilities,
the source-to-EXE lifecycle, and the phase-by-phase behavior of
`build_exe.bat`.

## Purpose

Nirbhighne is a Windows desktop prototype for vehicle detection, tracking,
traffic-density summaries, ambulance priority, and lane signal advice. It is a
decision-support demonstration; it is not certified traffic-signal control.

## Main flow

1. `main.py` starts the splash screen and launches the CustomTkinter GUI.
2. `gui.py` owns source selection, lane assignment, camera controls, display,
   statistics, and evaluation dialogs.
3. `detector.py` reads each source in parallel, runs YOLO tracking, draws
   overlays, updates counts, and publishes the newest frame to the GUI.
4. `config.py` centralizes model priority, confidence, lane, resolution,
   memory, overlay, and evaluation settings.
5. `evaluator.py` runs Ultralytics validation and writes metrics plus plots.

## Source and lane UX

The supported lane modes are 1, 2, and 4 independent feeds. They are separate
camera/video inputs, not geometric lanes inside one image. In 1-lane mode, the
unified Video Source area contains drag-and-drop, the YouTube URL field, Browse
Video, and YT Detect. In 2/4-lane mode, that top area is replaced by one source
card per lane; each card has Browse and YT Detect, which makes source-to-lane
assignment explicit. A selected file can be replaced by browsing again.

The Camera button opens local OpenCV camera indices. The `CCTV Camera` control
is the network-camera entry point in the interface.

The confidence slider is shared by video and camera detection for normal
vehicle classes. Ambulance is the deliberate exception: it always requires a
fixed 60% confidence, independent of the slider, and still needs confirmation
across multiple processed frames before emergency priority is shown.

## Hardware behavior

Run `python setup_environment.py` once on every new development PC. The script
uses `nvidia-smi` or the Windows video-adapter list as a fallback to choose the
CUDA PyTorch wheel for NVIDIA systems and the CPU PyTorch wheel otherwise. The
live detector then prefers NVIDIA CUDA, Intel OpenVINO, Windows DirectML, and
CPU fallback in that order.

The 4 GB GTX 1650 profile uses 960px single-feed inference. In 2/4-feed mode
it uses a 640px internal input and two lanes per CUDA call to reduce scheduling
overhead without exceeding the card's memory budget. Very small CUDA cards
remain at one lane per call, while larger cards can batch multiple lanes. CPU
and iGPU profiles also use smaller internal inputs, while the displayed/saved
frames keep the camera's original resolution.

For live cameras, OpenCV requests 1280x720 at 30 FPS, asks Windows cameras for
MJPG when supported, and uses a one-frame buffer. The reader discards a small
number of stale frames when inference is slower, which favors current-scene
latency over processing every historical frame. CPU/OpenCV worker counts are
also limited so host decoding does not compete excessively with Torch,
OpenVINO, DirectML, or CUDA inference.

## Model policy

`config.MODEL_PATH` prioritizes `weights/merged_v8m_from_s.pt`, then
`weights/best.pt`, then the other v8m fallbacks. The merged checkpoint keeps
the v8m architecture and incorporates compatible tensors from the local v8s
checkpoint. Because v8s and v8m have different internal shapes, this is not a
replacement for v8m fine-tuning or knowledge distillation.

Detection quality depends on training labels, camera angle, object size,
occlusion, blur, confidence, input size, and available compute. Hardware can
cause missed vehicles through lower resolution, dropped frames, or lag, but it
is only one part of the accuracy problem.

## Evaluation

The standard evaluator resolves dataset paths to absolute paths, supports the
portable validation manifest, and uses conservative FP32 validation settings
for small GPUs. Results are written to `evaluation_results/metrics.txt`,
`evaluation_results/metrics.json`, and `evaluation_results/plots/`.

The included portable validation split is a small smoke/benchmark set. A
larger representative test set should be used before claiming production
accuracy.

## Packaging and transfer

For a source-code transfer, create a virtual environment and run
`setup_environment.py`. For a non-Python target, run `build_exe.bat` and copy
the complete `dist/Nirbhighne/` folder. The build includes the selected model
and portable validation data. The target PC still needs a compatible NVIDIA
driver for CUDA; otherwise the application falls back to another available
provider.

The executable is `dist/Nirbhighne/Nirbhighne.exe`. Copy the complete folder,
including its `weights/`, `evaluation_data/`, and runtime-library folders; do
not copy only the EXE. The full training dataset, `venv/`, `build/`, and
research artefacts are not required on the target PC.

## YouTube

`youtube.py` downloads up to 720p, uses a safe video-ID filename, tries browser
cookies for restricted streams, and selects separate streams only when FFmpeg
is available. Without FFmpeg it requests a progressive stream that can be
saved directly. Restricted, private, age-gated, or unavailable videos can
still fail because yt-dlp cannot bypass the video's access policy.
