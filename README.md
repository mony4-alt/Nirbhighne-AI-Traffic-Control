# Nirbhighne: AI-Driven Dynamic Traffic Signal Controlling System

A Windows desktop prototype built with Python and CustomTkinter for real-time vehicle detection, tracking, and analytics using YOLOv8.

## Features
- **Flexible Stream Inputs**: Load local video files, drag and drop videos directly into the interface, or paste YouTube URLs.
- **Background Multi-Threaded Inference**: High-frame-rate YOLOv8 detection utilizing CUDA GPU acceleration (if present) or CPU, with an asynchronous thread-safe architecture that never freezes the GUI.
- **Accurate Object Tracking & Counting**: Uses the built-in tracking capabilities of YOLOv8 to assign unique track IDs to individual vehicles. Reports:
  - Total unique vehicles detected during the session.
  - Active vehicle count in the current frame.
  - Dynamic class-wise breakdown (e.g., Cars, Buses, Trucks, Motorcycles, Bicycles, CNG Ricksaws, etc.) for both active frames and cumulative totals.
- **High-Tech Sci-Fi Visuals**: Custom-rendered bounding boxes with rounded accent corners (camera-focus style) and dynamic color labels tailored per vehicle class.
- **Full Video Controls**: Start, Pause, Resume, and Stop controls, along with double-click for Borderless Fullscreen display mode.
- **Processed File Output**: Save analyzed video recordings to the `output/` directory in standard MP4 formats.
- **Model Evaluation Suite**: Built-in "Evaluate Model" feature using Ultralytics' `model.val()` validation engine to calculate Precision, Recall, mAP@0.5, mAP@0.5:0.95, and F1-score with live GUI progress indicator and automatic export of `metrics.txt`, `metrics.json`, confusion matrices, and PR curves to `evaluation_results/`.
- **Splash Screen Loader**: Startup loader displaying initialization stages and hardware diagnostics.

---

## Folder Structure
```
VehicleDetector/
├── main.py                # Entrypoint, boots splash screen and coordinates window lifecycle
├── gui.py                  # CustomTkinter interface, layout, and event pumps
├── detector.py             # Inference worker, custom camera corners box drawer, metrics calculations
├── evaluator.py            # Validation runner, metrics extraction, and evaluation exporter
├── evaluation_dataset.py   # Portable validation dataset preparation
├── emergency_overlay.py    # Emergency banner and signal-timer widgets
├── youtube.py              # Background yt-dlp downloader with progress tracking and abort callbacks
├── utils.py                # Hardware querying (GPU info), rotating file logger, and text format helpers
├── config.py               # Application configurations, directory builders, default values
├── theme.py                # UI custom colors, fonts, and corner radius tokens
├── requirements.txt        # Hardware-independent PIP dependencies
├── requirements-cuda.txt   # Matched NVIDIA CUDA PyTorch runtime
├── requirements-cpu.txt    # CPU PyTorch runtime for non-NVIDIA PCs
├── setup_environment.py    # Hardware detection and dependency installer
├── build_exe.bat           # PyInstaller packaging script
├── VehicleDetector.spec    # PyInstaller build specification
├── README.md               # Documentation
├── MODEL_CARD.md           # Model classes, provenance record, and latest metrics
├── PROJECT_GUIDE.txt       # Detailed architecture, packaging, and presentation guide
├── project guide.md        # Short project quick-start guide
├── weights/                # Preferred merged v8m checkpoint and documented fallbacks
├── evaluation_results/     # Generated evaluation reports (metrics.txt, metrics.json, plots)
├── evaluation_data/        # Portable validation copy created for the EXE/evaluation
├── robustness_eval/        # Robustness evaluation script suite (grayscale & lowres experiments)
├── tests/                  # Regression tests for pure application logic
├── scratch/                # One-off research/comparison scripts; not runtime code
├── runs/                   # Training outputs and experiment artefacts; not runtime code
├── data.yml & utility file/ # Dataset manifests, validation assets, and utilities
├── members code/           # Team notebooks and collaboration material
├── assets/                 # Image assets
├── icons/                  # User interface icons
├── logs/                   # Log output files (detector.log)
└── output/                 # Processed video output directory
```

The Python files are the application source, `weights/` and `assets/` are
runtime content, and `evaluation_results/`, `evaluation_data/`, `logs/`,
`output/`, `build/`, and `dist/` are generated or delivery folders. `venv/`
is a machine-local environment and should not be copied between PCs. The
training dataset is kept outside the normal runtime path so the application
and EXE do not carry unnecessary data.

---

## Installation & Setup

### Prerequisites
- **OS**: Windows 10 or 11
- **Python**: Version 3.11 or 3.12 (recommended)
- **NVIDIA GPU** (optional for CUDA acceleration): An up-to-date NVIDIA driver. The required CUDA runtime is installed with PyTorch.

### Steps
1. **Clone/Copy the project directory** to your computer.
2. **Open a Command Prompt or PowerShell** inside the `VehicleDetector` directory.
3. **Create a virtual environment (optional but recommended)**:
   ```cmd
   py -3.12 -m venv venv
   call venv\Scripts\activate
   ```
4. **Detect hardware and install the dependencies**:
   ```cmd
   python setup_environment.py
   ```
   The setup script detects NVIDIA with `nvidia-smi` and, on Windows, the video-adapter list as a fallback. It installs the matched CUDA PyTorch runtime only when NVIDIA is present, and installs CPU PyTorch otherwise. CPU PyTorch is also used for Intel/AMD iGPU machines while the live detector can use OpenVINO or DirectML when available. To inspect hardware without installing anything:
   ```cmd
   python setup_environment.py --check-only
   ```
   Confirm the selected runtime with:
   ```cmd
   python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU/iGPU runtime')"
   ```

5. **Place a compatible trained YOLOv8 model** in the `weights/` directory.
   - Preferred file: `VehicleDetector/weights/merged_v8m_from_s.pt`
   - Fallbacks include `weights/best.pt`, `weights/final_best_v8m.pt`, and the
     root-level `final_best_v8m.pt`; `config.MODEL_PATH` selects the first
     compatible candidate that exists.

---

## Usage

### Running as a Python script:
Activate your virtual environment and execute:
```cmd
python main.py
```

### Camera feeds, ambulance priority, and traffic advice

Use the **Lane & Traffic** dropdown to select **1 feed**, **2 feeds**, or
**4 feeds** before starting detection. Each selected lane represents one
independent camera/video feed. The application reads feeds in parallel, then
chooses a safe inference size and batch for the detected hardware. On a 4 GB
GTX 1650, four-lane v8m inference uses two 640px images per CUDA call to
reduce scheduling overhead while keeping VRAM usage safe; very small cards
remain at one image per call. The displayed and saved frames remain at their
original resolution.

In 1-lane mode the sidebar shows one unified source area: drop a video, paste a
YouTube URL, then use **Browse Video** or **YT Detect**. In 2/4-lane mode that
top area is replaced by separate lane cards. Each card has its own **Browse**
and **YT Detect** controls, so every source is assigned directly to the
intended lane. The **CCTV Camera** button sits beside the local **Camera**
control as the network-camera entry point.

The **Min Confidence** slider applies to normal vehicle detection; ambulance is
the deliberate exception and always requires a fixed 60% confidence regardless
of the slider. The normal threshold applies to local/YouTube video and
live camera detection. Around 30–40% is a practical starting range for distant
or partly occluded traffic; higher values reduce false positives but can hide
real vehicles.

For lower live-camera latency, capture requests 1280x720 at 30 FPS, requests
MJPG from Windows cameras when supported, keeps a one-frame buffer, discards a
small stale-frame backlog when inference falls behind, and limits GUI redraw
work to 30 FPS. This favors showing the current scene over displaying every
old frame.

- The video overlay and right panel show the live count for every selected
  camera feed. The interface calls these feeds "lanes" because each feed is
  intended to represent one traffic approach.
- If the trained model identifies an `ambulance` in a feed for the configured
  confirmation frames, the application marks that approach as an **emergency
  route** and recommends that other approaches yield.
- Otherwise, it recommends a green signal for the approach with the highest
  current vehicle count. Normal recommendation changes are held briefly to
  avoid flickering between similar counts.

These are visual decision-support recommendations only; they do not control a
real traffic signal and are not safe or certified for operational signal
control. The output can be affected by detection confidence, tracker ID changes,
camera placement, occlusion, and false ambulance detections. A real deployment
would require calibrated road zones, redundant sensors, safety interlocks,
field validation, and traffic-authority approval.

### Portable EXE evaluation

`build_exe.bat` builds a portable folder rather than a single standalone file.
It includes the model, CUDA runtime files, and a prepared copy of the 64-image
validation split in `evaluation_data/`. HEIC images are converted to JPEG while
building so standard evaluation includes every validation image. In the EXE,
choose **Evaluate Model → Standard validation**; the included `data.yaml` is
selected automatically. Results, including metrics, PR curves, and confusion
matrices, are written to `evaluation_results/` beside the EXE.

The bundle intentionally excludes the 1.72 GB training set. To evaluate a
different dataset, select that dataset's `data.yaml` using **Browse**.

### Hardware compatibility

The same source project runs on NVIDIA GPU and CPU-only Windows PCs. Run
`setup_environment.py` on each new PC; do not copy the NVIDIA CUDA Python
environment onto an iGPU-only machine. The live detector uses CUDA when
PyTorch can access NVIDIA, OpenVINO for supported Intel GPUs, DirectML for
supported Windows GPUs, and CPU as the final fallback. The displayed and
saved video remain at their original resolution while the model uses a
hardware-appropriate internal input size. Bounding boxes use a thicker stroke
so they remain visible in the four-lane preview.

### YouTube downloads

The downloader limits streams to 720p, uses browser cookies when available,
and detects FFmpeg. With FFmpeg it merges separate streams to MP4; without
FFmpeg it selects a progressive stream that can be saved directly. Install
FFmpeg and place it on `PATH` for the best format availability.

### Model weights and accuracy

The latest local trained v8s checkpoint is
`data.yml & utility file/final_best_v8s.pt`. The application target is the
larger `weights/merged_v8m_from_s.pt` when present, followed by `weights/best.pt`.
The merged file is still a v8m model, but only same-shaped tensors can be
copied from v8s into v8m; it is not equivalent to retraining v8m on the v8s
data. A genuine full transfer requires v8m fine-tuning or knowledge
distillation using the training images.

Hardware can cause missed vehicles indirectly through low input resolution,
frame skipping, dropped frames, or memory pressure. It is not the only cause:
small distant vehicles, occlusion, motion blur, camera angle, incomplete
labels, and confidence thresholds also matter. The default confidence was
lowered to 0.30 to improve recall, but lowering it further can increase false
positives.
This is a speed/very-small-object trade-off; it does not make the player look
pixelated. CPU frame skipping remains off by default to preserve tracking
continuity.

### GUI Guide:
1. **Model Validation**: The application checks the documented model candidates
   on startup, preferring `weights/merged_v8m_from_s.pt`. If none is found, a
   friendly pop-up dialog will notify you, and detection will be disabled until
   a compatible file is supplied.
2. **Hardware Check**: The hardware detection module displays your active processing device (e.g. `NVIDIA GeForce RTX 4060 (CUDA)` or `CPU`) on the right-hand panel.
3. **Loading a Video**:
   - Double-click the drag-and-drop target in the sidebar to browse for a local file.
   - Alternatively, drag any supported video file (MP4, AVI, MKV, MOV, WEBM) directly from Windows Explorer and drop it anywhere inside the sidebar.
   - To load a YouTube video, paste the URL in the text entry and click **YT Detect**. In 2/4-lane mode, use the **YT Detect** button on the target lane card. The download progress, speed, and ETA will show in the progress card, and the video will load automatically upon completion.
4. **Running Detection**:
  - Toggle **Save Processed Video** if you want to output a recording of the processed video. Each run receives a timestamped filename, so it does not overwrite a prior result.
   - Click **Start** to run analysis. The stats panel and counter cards will update dynamically in real time.
   - Use the **Pause**, **Resume**, and **Stop** buttons to control playback.
   - Double-click on the video canvas to toggle borderless fullscreen preview. Press **Escape** or double-click to exit.
   - Click **Open Output Folder** to access all saved video outputs.
   - Double-click on items in the **Recent Run History** table to reload previously processed videos.

---

## Building Standalone Executable (.EXE)

The project includes `build_exe.bat`, which uses the project virtual
environment and `VehicleDetector.spec` to package the application, its runtime
libraries, model candidates, UI assets, and portable validation data into a
Windows directory that runs without Python installed on the target PC.

1. Open a Command Prompt as Administrator (if required) in the `VehicleDetector` directory.
2. Execute the batch file:
   ```cmd
   build_exe.bat
   ```
3. PyInstaller will compile the source code. On completion, the build output is generated in:
   `dist\Nirbhighne\`
4. Confirm that the preferred model or a fallback was copied into
   `dist\Nirbhighne\weights\`.
5. Run the application via `dist\Nirbhighne\Nirbhighne.exe`.

---

## Troubleshooting

- **Model file not found alert**: Ensure one of the documented model candidates
  exists in the `weights` folder inside your execution path. The preferred file
  is `merged_v8m_from_s.pt`.
- **High CPU Usage / Low FPS**: Verify that CUDA GPU is active. Under the stats panel, the device should read `(CUDA)`. If it says `CPU`, ensure you have installed PyTorch with CUDA support.
- **YouTube download failures**: YouTube frequently updates its encryption protocols. If yt-dlp fails to download, upgrade it using your terminal:
  ```cmd
  pip install --upgrade yt-dlp
  ```
- **Drag & Drop issues**: On some Windows machines with elevated administrative privileges (UAC settings), drag and drop events might be blocked by the OS if Explorer and the App run on different privilege levels. Run the application under standard user privileges.
