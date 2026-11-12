"""Install the correct runtime for the current computer.

This project supports two separate deployment decisions:

* NVIDIA GPU -> CUDA-enabled PyTorch/torchvision.
* No NVIDIA GPU -> CPU PyTorch/torchvision, with OpenVINO or DirectML used by
  the application when a compatible Intel/AMD/DirectX GPU is available.

requirements.txt contains the platform-independent packages. This script
selects the hardware-specific Torch wheel before installing that common file.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _run_capture(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return completed.returncode, (completed.stdout or "").strip()
    except (FileNotFoundError, OSError) as exc:
        return 1, str(exc)


def detect_hardware() -> dict:
    """Return a small, JSON-serialisable hardware report."""
    nvidia_code, nvidia_output = _run_capture(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    nvidia_rows = []
    if nvidia_code == 0 and nvidia_output:
        for line in nvidia_output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            nvidia_rows.append(
                {
                    "name": parts[0] if parts else "NVIDIA GPU",
                    "driver": parts[1] if len(parts) > 1 else "",
                    "memory_mb": parts[2] if len(parts) > 2 else "",
                }
            )

    adapters = []
    if platform.system() == "Windows":
        ps_command = (
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,AdapterRAM,DriverVersion | "
            "ConvertTo-Json -Compress"
        )
        code, output = _run_capture(["powershell", "-NoProfile", "-Command", ps_command])
        if code == 0 and output:
            try:
                value = json.loads(output)
                adapters = value if isinstance(value, list) else [value]
            except json.JSONDecodeError:
                adapters = [{"raw": output}]

    nvidia_from_adapter = any(
        "nvidia" in str(adapter.get("Name", "")).lower()
        for adapter in adapters
        if isinstance(adapter, dict)
    )
    detection_sources = []
    if nvidia_rows:
        detection_sources.append("nvidia-smi")
    if nvidia_from_adapter:
        detection_sources.append("Windows video adapter")

    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        # WMI is a useful fallback on Windows systems where nvidia-smi is not
        # on PATH yet. Both signals indicate that the CUDA wheel is the right
        # installation choice; runtime CUDA availability is verified later.
        "nvidia_detected": bool(nvidia_rows) or nvidia_from_adapter,
        "nvidia_detection_sources": detection_sources,
        "nvidia_gpus": nvidia_rows,
        "video_adapters": adapters,
    }


def _pip_install(requirements_file: str, *, force: bool = False) -> None:
    command = [sys.executable, "-m", "pip", "install", "--upgrade"]
    if force:
        command.append("--force-reinstall")
    command += ["-r", str(ROOT / requirements_file)]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Detect and print hardware without installing packages.",
    )
    args = parser.parse_args()

    report = detect_hardware()
    selected = "cuda" if report["nvidia_detected"] else "cpu"
    report["selected_torch_runtime"] = selected
    print(json.dumps(report, indent=2))

    if args.check_only:
        return 0

    if not Path(sys.executable).exists():
        raise RuntimeError("The active Python interpreter is not available.")

    print(f"Installing {selected.upper()} PyTorch/torchvision runtime...")
    _pip_install("requirements-cuda.txt" if selected == "cuda" else "requirements-cpu.txt", force=True)
    print("Installing common application dependencies...")
    _pip_install("requirements.txt")

    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    (logs_dir / "hardware_setup.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    import torch
    import torchvision

    print(f"Torch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
    print("Environment setup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
