import os
import sys
import io
import logging
import socket
import torch
from logging.handlers import RotatingFileHandler
import config


class NullBytesStream(io.RawIOBase):
    """Fallback byte stream for PyInstaller GUI mode when stdout/stderr buffer is accessed."""

    def write(self, b):
        return len(b) if b else 0

    def read(self, size=-1):
        return b""

    def readinto(self, b):
        return 0

    def flush(self):
        pass


class NullStream(io.TextIOBase):
    """Fallback text stream for PyInstaller GUI mode where sys.stdout/stderr are None."""

    def __init__(self):
        super().__init__()
        self.buffer = NullBytesStream()

    def write(self, s):
        return len(s) if s else 0

    def writelines(self, lines):
        pass

    def flush(self):
        pass

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"

    @property
    def closed(self):
        return False


def fix_pyinstaller_streams():
    """Ensure sys.stdout, sys.stderr, and sys.stdin are valid stream objects when frozen without console."""
    if sys.stdout is None:
        sys.stdout = NullStream()
    if sys.stderr is None:
        sys.stderr = NullStream()
    if sys.stdin is None:
        sys.stdin = io.StringIO("")


# Ensure streams are valid immediately on module import
fix_pyinstaller_streams()


def get_device_info(device: str | None = None):
    """Detects CPU or CUDA GPU and returns (device_string, device_name_string).

    Parameters
    ----------
    device : str | None
        If 'cpu' or 'cuda', returns display info for that specific device.
        If None, auto-detects based on CUDA availability.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device.lower() == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return "cuda", f"{gpu_name} (CUDA)"
    if device.lower() == "intel:gpu":
        try:
            import openvino as ov

            core = ov.Core()
            for gpu in core.available_devices:
                if gpu.startswith("GPU"):
                    name = str(core.get_property(gpu, "FULL_DEVICE_NAME"))
                    if "intel" in name.lower():
                        return "intel:gpu", f"{name} (OpenVINO)"
        except Exception:
            pass
        return "intel:gpu", "Intel GPU (OpenVINO)"
    if device.lower() == "directml":
        return "directml", "DirectX 12 GPU (DirectML)"
    else:
        return "cpu", "CPU"


def setup_logger():
    """Sets up a rotating file logger and standard console logger."""
    os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)
    logger = logging.getLogger("VehicleDetector")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

    # Rotating File Handler
    file_handler = RotatingFileHandler(config.LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=3, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler (only if running with real console)
    if sys.__stdout__ is not None:
        try:
            if hasattr(sys.__stdout__, 'buffer'):
                console_stream = io.TextIOWrapper(sys.__stdout__.buffer, encoding='utf-8', errors='replace')
            else:
                console_stream = sys.__stdout__
            console_handler = logging.StreamHandler(console_stream)
        except Exception:
            console_handler = logging.StreamHandler(sys.__stdout__)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

logger = setup_logger()

def check_internet_connection(host="8.8.8.8", port=53, timeout=3):
    """
    Checks if internet is accessible.
    Default uses Google Public DNS.
    """
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
        return True
    except socket.error:
        return False

def format_time(seconds):
    """Formats elapsed seconds to MM:SS or HH:MM:SS."""
    seconds = max(0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
