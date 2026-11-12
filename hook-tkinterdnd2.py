"""PyInstaller hook for tkinterdnd2 without starting Tcl/Tk at build time.

The upstream hook asks tkinter.Tcl() for the Tcl version. That is unnecessary
for this Windows build and fails on systems whose Python Tcl installation
cannot initialize during PyInstaller analysis. The package contains the
platform-specific tkdnd files, so select the correct directory directly.
"""

import os
import pathlib
import platform

from PyInstaller.utils.hooks import get_package_paths


datas = []
binaries = []

if platform.system() == "Windows":
    machine = os.environ.get("PROCESSOR_ARCHITECTURE", platform.machine())
    dir_name = {
        "AMD64": "win-x64",
        "ARM64": "win-arm64",
        "x86": "win-x86",
    }.get(machine)

    if dir_name:
        _pkg_base, package_dir = get_package_paths("tkinterdnd2")
        source_dir = pathlib.Path(package_dir) / "tkdnd" / dir_name
        if source_dir.is_dir():
            target_dir = os.path.join("tkinterdnd2", "tkdnd", dir_name)
            binaries.extend(
                (str(path), target_dir)
                for path in source_dir.glob("*.dll")
            )
            datas.extend(
                (str(path), target_dir)
                for path in source_dir.glob("*.tcl")
            )
           