# -*- mode: python ; coding: utf-8 -*-
"""
VehicleDetector.spec
--------------------
PyInstaller build specification for Nirbhighne.

All hidden imports required by the torch → torchvision → sympy chain are
declared explicitly so the generated dist/ folder is fully self-contained
and runs on any Windows machine without a Python installation.
"""
import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas    = [('icons', 'icons'), ('assets', 'assets')]
binaries = []
hiddenimports = ['tkinter', '_tkinter', 'tkinter.filedialog', 'tkinter.messagebox', 'tkinter.ttk', 'tkinter.colorchooser']

# ── Ensure _tkinter extension and Tcl/Tk DLLs & libraries are collected from Python base prefix ──
py_base = sys.base_prefix
dlls_dir = os.path.join(py_base, 'DLLs')
tcl_dir = os.path.join(py_base, 'tcl')

if os.path.exists(dlls_dir):
    for f in os.listdir(dlls_dir):
        f_lower = f.lower()
        if f_lower.endswith(('.dll', '.pyd')) and ('tcl' in f_lower or 'tk' in f_lower or 'tkinter' in f_lower):
            binaries.append((os.path.join(dlls_dir, f), '.'))

if os.path.exists(tcl_dir):
    datas.append((tcl_dir, 'tcl'))

# ── Core UI packages ─────────────────────────────────────────────────────────

for pkg in ('customtkinter', 'tkinterdnd2'):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

# Optional hardware backends: Intel OpenVINO and Windows DirectML/ONNX.
for pkg in ('openvino', 'onnxruntime', 'onnx'):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

# ── YOLO (ultralytics) ────────────────────────────────────────────────────────
d, b, h = collect_all('ultralytics')
datas += d; binaries += b; hiddenimports += h

# ── Sympy — required deep in torch._dynamo.utils ─────────────────────────────
d, b, h = collect_all('sympy')
datas += d; binaries += b; hiddenimports += h

# ── yt-dlp ───────────────────────────────────────────────────────────────────
d, b, h = collect_all('yt_dlp')
datas += d; binaries += b; hiddenimports += h

# ── OpenCV & Matplotlib ───────────────────────────────────────────────────────
for pkg in ('cv2', 'matplotlib', 'pi_heif'):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

# ── Explicit hidden imports for torch / sympy dynamic import chain ────────────
hiddenimports += [
    'evaluator',
    'evaluation_dataset',
    'traffic_logic',
    'torch._C',
    'pkg_resources.py2_warn',
    'packaging.version',
    'packaging.specifiers',
    'packaging.requirements',
    'charset_normalizer',
    'charset_normalizer.md__mypyc',
]

for sub_pkg in ('sympy', 'torch.fx', 'torch._dynamo', 'torch._inductor', 'torchvision', 'PIL'):
    hiddenimports += collect_submodules(sub_pkg)

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    # Use the local tkinterdnd2 hook so PyInstaller does not need to start
    # Tcl/Tk just to detect the installed drag-and-drop package version.
    hookspath=['.'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Trim unused heavy packages to reduce bundle size
        'notebook',
        'IPython',
        'jupyter',
        'sphinx',
        'pytest',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Nirbhighne',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,                    # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=r'icons/app_icon.ico',
)


coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Nirbhighne',
)
