@echo off
chcp 65001 > nul
title VehicleDetector - Build Standalone EXE
cls

:: Always build with the project's prepared virtual environment. It may contain
:: the CUDA or CPU Torch runtime selected by setup_environment.py.
set "PYTHON=venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo  [ERROR] Project Python environment not found at %PYTHON%.
    echo          Create the venv and install requirements before building.
    goto :fail
)

echo.
echo  ============================================================
echo    NIRBHIGHNE  ^|  PyInstaller Build Script
echo  ============================================================
echo.

:: --- Step 1 - Verify Python --------------------------------------------------
echo  [1/6] Checking Python installation...
"%PYTHON%" --version 2>nul
if %errorlevel% neq 0 (
    echo  [ERROR] Python was not found in PATH.
    echo          Install Python 3.11+ and re-run this script.
    goto :fail
)
echo  [OK]
echo.

:: --- Step 2 - Verify PyInstaller ---------------------------------------------
echo  [2/6] Checking PyInstaller...
"%PYTHON%" -c "import PyInstaller" 2>nul
if %errorlevel% neq 0 (
    echo  Installing PyInstaller...
    "%PYTHON%" -m pip install "pyinstaller>=6.0.0"
    if %errorlevel% neq 0 goto :fail
) else (
    echo  [OK] PyInstaller is installed.
)
echo.

:: --- Step 3 - Verify all runtime packages ------------------------------------
echo  [3/6] Verifying runtime packages (sympy, torch, ultralytics, etc.)...
"%PYTHON%" -c "import sympy, torch, torchvision, cv2, ultralytics, customtkinter, tkinterdnd2, yt_dlp, PIL, pi_heif, matplotlib, openvino, onnxruntime, onnx" 2>nul
if %errorlevel% neq 0 (
    echo  [WARNING] One or more packages are missing. Detecting hardware and installing runtime...
    "%PYTHON%" setup_environment.py
    if %errorlevel% neq 0 goto :fail
) else (
    echo  [OK] All packages verified.
)
echo.

:: --- Step 4 - Clean previous build -------------------------------------------
echo  [4/6] Cleaning previous build artefacts...
if exist "build"            rmdir /s /q "build"
if exist "dist\Nirbhighne"   rmdir /s /q "dist\Nirbhighne"
echo  [OK] Clean complete.
echo.

:: --- Step 5 - Run PyInstaller via spec ---------------------------------------
echo  [5/6] Running PyInstaller (this may take 5-15 minutes)...
echo.
"%PYTHON%" -m PyInstaller VehicleDetector.spec --noconfirm

if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] PyInstaller failed. Review the output above for details.
    goto :fail
)

:: --- Step 6 - Post-build folder setup and weight copy ------------------------
echo.
echo  [6/6] Finalising output bundle...
mkdir "dist\Nirbhighne\weights" 2>nul
mkdir "dist\Nirbhighne\assets"  2>nul
mkdir "dist\Nirbhighne\icons"   2>nul
mkdir "dist\Nirbhighne\logs"    2>nul
mkdir "dist\Nirbhighne\output"  2>nul

if exist "icons" (
    xcopy "icons" "dist\Nirbhighne\icons" /E /I /Y >nul
    echo  [OK] icons copied into bundle.
)


if exist "weights\best.pt" (
    copy "weights\best.pt" "dist\Nirbhighne\weights\best.pt" /Y
    echo  [OK] weights\best.pt copied into bundle.
)
if exist "weights\merged_v8m_from_s.pt" (
    copy "weights\merged_v8m_from_s.pt" "dist\Nirbhighne\weights\merged_v8m_from_s.pt" /Y
    echo  [OK] weights\merged_v8m_from_s.pt copied into bundle.
)
if exist "weights\final_best_v8m.pt" (
    copy "weights\final_best_v8m.pt" "dist\Nirbhighne\weights\final_best_v8m.pt" /Y
    echo  [OK] weights\final_best_v8m.pt copied into bundle.
)
if exist "final_best_v8m.pt" (
    copy "final_best_v8m.pt" "dist\Nirbhighne\weights\final_best_v8m.pt" /Y
    echo  [OK] final_best_v8m.pt copied into bundle.
)
if not exist "dist\Nirbhighne\weights\merged_v8m_from_s.pt" if not exist "dist\Nirbhighne\weights\best.pt" if not exist "dist\Nirbhighne\weights\final_best_v8m.pt" (
    echo  [INFO] Model weights not found in the source folder.
    echo         The EXE will show a friendly warning on first launch
    echo         until you place best.pt or final_best_v8m.pt inside dist\Nirbhighne\weights\
)

:: Include only the validation split required by the in-app evaluator.
:: The full train/test dataset is intentionally excluded to keep the portable bundle practical.
if exist "data.yml & utility file\portable_eval.yaml" (
    echo  Preparing portable evaluation dataset (including HEIC conversion)...
    "%PYTHON%" evaluation_dataset.py "data.yml & utility file\portable_eval.yaml" "dist\Nirbhighne\evaluation_data"
    if errorlevel 1 (
        echo  [ERROR] Could not copy the portable evaluation dataset.
        goto :fail
    )
    echo  [OK] Validation dataset copied for offline evaluation.
) else (
    echo  [WARNING] Portable evaluation dataset was not found.
    echo            The EXE can still evaluate a data.yaml selected from an external dataset.
)

echo.
echo  ============================================================
echo    BUILD COMPLETE
echo.
echo    Output folder : dist\Nirbhighne\
echo    Executable    : dist\Nirbhighne\Nirbhighne.exe
echo.
echo    TO DISTRIBUTE TO ANOTHER PC:
echo      1. Zip the entire  dist\Nirbhighne\  folder
echo      2. Extract on the target PC anywhere you like
echo      3. Double-click  Nirbhighne.exe
echo      4. No Python installation required on the target machine
echo      5. The included validation dataset enables offline standard evaluation
echo  ============================================================

echo.
goto :end

:fail
echo.
echo  ============================================================
echo    BUILD FAILED.  Review errors printed above.
echo  ============================================================
echo.

:end
pause
