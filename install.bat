@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

echo ========================================================
echo   Installing Pedestrian Safety Detection Environment
echo ========================================================

:: Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed. Please install Python 3.7 or higher.
    exit /b 1
)

:: Check Python version
for /f "tokens=*" %%i in ('python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"') do set PY_VERSION=%%i
echo Using Python version: %PY_VERSION%

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install required packages
echo Installing required packages...
pip install -r requirements.txt

:: Check if CUDA is available for GPU acceleration
for /f "tokens=*" %%i in ('python -c "import torch; print(torch.cuda.is_available())"') do set CUDA_AVAILABLE=%%i
if "%CUDA_AVAILABLE%" == "True" (
    echo CUDA is available. GPU acceleration will be used.
) else (
    echo CUDA is not available. Using CPU mode.
)

:: Create data directory
if not exist data mkdir data

echo ========================================================
echo Installation completed!
echo.
echo To activate the environment, run:
echo venv\Scripts\activate.bat
echo.
echo Example usage:
echo python pedestrian_detection_haar.py --webcam
echo ========================================================

ENDLOCAL 