@echo off
setlocal enabledelayedexpansion

:: Set virtual environment path to AppData (consistent with other scripts)
set "VENV_PATH=%LOCALAPPDATA%\TradingAlgorithm\venv"
set "VENV_ACTIVATE=%VENV_PATH%\Scripts\activate.bat"

echo.
echo ====================================================================
echo 🚀 OPTIMIZED ENVIRONMENT SETUP
echo ====================================================================
echo.

:: Create AppData directory if it doesn't exist
if not exist "%LOCALAPPDATA%\TradingAlgorithm" (
    mkdir "%LOCALAPPDATA%\TradingAlgorithm"
)

:: Create and activate virtual environment
echo Step 1: Creating Python virtual environment in: %VENV_PATH%
py -3.9 -m venv "%VENV_PATH%"
call "%VENV_ACTIVATE%"

:: Upgrade pip and install wheel first
echo Step 2: Upgrading pip and installing wheel...
py -3.9 -m pip install --upgrade pip --no-cache-dir
py -3.9 -m pip install wheel --no-cache-dir

:: Configure pip for faster downloads
echo Step 3: Configuring pip for faster downloads...
py -3.9 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
py -3.9 -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
py -3.9 -m pip config set global.parallel-downloads 8
py -3.9 -m pip config set global.prefer-binary true
py -3.9 -m pip config set global.timeout 60
py -3.9 -m pip config set global.progress-bar on

:: Install core packages first (these are needed for other packages)
echo Step 4: Installing core packages...
py -3.9 -m pip install --no-cache-dir "numpy==1.24.3" "pandas==2.0.3" "tensorflow==2.13.0" "PyQt5==5.15.9"

:: Install remaining packages in parallel using requirements file
echo Step 5: Installing remaining packages...
py -3.9 -m pip install --no-cache-dir -r "%~dp0..\..\requirements.txt"

:: Verify installation
echo Step 6: Verifying installation...
py -3.9 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
py -3.9 -c "import flask; print(f'Flask version: {flask.__version__}')"

echo.
echo ====================================================================
echo ✅ ENVIRONMENT SETUP COMPLETE
echo ====================================================================
echo.

endlocal 