@echo off
setlocal enabledelayedexpansion

echo.
echo ====================================================================
echo 🚀 OPTIMIZED ENVIRONMENT SETUP
echo ====================================================================
echo.

:: Create and activate virtual environment
echo Step 1: Creating Python virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

:: Upgrade pip and install wheel first
echo Step 2: Upgrading pip and installing wheel...
python -m pip install --upgrade pip --no-cache-dir
pip install wheel --no-cache-dir

:: Configure pip for faster downloads
echo Step 3: Configuring pip for faster downloads...
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
pip config set global.parallel-downloads 8
pip config set global.prefer-binary true
pip config set global.timeout 60
pip config set global.progress-bar on

:: Install core packages first (these are needed for other packages)
echo Step 4: Installing core packages...
pip install --no-cache-dir numpy==1.24.3 pandas==2.0.3 tensorflow==2.13.0

:: Install remaining packages in parallel using requirements file
echo Step 5: Installing remaining packages...
pip install --no-cache-dir -r web_requirements.txt

:: Verify installation
echo Step 6: Verifying installation...
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import streamlit as st; print(f'Streamlit version: {st.__version__}')"

echo.
echo ====================================================================
echo ✅ ENVIRONMENT SETUP COMPLETE
echo ====================================================================
echo.

endlocal 