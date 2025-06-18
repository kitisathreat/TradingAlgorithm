@echo off
setlocal enabledelayedexpansion

REM Neural Network Trading System - Flask Web Interface Launcher
REM This script sets up the environment and launches the Flask web interface

echo.
echo ================================================================================
echo                    NEURAL NETWORK TRADING SYSTEM - FLASK WEB INTERFACE
echo ================================================================================
echo.
echo  This system trains an AI to mimic your trading decisions by combining:
echo  * Technical Analysis (RSI, MACD, Bollinger Bands)
echo  * Sentiment Analysis (VADER analysis of your reasoning)
echo  * Neural Network Learning (TensorFlow deep learning)
echo.
echo  Web interface will be available at: http://localhost:5000
echo.

REM Get script directory
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\.."

REM Check if Python 3.9 is available
echo [1/5] Checking Python installation...
python --version 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH
    echo         Please install Python 3.9 from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
echo [OK] Found Python %PYTHON_VERSION%

REM Detect virtual environment path
echo.
echo [2/5] Setting up virtual environment...

REM Try LOCALAPPDATA first
if defined LOCALAPPDATA (
    set "VENV_PATH=%LOCALAPPDATA%\TradingAlgorithm\venv"
    echo [INFO] Using AppData location: %VENV_PATH%
) else (
    set "VENV_PATH=%REPO_ROOT%\venv"
    echo [INFO] Using project directory: %VENV_PATH%
)

REM Check if virtual environment exists
if exist "%VENV_PATH%\Scripts\activate.bat" (
    echo [OK] Found existing virtual environment
    echo     Activating existing environment...
    
    REM Activate virtual environment
    call "%VENV_PATH%\Scripts\activate.bat"
    if errorlevel 1 (
        echo [WARNING] Failed to activate existing virtual environment
        echo           Will create a new one...
        goto CREATE_VENV
    )
    
    REM Check if Flask is installed
    echo.
    echo [3/5] Checking installed packages...
    python -c "import flask" 2>nul
    if errorlevel 1 (
        echo [INFO] Flask not found in virtual environment
        echo        Will install required packages...
    ) else (
        echo [OK] Flask is already installed
    )
    
    REM Skip to dependency installation
    goto INSTALL_DEPS
    
) else (
    echo [INFO] No existing virtual environment found
    echo        Creating new virtual environment...
    goto CREATE_VENV
)

:CREATE_VENV
REM Create parent directory if needed
if not exist "%VENV_PATH%\.." (
    echo     Creating directory: %VENV_PATH%\..
    mkdir "%VENV_PATH%\.." 2>nul
    if errorlevel 1 (
        echo [ERROR] Could not create directory
        echo         This may be due to insufficient permissions or disk space.
        echo.
        pause
        exit /b 1
    )
)

REM Create virtual environment
echo.
echo [3/5] Creating virtual environment...
echo     This may take a few minutes for first-time setup...
python -m venv "%VENV_PATH%"
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    echo         Please check if you have write permissions to: %VENV_PATH%
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo     Activating new virtual environment...
call "%VENV_PATH%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    echo.
    pause
    exit /b 1
)

:INSTALL_DEPS
REM Upgrade pip and install dependencies
echo.
echo [4/5] Installing/updating dependencies...

REM Only upgrade pip if needed
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo     Upgrading pip...
    python -m pip install --upgrade pip
)

REM Install consolidated requirements
echo     Installing consolidated dependencies...
cd /d "%REPO_ROOT%"
python -m pip install -r requirements.txt --upgrade-strategy only-if-needed
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    echo.
    pause
    exit /b 1
)

REM Launch Flask web interface
echo.
echo [5/5] Launching Flask web interface...
echo.
echo ================================================================================
echo                    STARTING FLASK WEB INTERFACE
echo ================================================================================
echo.
echo  Web interface will be available at: http://localhost:5000
echo  Press Ctrl+C to stop the server
echo.
echo ================================================================================
echo.

cd /d "%SCRIPT_DIR%"
python flask_app.py

REM If we get here, the Flask app has stopped
echo.
echo Flask web interface has stopped.
pause 