@echo off
setlocal enabledelayedexpansion

REM Neural Network Trading System - Production Server Launcher
REM This script runs the Flask web interface in production mode

echo.
echo ================================================================================
echo                    NEURAL NETWORK TRADING SYSTEM - PRODUCTION SERVER
echo ================================================================================
echo.
echo  This system trains an AI to mimic your trading decisions by combining:
echo  * Technical Analysis (RSI, MACD, Bollinger Bands)
echo  * Sentiment Analysis (VADER analysis of your reasoning)
echo  * Neural Network Learning (TensorFlow deep learning)
echo.
echo  Production server will be available at: http://0.0.0.0:5000
echo.

REM Get script directory
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\.."

REM Set production environment variables
set "FLASK_ENV=production"
set "SECRET_KEY=trading_algorithm_production_secret_2024"
set "LOG_LEVEL=INFO"
set "CORS_ORIGINS=*"

REM Check if Python is available
echo [1/4] Checking Python installation...
python --version 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH
    echo         Please install Python 3.9 from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM Check if virtual environment exists
echo.
echo [2/4] Setting up virtual environment...
if exist "venv\Scripts\activate.bat" (
    echo [OK] Found existing virtual environment
    call venv\Scripts\activate.bat
) else (
    echo [INFO] Creating new virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo [INFO] Installing dependencies...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
)

REM Check if Gunicorn is available
echo.
echo [3/4] Checking production dependencies...
python -c "import gunicorn" 2>nul
if errorlevel 1 (
    echo [INFO] Installing production dependencies...
    python -m pip install gunicorn eventlet
)

REM Check for additional dependencies
echo [INFO] Checking additional dependencies...
python -c "import lxml" 2>nul
if errorlevel 1 (
    echo [INFO] Installing additional dependencies...
    python -m pip install lxml tqdm websockets textblob tenacity
)

REM Create necessary directories
echo.
echo [4/4] Setting up directories...
if not exist "logs" mkdir logs
if not exist "models" mkdir models
if not exist "cache" mkdir cache

REM Launch production server
echo.
echo ================================================================================
echo                    STARTING PRODUCTION SERVER
echo ================================================================================
echo.
echo  Server will be available at: http://0.0.0.0:5000
echo  Environment: Production
echo  Press Ctrl+C to stop the server
echo.
echo ================================================================================
echo.

REM Run with Gunicorn for production
python -m gunicorn --config gunicorn.conf.py wsgi:application

REM If we get here, the server has stopped
echo.
echo Production server has stopped.
pause 