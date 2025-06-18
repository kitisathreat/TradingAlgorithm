@echo off
echo ========================================
echo   Enhanced Flask Trading App Launcher
echo ========================================
echo.
echo Starting enhanced Flask web application...
echo Features:
echo - Comprehensive technical indicators
echo - Interactive chart with zoom and pan
echo - Days selector dropdown (30, 90, 180, 365, Custom)
echo - Full S&P 100 stock list
echo - Auto-fit chart controls
echo.

cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9+ and try again
    pause
    exit /b 1
)

REM Check if required packages are installed
echo [INFO] Checking required packages...
python -c "import flask, flask_socketio, yfinance, pandas, numpy" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Some required packages may be missing
    echo Installing required packages...
    pip install flask flask-socketio yfinance pandas numpy
    if errorlevel 1 (
        echo [ERROR] Failed to install required packages
        pause
        exit /b 1
    )
)

echo [INFO] Starting Flask application...
echo [INFO] Web interface will be available at: http://localhost:5000
echo [INFO] Press Ctrl+C to stop the server
echo.

REM Start the Flask application
python flask_app.py

echo.
echo [INFO] Flask application stopped
pause 