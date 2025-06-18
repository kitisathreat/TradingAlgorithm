@echo off
echo ========================================
echo   Enhanced Local GUI Trading App
echo ========================================
echo.
echo Starting enhanced PyQt5 GUI application...
echo Features:
echo - Fixed QSpinBox compatibility
echo - Comprehensive technical indicators
echo - Interactive chart with zoom and pan
echo - Days selector dropdown (30, 90, 180, 365, Custom)
echo - Full S&P 100 stock list
echo - Auto-fit chart controls
echo.

cd /d "%~dp0\..\..\_3_Networking_and_User_Input\local_gui"

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
python -c "import PyQt5, pyqtgraph, yfinance, pandas, numpy" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Some required packages may be missing
    echo Installing required packages...
    pip install PyQt5 pyqtgraph yfinance pandas numpy
    if errorlevel 1 (
        echo [ERROR] Failed to install required packages
        pause
        exit /b 1
    )
)

echo [INFO] Starting PyQt5 GUI application...
echo [INFO] The GUI window should appear shortly
echo [INFO] Close the window to exit
echo.

REM Start the PyQt5 application
python main.py

echo.
echo [INFO] GUI application closed
pause 