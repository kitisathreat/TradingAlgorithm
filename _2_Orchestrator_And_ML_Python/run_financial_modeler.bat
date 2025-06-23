@echo off
echo ========================================
echo FINANCIAL MODELER LAUNCHER
echo ========================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ and try again
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking required packages...
python -c "import pandas, numpy, yfinance, openpyxl" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install pandas numpy yfinance openpyxl
    if errorlevel 1 (
        echo ERROR: Failed to install required packages
        pause
        exit /b 1
    )
)

echo.
echo Starting Financial Modeler...
echo.

REM Run the financial modeler
python run_financial_modeler.py

echo.
echo Financial Modeler completed.
pause 