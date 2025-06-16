@echo off
setlocal enabledelayedexpansion

:: Check for Python 3.9 using py launcher
py -3.9 --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python 3.9 is required but not found or not registered with the 'py' launcher.
    echo        Please install Python 3.9 and ensure it is registered with the 'py' launcher.
    echo        Download: https://www.python.org/downloads/release/python-390/
    echo.
    pause
    exit /b 1
)

:: Set virtual environment path to AppData
set "VENV_PATH=%LOCALAPPDATA%\TradingAlgorithm\venv"
set "VENV_ACTIVATE=%VENV_PATH%\Scripts\activate.bat"

:: Clear screen and set console width for better formatting
mode con: cols=100 lines=40
cls

echo.
echo ================================================================================
echo                    NEURAL NETWORK TRADING SYSTEM - LOCAL GUI
echo ================================================================================
echo.
echo  This system trains an AI to mimic your trading decisions by combining:
echo  * Technical Analysis (RSI, MACD, Bollinger Bands)
echo  * Sentiment Analysis (VADER analysis of your reasoning)
echo  * Neural Network Learning (TensorFlow deep learning)
echo.
echo --------------------------------------------------------------------------------
echo.

:: Check for Python 3.9
echo [1/5] Checking Python installation...
python --version 2>nul | findstr /R "Python 3.9" >nul
if errorlevel 1 (
    echo.
    echo [ERROR] Python 3.9 is required but not found
    echo        Please install Python 3.9 from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
) else (
    echo [OK] Python 3.9 found
)

:: Check and remove existing virtual environment
echo.
echo [2/5] Checking virtual environment...
if exist "%VENV_PATH%" (
    echo    Found existing virtual environment
    echo    Attempting to remove...
    
    :: Remove virtual environment (no admin rights needed in AppData)
    rmdir /s /q "%VENV_PATH%" 2>nul
    if exist "%VENV_PATH%" (
        echo.
        echo [ERROR] Could not remove existing virtual environment
        echo        Please close any applications using the virtual environment and try again
        echo.
        pause
        exit /b 1
    )
    echo [OK] Successfully removed existing virtual environment
) else (
    echo [OK] No existing virtual environment found
)

:: Create AppData directory if it doesn't exist
if not exist "%LOCALAPPDATA%\TradingAlgorithm" (
    mkdir "%LOCALAPPDATA%\TradingAlgorithm"
)

:: Set up new environment
echo.
echo [3/5] Setting up Python environment...
echo    This may take a few minutes for first-time setup...
echo    Installing dependencies (this will show progress)...

:: Create and activate virtual environment
echo    Creating virtual environment in: %VENV_PATH%
py -3.9 -m venv "%VENV_PATH%"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to create virtual environment
    echo        Please check if you have write permissions to: %LOCALAPPDATA%
    echo.
    pause
    exit /b 1
)

:: Activate virtual environment and install dependencies
call "%VENV_ACTIVATE%"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to activate virtual environment
    echo        Please check if the virtual environment was created correctly
    echo.
    pause
    exit /b 1
)

:: Install dependencies
echo    Upgrading pip and installing core packages...
py -3.9 -m pip install --upgrade pip
py -3.9 -m pip install wheel

echo    Installing GUI dependencies...
py -3.9 -m pip install numpy==1.23.5 scikit-learn==1.3.0 PyQt6==6.4.2 pyqtgraph==0.13.3 pandas==2.0.3 yfinance==0.2.36 qt-material==2.14 tensorflow==2.13.0

if errorlevel 1 (
    echo.
    echo [ERROR] Setup failed!
    echo.
    echo Common solutions:
    echo    * Ensure you have a stable internet connection
    echo    * Make sure Python 3.9 is properly installed
    echo    * Check if antivirus is blocking the installation
    echo.
    pause
    exit /b 1
)

echo [OK] Environment setup completed successfully

:: Verify required packages
echo.
echo [4/5] Verifying required packages...
python -c "import PyQt6" 2>nul
if errorlevel 1 (
    echo [ERROR] PyQt6 is not installed
    echo        Please check the installation logs above
    echo.
    pause
    exit /b 1
)

python -c "import tensorflow" 2>nul
if errorlevel 1 (
    echo [ERROR] TensorFlow is not installed
    echo        Please check the installation logs above
    echo.
    pause
    exit /b 1
)

echo [OK] All required packages verified

:: Launch the GUI application
echo.
echo [5/5] Launching Neural Network Trading System GUI...
echo.
echo ================================================================================
echo  NOTES:
echo  * The GUI will open in a new window
echo  * Close the GUI window to exit the application
echo  * To deactivate the virtual environment after stopping, run: deactivate
echo  * Virtual environment location: %VENV_PATH%
echo ================================================================================
echo.

py -3.9 "..\_3_Networking_and_User_Input\local_gui\main.py"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start GUI application
    echo        Please check the error messages above
    echo.
    pause
    exit /b 1
)

endlocal 