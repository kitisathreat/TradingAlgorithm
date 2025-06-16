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

:: Cloud-compatible virtual environment path detection
echo [INFO] Detecting environment for virtual environment path...

if defined LOCALAPPDATA (
    :: Test if LOCALAPPDATA is writable
    echo test > "%LOCALAPPDATA%\write_test.tmp" 2>nul
    if exist "%LOCALAPPDATA%\write_test.tmp" (
        del "%LOCALAPPDATA%\write_test.tmp" 2>nul
        set "VENV_PATH=%LOCALAPPDATA%\TradingAlgorithm\venv"
        echo [OK] Using AppData location: %VENV_PATH%
    ) else (
        set "VENV_PATH=%~dp0..\venv"
        echo [WARNING] LOCALAPPDATA not writable, using project directory: %VENV_PATH%
    )
) else (
    set "VENV_PATH=%~dp0..\venv"
    echo [INFO] LOCALAPPDATA not available, using project directory: %VENV_PATH%
)

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
echo  Virtual Environment Location: %VENV_PATH%
echo  (This location is accessible to all users and requires no admin rights)
echo.
echo --------------------------------------------------------------------------------
echo.

:: Check for Python 3.9
echo [1/6] Checking Python installation...
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
echo [2/6] Checking virtual environment...
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

:: Create AppData directory if it doesn't exist (only for AppData path)
if "%VENV_PATH%"=="%LOCALAPPDATA%\TradingAlgorithm\venv" (
    if not exist "%LOCALAPPDATA%\TradingAlgorithm" (
        echo    Creating directory: %LOCALAPPDATA%\TradingAlgorithm
        mkdir "%LOCALAPPDATA%\TradingAlgorithm"
        if errorlevel 1 (
            echo.
            echo [ERROR] Could not create directory: %LOCALAPPDATA%\TradingAlgorithm
            echo        This may be due to insufficient permissions or disk space.
            echo        Please ensure you have write access to your AppData folder.
            echo.
            pause
            exit /b 1
        )
    )
)

:: Set up new environment
echo.
echo [3/6] Setting up Python environment...
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

echo    Installing core dependencies first...
py -3.9 -m pip install "numpy==1.23.5" "scikit-learn==1.3.0" "pandas==2.0.3" "yfinance==0.2.36" "tensorflow==2.13.0"

echo    Installing PyQt6 with specific version for Windows compatibility...
py -3.9 -m pip install "PyQt6==6.4.2" "PyQt6-Qt6==6.4.2" "PyQt6-sip==13.4.1"

echo    Installing additional GUI dependencies...
py -3.9 -m pip install "pyqtgraph==0.13.3" "qt-material==2.14"

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

:: Test PyQt6 import
echo.
echo [4/6] Testing PyQt6 installation...
py -3.9 -c "from PyQt6.QtWidgets import QApplication; print('PyQt6 import successful')" 2>nul
if errorlevel 1 (
    echo.
    echo [WARNING] PyQt6 import test failed. Attempting alternative installation...
    echo    Trying PyQt5 as fallback...
    py -3.9 -m pip uninstall PyQt6 PyQt6-Qt6 PyQt6-sip -y
    py -3.9 -m pip install "PyQt5==5.15.9"
    echo    Testing PyQt5...
    py -3.9 -c "from PyQt5.QtWidgets import QApplication; print('PyQt5 import successful')" 2>nul
    if errorlevel 1 (
        echo.
        echo [ERROR] Both PyQt6 and PyQt5 failed to import!
        echo.
        echo This may be due to:
        echo    * Missing Visual C++ Redistributable (download from Microsoft)
        echo    * Conflicting Qt installations
        echo    * System architecture mismatch
        echo.
        echo Please try:
        echo    1. Install Visual C++ Redistributable 2015-2022
        echo    2. Restart your computer
        echo    3. Run this script again
        echo.
        pause
        exit /b 1
    ) else (
        echo [OK] PyQt5 import successful - will use PyQt5 instead
    )
) else (
    echo [OK] PyQt6 import successful
)

:: Launch the GUI application
echo.
echo [5/6] Launching Neural Network Trading System GUI...
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
    echo [TROUBLESHOOTING] If you're still having issues:
    echo    1. Try running: py -3.9 -c "import sys; print(sys.version)"
    echo    2. Check if you have Visual C++ Redistributable installed
    echo    3. Try restarting your computer and running again
    echo.
    pause
    exit /b 1
)

endlocal 