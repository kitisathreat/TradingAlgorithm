@echo off
setlocal enabledelayedexpansion

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
echo                    NEURAL NETWORK TRADING SYSTEM - QUICK LAUNCH
echo ================================================================================
echo.
echo  This system trains an AI to mimic your trading decisions by combining:
echo  * Technical Analysis (RSI, MACD, Bollinger Bands)
echo  * Sentiment Analysis (VADER analysis of your reasoning)
echo  * Neural Network Learning (TensorFlow deep learning)
echo.
echo  Using PyQt5 for better Windows compatibility
echo  Virtual Environment Location: %VENV_PATH%
echo  (This location is accessible to all users and requires no admin rights)
echo.
echo --------------------------------------------------------------------------------
echo.

:: Check for Python 3.9
echo [1/4] Checking Python installation...
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

:: Check if virtual environment exists
echo.
echo [2/4] Checking virtual environment...
if not exist "%VENV_PATH%" (
    echo    No existing virtual environment found
    echo    Running full setup...
    call "%~dp0run_local_gui.bat"
    if errorlevel 1 (
        echo.
        echo [ERROR] Environment setup failed
        echo        Please check the error messages above
        echo.
        pause
        exit /b 1
    )
    exit /b 0
) else (
    echo [OK] Found existing virtual environment
)

:: Test PyQt5 installation
echo.
echo [3/4] Testing PyQt5 installation...
call "%VENV_ACTIVATE%"
py -3.9 -c "from PyQt5.QtWidgets import QApplication; print('PyQt5 import successful')" 2>nul
if errorlevel 1 (
    echo.
    echo [WARNING] PyQt5 not found in existing environment
    echo    Reinstalling environment with PyQt5...
    call "%~dp0run_local_gui.bat"
    if errorlevel 1 (
        echo.
        echo [ERROR] Environment setup failed
        echo        Please check the error messages above
        echo.
        pause
        exit /b 1
    )
    exit /b 0
) else (
    echo [OK] PyQt5 import successful
)

:: Activate environment and launch GUI
echo.
echo [4/4] Launching Neural Network Trading System GUI...
echo.
echo ================================================================================
echo  NOTES:
echo  * The GUI will open in a new window
echo  * Close the GUI window to exit the application
echo  * To deactivate the virtual environment after stopping, run: deactivate
echo  * Virtual environment location: %VENV_PATH%
echo  * Using PyQt5 for better Windows compatibility
echo ================================================================================
echo.

:: Activate virtual environment and run the GUI
call "%VENV_ACTIVATE%"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to activate virtual environment
    echo        Please check if the virtual environment was created correctly
    echo.
    pause
    exit /b 1
)

py -3.9 "..\..\_3_Networking_and_User_Input\local_gui\main.py"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start GUI application
    echo        Please check the error messages above
    echo.
    echo [TROUBLESHOOTING] If you're still having issues:
    echo    1. Try running: py -3.9 -c "import sys; print(sys.version)"
    echo    2. Check if you have any conflicting Qt installations
    echo    3. Try restarting your computer and running again
    echo.
    pause
    exit /b 1
)

endlocal 