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

:: Check for Python 3.9
echo [1/3] Checking Python installation...
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
echo [2/3] Checking virtual environment...
if not exist "%VENV_PATH%" (
    echo    No existing virtual environment found
    echo    Running full setup...
    call "%~dp0setup_local_env.bat"
    if errorlevel 1 (
        echo.
        echo [ERROR] Environment setup failed
        echo        Please check the error messages above
        echo.
        pause
        exit /b 1
    )
) else (
    echo [OK] Found existing virtual environment
)

:: Activate environment and launch GUI
echo.
echo [3/3] Launching Neural Network Trading System GUI...
echo.
echo ================================================================================
echo  NOTES:
echo  * The GUI will open in a new window
echo  * Close the GUI window to exit the application
echo  * To deactivate the virtual environment after stopping, run: deactivate
echo  * Virtual environment location: %VENV_PATH%
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

python "..\_3_Networking_and_User_Input\local_gui\main.py"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start GUI application
    echo        Please check the error messages above
    echo.
    pause
    exit /b 1
)

endlocal 