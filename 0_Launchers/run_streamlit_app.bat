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

echo Starting Streamlit app...

:: Find Python 3.9 installation dynamically
for /f "tokens=*" %%i in ('where python') do (
    set PYTHON_PATH=%%i
    python --version 2>nul | findstr /R "Python 3.9" >nul
    if not errorlevel 1 goto :found_python
)
echo ❌ Error: Python 3.9 not found in PATH
echo Please ensure Python 3.9 is installed and added to PATH
pause
exit /b 1

:found_python
echo ✓ Found Python 3.9 at: %PYTHON_PATH%

:: Verify virtual environment
if not exist "%VENV_ACTIVATE%" (
    echo ❌ Error: Virtual environment not found at: %VENV_PATH%
    echo Please run setup_and_run.bat to create the virtual environment
    pause
    exit /b 1
)

:: Change to script directory
cd /d "%~dp0"

:: Activate the virtual environment
echo Activating virtual environment...
call "%VENV_ACTIVATE%"
if errorlevel 1 (
    echo ❌ Error: Failed to activate virtual environment
    echo Please ensure no other processes are using the virtual environment
    pause
    exit /b 1
)

:: Run the Neural Network Training Streamlit app
echo.
echo ════════════════════════════════════════════════════════════════════════════════════════════════════════
echo 🚀 Starting Advanced Neural Network Trading System...
echo 📊 Open your browser to: http://localhost:8501
echo ════════════════════════════════════════════════════════════════════════════════════════════════════════
echo.
echo Press Ctrl+C to stop the server when done
echo.

streamlit run "..\_3_Networking_and_User_Input\web_interface\streamlit_app.py" --server.port 8501 --server.address localhost
if errorlevel 1 (
    echo.
    echo ❌ Error: Failed to start Streamlit app
    echo Common issues:
    echo - Port 8501 is already in use
    echo - Streamlit installation is corrupted
    echo - Virtual environment is not properly activated
    echo.
    echo Try these solutions:
    echo 1. Close any other Streamlit apps
    echo 2. Run setup_and_run.bat again
    echo 3. Check if antivirus is blocking the connection
    echo.
    pause
    exit /b 1
)

endlocal 