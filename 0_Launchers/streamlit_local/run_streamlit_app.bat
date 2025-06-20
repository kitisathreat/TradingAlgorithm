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

:: Change to project root directory (two levels up from streamlit_local)
cd /d "%~dp0..\.."

:: Verify we're in the correct directory and streamlit_app.py exists
if not exist "streamlit_app.py" (
    echo ❌ Error: streamlit_app.py not found in current directory: %CD%
    echo Expected location: %CD%\streamlit_app.py
    echo Please ensure you're running this from the correct project structure
    pause
    exit /b 1
)

echo ✓ Found streamlit_app.py in: %CD%

:: Activate the virtual environment
echo Activating virtual environment...
call "%VENV_ACTIVATE%"
if errorlevel 1 (
    echo ❌ Error: Failed to activate virtual environment
    echo Please ensure no other processes are using the virtual environment
    pause
    exit /b 1
)

:: Check if port 8501 is already in use before starting
echo Checking if port 8501 is available...
netstat -an | findstr ":8501" | findstr "LISTENING" >nul
if not errorlevel 1 (
    echo ❌ Error: Port 8501 is already in use
    echo Please close any other applications using port 8501 and try again
    echo.
    echo To find what's using the port, run: netstat -ano | findstr :8501
    echo.
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

streamlit run "streamlit_app.py" --server.port 8501 --server.address localhost
set STREAMLIT_EXIT_CODE=%errorlevel%

:: Only show error message if it's not a normal interruption (Ctrl+C typically returns 2)
if %STREAMLIT_EXIT_CODE% neq 0 (
    if %STREAMLIT_EXIT_CODE% neq 2 (
        echo.
        echo ❌ Error: Failed to start Streamlit app (Exit code: %STREAMLIT_EXIT_CODE%)
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
    ) else (
        echo.
        echo ✓ Streamlit server stopped successfully
    )
) else (
    echo.
    echo ✓ Streamlit server stopped successfully
)

endlocal 