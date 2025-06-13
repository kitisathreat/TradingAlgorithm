@echo off
setlocal enabledelayedexpansion

echo Step 1: Checking for existing virtual environment...
if exist "..\venv" (
    echo Removing existing virtual environment...
    rmdir /s /q "..\venv"
)

echo Step 2: Running setup_local_env.bat...
call "%~dp0setup_local_env.bat"

echo Step 3: Launching Streamlit app...
call "%~dp0run_streamlit_app.bat"

echo.
echo Note: The script will keep running until you press Ctrl+C to stop the Streamlit server
echo To deactivate the virtual environment after stopping, run: deactivate
echo.

endlocal 