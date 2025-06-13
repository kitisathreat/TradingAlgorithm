@echo off
echo Starting Streamlit app...

:: Set Python 3.9 path
set PYTHON_PATH=C:\Users\KitKumar\AppData\Local\Programs\Python\Python39

:: Verify Python installation
if not exist "%PYTHON_PATH%\python.exe" (
    echo Error: Python 3.9 not found at %PYTHON_PATH%
    echo Please ensure Python 3.9 is installed correctly
    pause
    exit /b 1
)

:: Verify virtual environment
if not exist "..\venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found
    echo Please run setup_local_env.bat first to create the virtual environment
    pause
    exit /b 1
)

:: Change to script directory
cd /d "%~dp0"

:: Activate the virtual environment
echo Activating virtual environment...
call ..\venv\Scripts\activate.bat
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

:: Verify Streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Error: Streamlit is not installed
    echo Please run setup_local_env.bat to install dependencies
    pause
    exit /b 1
)

:: Run the Streamlit app
echo Starting Streamlit app...
streamlit run "..\3_Networking_and_User_Input\web_interface\streamlit_app.py"
if errorlevel 1 (
    echo Error: Failed to start Streamlit app
    pause
    exit /b 1
)

pause 