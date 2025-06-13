@echo off
echo Setting up local Python environment for Trading Algorithm...

:: Set Python 3.9 path
set PYTHON_PATH=C:\Users\KitKumar\AppData\Local\Programs\Python\Python39
set PATH=%PYTHON_PATH%;%PYTHON_PATH%\Scripts;%PATH%

:: Check if Python 3.9 is available
%PYTHON_PATH%\python.exe --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.9 is not found at %PYTHON_PATH%
    echo Please install Python 3.9.x from https://www.python.org/downloads/release/python-3913/
    exit /b 1
)

:: Check Python version
for /f "tokens=2" %%I in ('%PYTHON_PATH%\python.exe --version 2^>^&1') do set PYTHON_VERSION=%%I
echo Detected Python version: %PYTHON_VERSION%

:: Verify Python version is exactly 3.9.x
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    if not "%%a"=="3" (
        echo Error: Python 3.9.x is required, but found version %PYTHON_VERSION%
        echo Please install Python 3.9.x from https://www.python.org/downloads/release/python-3913/
        exit /b 1
    )
    if not "%%b"=="9" (
        echo Error: Python 3.9.x is required, but found version %PYTHON_VERSION%
        echo Please install Python 3.9.x from https://www.python.org/downloads/release/python-3913/
        exit /b 1
    )
)

:: Create and activate virtual environment in root directory
if not exist "..\venv" (
    echo Creating virtual environment in root directory...
    %PYTHON_PATH%\python.exe -m venv "..\venv"
)

:: Activate virtual environment from root
call "..\venv\Scripts\activate.bat"

:: Upgrade pip and install wheel
echo Upgrading pip and installing wheel...
python -m pip install --upgrade pip
python -m pip install --upgrade wheel

:: Install packages from root_requirements.txt
echo Installing dependencies from root_requirements.txt...
python -m pip install -r ..\root_requirements.txt

echo.
echo Environment setup complete. You can now run your Streamlit app with:
echo streamlit run streamlit_app.py
echo.
pause 