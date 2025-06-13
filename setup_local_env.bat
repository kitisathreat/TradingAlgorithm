@echo off
echo Setting up local Python environment for Trading Algorithm...

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    exit /b 1
)

:: Check Python version
for /f "tokens=2" %%I in ('python --version 2^>^&1') do set PYTHON_VERSION=%%I
echo Detected Python version: %PYTHON_VERSION%

:: Verify Python version is <= 3.11 for TensorFlow compatibility
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    if %%a GTR 3 (
        echo Python version must be <= 3.11 for TensorFlow compatibility
        exit /b 1
    )
    if %%a EQU 3 if %%b GTR 11 (
        echo Python version must be <= 3.11 for TensorFlow compatibility
        exit /b 1
    )
)

:: Create and activate virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Upgrade pip and install wheel
echo Upgrading pip and installing wheel...
python -m pip install --upgrade pip
python -m pip install --upgrade wheel

:: Install packages in parallel using pip-tools
echo Installing pip-tools for parallel installation...
python -m pip install pip-tools

:: Create a temporary requirements file with direct URLs for large packages
echo Creating optimized requirements file...
(
    echo tensorflow-cpu>=2.15.0,^<2.20.0
    echo numpy>=1.24.0,^<2.0.0
    echo pandas>=2.0.0
    echo streamlit>=1.31.0
    echo plotly>=5.18.0
    echo python-dotenv>=1.0.0
    echo requests>=2.31.0
    echo alpaca-trade-api>=3.0.0
    echo yfinance>=0.2.36
    echo psutil
    echo tqdm>=4.65.0
    echo lxml>=4.9.0
    echo vaderSentiment>=3.3.2
    echo tenacity>=8.2.2
    echo pytest>=7.4.0
    echo pybind11>=2.6.0
    echo setuptools>=65.0.0
) > requirements.tmp

:: Install packages using pip-tools with parallel processing
echo Installing packages (this may take a while)...
python -m piptools compile requirements.tmp -o requirements.lock
python -m piptools sync requirements.lock --parallel

:: Clean up
del requirements.tmp
del requirements.lock

echo Setup complete! Virtual environment is ready.
echo To activate the environment, run: venv\Scripts\activate.bat

REM Optional: Disk write speed test (writes a 100MB file and times it)
echo Testing disk write speed...
set start=%time%
fsutil file createnew testfile.tmp 104857600 >nul
set end=%time%
echo Disk write test file created (100MB).
del testfile.tmp
echo Start time: %start%
echo End time:   %end%
echo.

REM Record start time for pip install
echo Installing dependencies...
set pipstart=%time%

pip install -r requirements.txt

set pipend=%time%
echo.
echo pip install started at: %pipstart%
echo pip install ended at:   %pipend%

echo.
echo Environment setup complete. You can now run your Streamlit app with:
echo streamlit run streamlit_app.py
echo.
pause 