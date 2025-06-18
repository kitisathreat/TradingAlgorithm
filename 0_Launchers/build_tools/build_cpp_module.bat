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
        set "VENV_PATH=./venv"
        echo [WARNING] LOCALAPPDATA not writable, using project directory: %VENV_PATH%
    )
) else (
    set "VENV_PATH=./venv"
    echo [INFO] LOCALAPPDATA not available, using project directory: %VENV_PATH%
)

set "VENV_ACTIVATE=%VENV_PATH%\Scripts\activate.bat"

echo Building and installing C++ trading engine module...

:: Activate virtual environment if it exists
if exist "%VENV_ACTIVATE%" (
    call "%VENV_ACTIVATE%"
) else (
    echo Virtual environment not found at: %VENV_PATH%
    echo Please run setup_local_env.bat first.
    exit /b 1
)

:: Change to C++ module directory
cd 1_High_Performance_Module_(C++)

:: Install required packages from consolidated requirements
echo Installing required packages...
cd ..
py -3.9 -m pip install -r requirements.txt
cd 1_High_Performance_Module_(C++)

:: Build and install the module
echo Building C++ module...
py -3.9 setup.py build_ext --inplace

:: Install the module
echo Installing module...
py -3.9 -m pip install -e .

:: Return to root directory
cd ..

echo C++ module build and installation complete!
echo You can now use the decision_engine module in Python. 