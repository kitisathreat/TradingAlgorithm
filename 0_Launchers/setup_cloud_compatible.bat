@echo off
setlocal enabledelayedexpansion

:: Cloud-compatible virtual environment setup
:: Automatically detects environment and uses appropriate paths

echo.
echo ================================================================================
echo                    CLOUD-COMPATIBLE ENVIRONMENT SETUP
echo ================================================================================
echo.

:: Detect environment type
echo [1/5] Detecting environment...

if "%OS%"=="Windows_NT" (
    echo [INFO] Windows environment detected
    set "IS_WINDOWS=1"
) else (
    echo [INFO] Non-Windows environment detected (Linux/macOS/Cloud)
    set "IS_WINDOWS=0"
)

:: Check for cloud-specific environment variables
if defined STREAMLIT_SERVER_RUNNING_ON_CLOUD (
    echo [INFO] Streamlit Cloud environment detected
    set "IS_CLOUD=1"
    set "CLOUD_TYPE=streamlit"
) else if defined HEROKU_APP_NAME (
    echo [INFO] Heroku environment detected
    set "IS_CLOUD=1"
    set "CLOUD_TYPE=heroku"
) else if defined AWS_LAMBDA_FUNCTION_NAME (
    echo [INFO] AWS Lambda environment detected
    set "IS_CLOUD=1"
    set "CLOUD_TYPE=aws_lambda"
) else if defined AZURE_FUNCTIONS_ENVIRONMENT (
    echo [INFO] Azure Functions environment detected
    set "IS_CLOUD=1"
    set "CLOUD_TYPE=azure"
) else if defined GCP_PROJECT (
    echo [INFO] Google Cloud environment detected
    set "IS_CLOUD=1"
    set "CLOUD_TYPE=gcp"
) else (
    echo [INFO] Local environment detected
    set "IS_CLOUD=0"
    set "CLOUD_TYPE=local"
)

:: Determine appropriate virtual environment path
echo.
echo [2/5] Determining virtual environment path...

if "%IS_CLOUD%"=="1" (
    echo [INFO] Cloud environment detected - using temporary directory
    if defined TMPDIR (
        set "VENV_PATH=%TMPDIR%\trading_algorithm_venv"
    ) else if defined TEMP (
        set "VENV_PATH=%TEMP%\trading_algorithm_venv"
    ) else (
        set "VENV_PATH=./venv"
    )
    echo [INFO] Using cloud path: %VENV_PATH%
) else if "%IS_WINDOWS%"=="1" (
    echo [INFO] Windows local environment - testing LOCALAPPDATA
    if defined LOCALAPPDATA (
        set "TEST_PATH=%LOCALAPPDATA%\TradingAlgorithm\venv"
        echo test > "%LOCALAPPDATA%\write_test.tmp" 2>nul
        if exist "%LOCALAPPDATA%\write_test.tmp" (
            del "%LOCALAPPDATA%\write_test.tmp" 2>nul
            set "VENV_PATH=%TEST_PATH%"
            echo [OK] LOCALAPPDATA is writable - using: %VENV_PATH%
        ) else (
            set "VENV_PATH=./venv"
            echo [WARNING] LOCALAPPDATA not writable - using: %VENV_PATH%
        )
    ) else (
        set "VENV_PATH=./venv"
        echo [WARNING] LOCALAPPDATA not available - using: %VENV_PATH%
    )
) else (
    echo [INFO] Non-Windows local environment - using relative path
    set "VENV_PATH=./venv"
)

set "VENV_ACTIVATE=%VENV_PATH%\Scripts\activate.bat"

:: Check Python availability
echo.
echo [3/5] Checking Python installation...

if "%IS_WINDOWS%"=="1" (
    py -3.9 --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python 3.9 not found via 'py' launcher
        echo        Trying 'python' command...
        python --version >nul 2>&1
        if errorlevel 1 (
            echo [ERROR] Python not found. Please install Python 3.9
            pause
            exit /b 1
        ) else (
            set "PYTHON_CMD=python"
            echo [OK] Python found via 'python' command
        )
    ) else (
        set "PYTHON_CMD=py -3.9"
        echo [OK] Python 3.9 found via 'py' launcher
    )
) else (
    python3 --version >nul 2>&1
    if errorlevel 1 (
        python --version >nul 2>&1
        if errorlevel 1 (
            echo [ERROR] Python not found. Please install Python 3.9
            pause
            exit /b 1
        ) else (
            set "PYTHON_CMD=python"
            echo [OK] Python found via 'python' command
        )
    ) else (
        set "PYTHON_CMD=python3"
        echo [OK] Python found via 'python3' command
    )
)

:: Create virtual environment
echo.
echo [4/5] Creating virtual environment...

if exist "%VENV_PATH%" (
    echo [INFO] Removing existing virtual environment...
    rmdir /s /q "%VENV_PATH%" 2>nul
)

:: Create parent directory if needed
for %%i in ("%VENV_PATH%") do set "PARENT_DIR=%%~dpi"
if not exist "%PARENT_DIR%" (
    echo [INFO] Creating parent directory: %PARENT_DIR%
    mkdir "%PARENT_DIR%" 2>nul
    if errorlevel 1 (
        echo [ERROR] Could not create directory: %PARENT_DIR%
        echo        Trying alternative location...
        set "VENV_PATH=./venv"
        set "VENV_ACTIVATE=%VENV_PATH%\Scripts\activate.bat"
    )
)

echo [INFO] Creating virtual environment at: %VENV_PATH%
%PYTHON_CMD% -m venv "%VENV_PATH%"
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    echo        This may be due to insufficient permissions or disk space
    pause
    exit /b 1
)

:: Install dependencies
echo.
echo [5/5] Installing dependencies...

if "%IS_WINDOWS%"=="1" (
    call "%VENV_ACTIVATE%"
    if errorlevel 1 (
        echo [ERROR] Failed to activate virtual environment
        pause
        exit /b 1
    )
    
    %PYTHON_CMD% -m pip install --upgrade pip
    %PYTHON_CMD% -m pip install wheel
    %PYTHON_CMD% -m pip install "numpy==1.23.5" "scikit-learn==1.3.0" "PyQt5==5.15.9" "pyqtgraph==0.13.3" "pandas==2.0.3" "yfinance==0.2.36" "qt-material==2.14" "tensorflow==2.13.0"
) else (
    echo [INFO] Non-Windows environment - using pip directly
    %PYTHON_CMD% -m pip install --upgrade pip
    %PYTHON_CMD% -m pip install wheel
    %PYTHON_CMD% -m pip install "numpy==1.23.5" "scikit-learn==1.3.0" "pandas==2.0.3" "yfinance==0.2.36" "tensorflow==2.13.0"
)

if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

:: Success message
echo.
echo ================================================================================
echo                              SETUP COMPLETE
echo ================================================================================
echo.
echo Environment Details:
echo   Type: %CLOUD_TYPE%
echo   Path: %VENV_PATH%
echo   Python: %PYTHON_CMD%
echo   Status: Ready to use
echo.
echo For other scripts, use these environment variables:
echo   VENV_PATH=%VENV_PATH%
echo   VENV_ACTIVATE=%VENV_ACTIVATE%
echo   PYTHON_CMD=%PYTHON_CMD%
echo.
echo ================================================================================
echo.
pause 