@echo off
setlocal enabledelayedexpansion

:: Check for Python 3.9 using py launcher
py -3.9 --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python 3.9 is required but not found or not registered with the 'py' launcher.
    echo        Please install Python 3.9 and ensure it is registered with the 'py' launcher.
    echo        Download: https://www.python.org/downloads/release/python-390/
    echo.
    pause
    exit /b 1
)

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
echo                    ADVANCED NEURAL NETWORK TRADING ALGORITHM
echo ================================================================================
echo.
echo  This system trains an AI to mimic your trading decisions by combining:
echo  * Technical Analysis (RSI, MACD, Bollinger Bands)
echo  * Sentiment Analysis (VADER analysis of your reasoning)
echo  * Neural Network Learning (TensorFlow deep learning)
echo.
echo --------------------------------------------------------------------------------
echo.

:: Create and activate virtual environment
echo [1/4] Creating virtual environment in: %VENV_PATH%

:: Create parent directory if needed (only for AppData path)
if "%VENV_PATH%"=="%LOCALAPPDATA%\TradingAlgorithm\venv" (
    if not exist "%LOCALAPPDATA%\TradingAlgorithm" (
        echo    Creating directory: %LOCALAPPDATA%\TradingAlgorithm
        mkdir "%LOCALAPPDATA%\TradingAlgorithm"
        if errorlevel 1 (
            echo.
            echo [ERROR] Could not create directory: %LOCALAPPDATA%\TradingAlgorithm
            echo        This may be due to insufficient permissions or disk space.
            echo.
            pause
            exit /b 1
        )
    )
)

py -3.9 -m venv "%VENV_PATH%"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to create virtual environment
    echo        Please check if you have write permissions to: %VENV_PATH%
    echo.
    pause
    exit /b 1
)

:: Activate virtual environment and install dependencies
call "%VENV_ACTIVATE%"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to activate virtual environment
    echo        Please check if the virtual environment was created correctly
    echo.
    pause
    exit /b 1
)

:: Install dependencies
echo [2/4] Upgrading pip and installing core packages...
py -3.9 -m pip install --upgrade pip
py -3.9 -m pip install wheel
py -3.9 -m pip install "numpy==1.23.5" "scikit-learn==1.3.0" "PyQt5==5.15.9" "pyqtgraph==0.13.3" "pandas==2.0.3" "yfinance==0.2.63" "qt-material==2.14" "tensorflow==2.13.0"

echo [3/4] Installing project dependencies...

:: Install packages with conflicting dependencies separately
echo    Installing alpaca-trade-api with dependency override...
py -3.9 -m pip install alpaca-trade-api==3.2.0 --no-deps

echo    Installing yfinance with dependency override...
py -3.9 -m pip install yfinance==0.2.63 --no-deps

:: Install websockets 13.0 explicitly since we bypassed dependency checking
echo    Installing websockets 13.0 explicitly...
py -3.9 -m pip install websockets==13.0

:: Install remaining dependencies normally
echo    Installing remaining dependencies...
py -3.9 -m pip install -r "%~dp0..\..\requirements.txt"

if errorlevel 1 (
    echo.
    echo [ERROR] Setup failed!
    echo.
    echo Common solutions:
    echo    * Ensure you have a stable internet connection
    echo    * Make sure Python 3.9 is properly installed
    echo    * Check if antivirus is blocking the installation
    echo.
    pause
    exit /b 1
)

echo [OK] Environment setup completed successfully

:: Launch Flask Web Interface
echo.
echo [4/4] Launching Neural Network Training Interface...
echo.
call "%~dp0..\flask_web\run_flask_web.py"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start Flask web interface
    echo        Please check the error messages above
    echo.
    pause
    exit /b 1
)

:: Display quick start guide
echo.
echo ================================================================================
echo                                QUICK START GUIDE
echo ================================================================================
echo.
echo  HOW TO USE THE SYSTEM:
echo.
echo    1. Click "Get New Stock" to see a random S&P100 stock
echo    2. Analyze the charts and technical indicators
echo    3. Make your trading decision (BUY/SELL/HOLD)
echo    4. Explain your reasoning in detail
echo    5. Submit to train the neural network
echo    6. Repeat 10+ times, then train the neural network
echo    7. Test the AI's predictions on new stocks!
echo.
echo  THE AI LEARNS FROM:
echo    * Your technical analysis preferences
echo    * Keywords and sentiment in your explanations
echo    * Pattern recognition in your decision-making
echo.
echo  FEATURES:
echo    * 25 years of historical S&P100 data
echo    * Interactive candlestick charts
echo    * Advanced technical indicators
echo    * Real-time sentiment analysis
echo    * Neural network training progress
echo.
echo  NOTES:
echo    * The script will keep running until you press Ctrl+C to stop the Flask server
echo    * To deactivate the virtual environment after stopping, run: deactivate
echo    * Virtual environment location: %VENV_PATH%
echo.
echo --------------------------------------------------------------------------------
echo.

endlocal 