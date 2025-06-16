@echo off
setlocal enabledelayedexpansion

:: Set virtual environment path to AppData
set "VENV_PATH=%LOCALAPPDATA%\TradingAlgorithm\venv"
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

:: Check for Python 3.9
echo [1/4] Checking Python installation...
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

:: Check and remove existing virtual environment
echo.
echo [2/4] Checking virtual environment...
if exist "%VENV_PATH%" (
    echo    Found existing virtual environment
    echo    Attempting to remove...
    
    :: Remove virtual environment (no admin rights needed in AppData)
    rmdir /s /q "%VENV_PATH%" 2>nul
    if exist "%VENV_PATH%" (
        echo.
        echo [ERROR] Could not remove existing virtual environment
        echo        Please close any applications using the virtual environment and try again
        echo.
        pause
        exit /b 1
    )
    echo [OK] Successfully removed existing virtual environment
) else (
    echo [OK] No existing virtual environment found
)

:: Create AppData directory if it doesn't exist
if not exist "%LOCALAPPDATA%\TradingAlgorithm" (
    mkdir "%LOCALAPPDATA%\TradingAlgorithm"
)

:: Set up new environment
echo.
echo [3/4] Setting up Python environment...
echo    This may take 5-10 minutes for first-time setup...
echo    Installing dependencies (this will show progress)...

:: Create and activate virtual environment
echo    Creating virtual environment in: %VENV_PATH%
python -m venv "%VENV_PATH%"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to create virtual environment
    echo        Please check if you have write permissions to: %LOCALAPPDATA%
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
echo    Upgrading pip and installing core packages...
python -m pip install --upgrade pip
python -m pip install wheel
echo    Installing project dependencies...
python -m pip install -r "%~dp0..\_3_Networking_and_User_Input\web_interface\web_requirements.txt"

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

:: Launch Streamlit
echo.
echo [4/4] Launching Neural Network Training Interface...
echo.
call "%~dp0run_streamlit_app.bat"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start Streamlit interface
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
echo    * The script will keep running until you press Ctrl+C to stop the Streamlit server
echo    * To deactivate the virtual environment after stopping, run: deactivate
echo    * Virtual environment location: %VENV_PATH%
echo.
echo --------------------------------------------------------------------------------
echo.

endlocal 