@echo off
echo ================================================================================
echo                    FLASK WEB ENVIRONMENT SETUP
echo ================================================================================
echo.
echo This script will set up the Flask web environment with all required dependencies.
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9.x and try again
    pause
    exit /b 1
)

echo [INFO] Python version:
python --version

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip is not available
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)

echo [INFO] pip version:
pip --version

echo.
echo [STEP 1] Upgrading pip to latest version...
python -m pip install --upgrade pip

echo.
echo [STEP 2] Installing core dependencies...
pip install Flask==2.3.3
pip install Flask-SocketIO==5.3.6
pip install python-socketio==5.8.0
pip install python-engineio==4.7.1
pip install Werkzeug==2.3.7

echo.
echo [STEP 3] Installing async server dependencies...
pip install eventlet==0.33.3
pip install gevent==23.9.1
pip install gunicorn==21.2.0

echo.
echo [STEP 4] Installing ML and data processing dependencies...
pip install tensorflow==2.13.0
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install scikit-learn==1.3.0
pip install joblib==1.3.0

echo.
echo [STEP 5] Installing trading and market data dependencies...
pip install yfinance==0.2.63
pip install websockets==13.0
pip install lxml>=4.9.1
pip install tqdm==4.65.0

echo.
echo [STEP 6] Installing sentiment analysis dependencies...
pip install vaderSentiment==3.3.2
pip install textblob==0.17.1

echo.
echo [STEP 7] Installing system and performance dependencies...
pip install psutil==5.9.0
pip install tenacity==8.2.2

echo.
echo [STEP 8] Installing security and utilities...
pip install cryptography==41.0.7
pip install python-dotenv==1.0.0
pip install requests==2.31.0

echo.
echo [STEP 9] Installing Alpaca dependencies...
pip install deprecation==2.1.0
pip install msgpack==1.0.3
pip install PyYAML==6.0.1
pip install "websocket-client>=0.56.0,<2"
pip install "urllib3>=1.25,<2"

echo.
echo [STEP 10] Verifying eventlet installation...
python -c "import eventlet; print(f'[OK] eventlet {eventlet.__version__} installed successfully')"

echo.
echo [STEP 11] Verifying Flask-SocketIO installation...
python -c "import flask_socketio; print(f'[OK] Flask-SocketIO {flask_socketio.__version__} installed successfully')"

echo.
echo ================================================================================
echo                           SETUP COMPLETE
echo ================================================================================
echo.
echo All dependencies have been installed successfully!
echo.
echo To run the Flask web application:
echo   1. Navigate to: 0_Launchers\flask_web\
echo   2. Run: python flask_app.py
echo   3. Open browser to: http://localhost:5000
echo.
echo Press any key to exit...
pause >nul 