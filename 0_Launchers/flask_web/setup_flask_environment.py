#!/usr/bin/env python3
"""
Flask Web Environment Setup Script
Installs all required dependencies for the TradingAlgorithm Flask web interface
"""

import subprocess
import sys
import importlib
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"[INFO] {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is properly installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"[OK] {package_name} {version} is properly installed")
        return True
    except ImportError as e:
        print(f"[ERROR] {package_name} is not properly installed: {e}")
        return False

def main():
    print("=" * 80)
    print("                    FLASK WEB ENVIRONMENT SETUP")
    print("=" * 80)
    print()
    
    # Check Python version
    python_version = sys.version_info
    print(f"[INFO] Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major != 3 or python_version.minor < 9:
        print("[WARNING] Python 3.9+ is recommended for this application")
    
    # Upgrade pip
    if not run_command("python -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Core Flask dependencies
    print("\n[STEP 1] Installing core Flask dependencies...")
    core_deps = [
        "Flask==2.3.3",
        "Flask-SocketIO==5.3.6", 
        "python-socketio==5.8.0",
        "python-engineio==4.7.1",
        "Werkzeug==2.3.7"
    ]
    
    for dep in core_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
    
    # Async server dependencies (critical for eventlet)
    print("\n[STEP 2] Installing async server dependencies...")
    async_deps = [
        "eventlet==0.33.3",
        "gevent==23.9.1", 
        "gunicorn==21.2.0"
    ]
    
    for dep in async_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
    
    # ML and data processing
    print("\n[STEP 3] Installing ML and data processing dependencies...")
    ml_deps = [
        "tensorflow==2.13.0",
        "numpy==1.24.3",
        "pandas==2.0.3", 
        "scikit-learn==1.3.0",
        "joblib==1.3.0"
    ]
    
    for dep in ml_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
    
    # Trading and market data
    print("\n[STEP 4] Installing trading and market data dependencies...")
    trading_deps = [
        "yfinance==0.2.63",
        "websockets==13.0",
        "lxml>=4.9.1",
        "tqdm==4.65.0"
    ]
    
    for dep in trading_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
    
    # Sentiment analysis
    print("\n[STEP 5] Installing sentiment analysis dependencies...")
    sentiment_deps = [
        "vaderSentiment==3.3.2",
        "textblob==0.17.1"
    ]
    
    for dep in sentiment_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
    
    # System and performance
    print("\n[STEP 6] Installing system and performance dependencies...")
    system_deps = [
        "psutil==5.9.0",
        "tenacity==8.2.2"
    ]
    
    for dep in system_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
    
    # Security and utilities
    print("\n[STEP 7] Installing security and utilities...")
    util_deps = [
        "cryptography==41.0.7",
        "python-dotenv==1.0.0",
        "requests==2.31.0"
    ]
    
    for dep in util_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
    
    # Alpaca dependencies
    print("\n[STEP 8] Installing Alpaca dependencies...")
    alpaca_deps = [
        "deprecation==2.1.0",
        "msgpack==1.0.3",
        "PyYAML==6.0.1",
        "websocket-client>=0.56.0,<2",
        "urllib3>=1.25,<2"
    ]
    
    for dep in alpaca_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
    
    # Verify critical installations
    print("\n[STEP 9] Verifying critical installations...")
    critical_packages = [
        ("eventlet", "eventlet"),
        ("Flask-SocketIO", "flask_socketio"),
        ("Flask", "flask"),
        ("tensorflow", "tensorflow")
    ]
    
    all_verified = True
    for package_name, import_name in critical_packages:
        if not check_package(package_name, import_name):
            all_verified = False
    
    if not all_verified:
        print("\n[ERROR] Some critical packages failed verification")
        return False
    
    # Test eventlet specifically
    print("\n[STEP 10] Testing eventlet functionality...")
    try:
        import eventlet
        import flask_socketio
        
        # Test basic eventlet functionality
        eventlet.monkey_patch()
        print("[OK] eventlet monkey patching successful")
        
        # Test Flask-SocketIO with eventlet
        from flask import Flask
        app = Flask(__name__)
        socketio = flask_socketio.SocketIO(app, async_mode='eventlet')
        print("[OK] Flask-SocketIO with eventlet mode initialized successfully")
        
    except Exception as e:
        print(f"[ERROR] eventlet functionality test failed: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("                           SETUP COMPLETE")
    print("=" * 80)
    print()
    print("All dependencies have been installed successfully!")
    print()
    print("To run the Flask web application:")
    print("   1. Navigate to: 0_Launchers\\flask_web\\")
    print("   2. Run: python flask_app.py")
    print("   3. Open browser to: http://localhost:5000")
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n[ERROR] Setup failed. Please check the error messages above.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] Setup completed successfully!")
        input("Press Enter to exit...") 