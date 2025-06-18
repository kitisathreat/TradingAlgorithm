#!/usr/bin/env python3
"""
Neural Network Trading System - Flask Web Interface Launcher
Python executable version of run_flask_web.bat

This script:
1. Checks for Python 3.9
2. Detects and manages virtual environment location
3. Sets up the environment with required dependencies
4. Launches the Flask web interface
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path
import time

def print_header():
    """Print the application header"""
    print()
    print("=" * 80)
    print("                    NEURAL NETWORK TRADING SYSTEM - FLASK WEB INTERFACE")
    print("=" * 80)
    print()
    print(" This system trains an AI to mimic your trading decisions by combining:")
    print(" * Technical Analysis (RSI, MACD, Bollinger Bands)")
    print(" * Sentiment Analysis (VADER analysis of your reasoning)")
    print(" * Neural Network Learning (TensorFlow deep learning)")
    print()
    print(" Web interface will be available at: http://localhost:5000")
    print()

def print_section(title):
    """Print a section header"""
    print()
    print("-" * 80)
    print(f" {title}")
    print("-" * 80)
    print()

def check_python_version():
    """Check if Python 3.9 is available"""
    print("[1/5] Checking Python installation...")
    
    # Check current Python version
    current_version = sys.version_info
    if current_version.major == 3 and current_version.minor == 9:
        print("[OK] Python 3.9 found")
        return True
    
    # Try to find Python 3.9 using py launcher (Windows)
    if platform.system() == "Windows":
        try:
            result = subprocess.run(["py", "-3.9", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("[OK] Python 3.9 found via py launcher")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    print()
    print("[ERROR] Python 3.9 is required but not found")
    print("        Please install Python 3.9 from https://www.python.org/downloads/")
    print()
    input("Press Enter to exit...")
    return False

def detect_venv_path():
    """Detect the appropriate virtual environment path"""
    print("[INFO] Detecting environment for virtual environment path...")
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Check LOCALAPPDATA (Windows)
    local_app_data = os.environ.get('LOCALAPPDATA')
    if local_app_data:
        # Test if LOCALAPPDATA is writable
        test_file = Path(local_app_data) / "write_test.tmp"
        try:
            test_file.write_text("test")
            test_file.unlink()  # Remove test file
            venv_path = Path(local_app_data) / "TradingAlgorithm" / "venv"
            print(f"[OK] Using AppData location: {venv_path}")
            return venv_path
        except (OSError, PermissionError):
            venv_path = script_dir.parent / "venv"
            print(f"[WARNING] LOCALAPPDATA not writable, using project directory: {venv_path}")
            return venv_path
    else:
        venv_path = script_dir.parent / "venv"
        print(f"[INFO] LOCALAPPDATA not available, using project directory: {venv_path}")
        return venv_path

def remove_existing_venv(venv_path):
    """Remove existing virtual environment if it exists"""
    print()
    print("[2/5] Checking virtual environment...")
    
    if venv_path.exists():
        print("    Found existing virtual environment")
        print("    Attempting to remove...")
        
        try:
            shutil.rmtree(venv_path)
            print("[OK] Successfully removed existing virtual environment")
        except OSError as e:
            print()
            print("[ERROR] Could not remove existing virtual environment")
            print("        Please close any applications using the virtual environment and try again")
            print(f"        Error: {e}")
            print()
            input("Press Enter to exit...")
            return False
    else:
        print("[OK] No existing virtual environment found")
    
    return True

def create_venv_directory(venv_path):
    """Create parent directory for virtual environment if needed"""
    parent_dir = venv_path.parent
    if not parent_dir.exists():
        print(f"    Creating directory: {parent_dir}")
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print()
            print("[ERROR] Could not create directory")
            print(f"        This may be due to insufficient permissions or disk space.")
            print(f"        Error: {e}")
            print()
            input("Press Enter to exit...")
            return False
    return True

def setup_environment(venv_path):
    """Set up the Python environment with dependencies"""
    print()
    print("[3/5] Setting up Python environment...")
    print("    This may take a few minutes for first-time setup...")
    print("    Installing dependencies (this will show progress)...")
    
    # Create virtual environment
    print(f"    Creating virtual environment in: {venv_path}")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], 
                      check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print()
        print("[ERROR] Failed to create virtual environment")
        print(f"        Please check if you have write permissions to: {venv_path}")
        print(f"        Error: {e}")
        print()
        input("Press Enter to exit...")
        return False
    
    # Determine activation script path
    if platform.system() == "Windows":
        activate_script = venv_path / "Scripts" / "activate.bat"
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        activate_script = venv_path / "bin" / "activate"
        python_exe = venv_path / "bin" / "python"
    
    # Upgrade pip and install core packages
    print("    Upgrading pip and installing core packages...")
    try:
        subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True, text=True)
        subprocess.run([str(python_exe), "-m", "pip", "install", "wheel"], 
                      check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print()
        print("[ERROR] Failed to upgrade pip or install wheel")
        print(f"        Error: {e}")
        print()
        input("Press Enter to exit...")
        return False
    
    # Install Flask dependencies
    print("    Installing Flask web interface dependencies...")
    script_dir = Path(__file__).parent
    requirements_file = script_dir / "requirements.txt"
    
    if requirements_file.exists():
        try:
            subprocess.run([str(python_exe), "-m", "pip", "install", "-r", str(requirements_file)], 
                          check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print()
            print("[ERROR] Failed to install Flask dependencies")
            print(f"        Error: {e}")
            print()
            input("Press Enter to exit...")
            return False
    else:
        print("[WARNING] requirements.txt not found, installing basic Flask dependencies...")
        flask_deps = [
            "Flask==2.3.3",
            "Flask-SocketIO==5.3.6",
            "python-socketio==5.8.0",
            "python-engineio==4.7.1",
            "numpy==1.23.5",
            "pandas==2.0.3",
            "yfinance==0.2.63",
            "tensorflow==2.13.0",
            "scikit-learn==1.3.0",
            "vaderSentiment==3.3.2",
            "Werkzeug==2.3.7"
        ]
        
        try:
            for dep in flask_deps:
                print(f"        Installing {dep}...")
                subprocess.run([str(python_exe), "-m", "pip", "install", dep], 
                              check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print()
            print("[ERROR] Failed to install Flask dependencies")
            print(f"        Error: {e}")
            print()
            input("Press Enter to exit...")
            return False
    
    # Install additional dependencies from orchestrator
    print("    Installing ML dependencies...")
    repo_root = script_dir.parent.parent
    root_requirements = repo_root / "requirements.txt"
    
    if root_requirements.exists():
        try:
            subprocess.run([str(python_exe), "-m", "pip", "install", "-r", str(root_requirements)], 
                          check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("[WARNING] Failed to install root requirements, continuing anyway...")
    
    return True

def launch_flask_web(venv_path):
    """Launch the Flask web interface"""
    print()
    print("[4/5] Launching Flask web interface...")
    print()
    print("=" * 80)
    print("                    STARTING FLASK WEB INTERFACE")
    print("=" * 80)
    print()
    print(" Web interface will be available at: http://localhost:5000")
    print(" Press Ctrl+C to stop the server")
    print()
    print("=" * 80)
    print()
    
    # Determine Python executable path
    if platform.system() == "Windows":
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"
    
    # Change to script directory and launch Flask app
    script_dir = Path(__file__).parent
    flask_app = script_dir / "flask_app.py"
    
    if not flask_app.exists():
        print("[ERROR] Flask app not found")
        print(f"        Expected location: {flask_app}")
        print()
        input("Press Enter to exit...")
        return False
    
    try:
        # Change to script directory
        os.chdir(script_dir)
        
        # Launch Flask app
        subprocess.run([str(python_exe), str(flask_app)], check=True)
        
    except subprocess.CalledProcessError as e:
        print()
        print("[ERROR] Flask app failed to start")
        print(f"        Error: {e}")
        print()
        input("Press Enter to exit...")
        return False
    except KeyboardInterrupt:
        print()
        print("Flask web interface stopped by user.")
        return True
    
    return True

def main():
    """Main function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Detect virtual environment path
    venv_path = detect_venv_path()
    
    # Remove existing virtual environment
    if not remove_existing_venv(venv_path):
        return False
    
    # Create virtual environment directory
    if not create_venv_directory(venv_path):
        return False
    
    # Set up environment
    if not setup_environment(venv_path):
        return False
    
    # Launch Flask web interface
    return launch_flask_web(venv_path)

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nFlask web interface launcher stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        input("Press Enter to exit...")
        sys.exit(1) 