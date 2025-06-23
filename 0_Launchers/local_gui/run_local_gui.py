#!/usr/bin/env python3
"""
Neural Network Trading System - Local GUI Launcher
Python executable version of run_local_gui.bat

This script:
1. Checks for Python 3.9
2. Detects and manages virtual environment location
3. Sets up the environment with required dependencies
4. Launches the local GUI application
"""

import os
import sys
import subprocess
import shutil
import tempfile
import platform
from pathlib import Path
import time

def print_header():
    """Print the application header"""
    print("=" * 80)
    print("           NEURAL NETWORK TRADING SYSTEM & FINANCIAL MODELER")
    print("                        LOCAL GUI SETUP")
    print("=" * 80)
    print()
    print("This setup will:")
    print("  • Create a Python virtual environment")
    print("  • Install all required dependencies")
    print("  • Launch the integrated trading system and financial modeler")
    print()
    print("Features:")
    print("  • 📈 Interactive trading system with neural network models")
    print("  • 📊 Excel-like financial modeler with 3-part financial models")
    print("  • 🎯 SEC filing data extraction and sensitivity analysis")
    print("  • 💾 Professional Excel export capabilities")
    print()
    print("-" * 80)
    print()

def print_section(title):
    """Print a section header"""
    print()
    print(f"[{title}]")
    print("-" * 40)

def check_python_version():
    """Check if Python version is compatible"""
    print_section("1/5")
    print("Checking Python version...")
    
    version = sys.version_info
    print(f"    Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3:
        print()
        print("[ERROR] Python 3 is required")
        print("        Please install Python 3.9 or later")
        print()
        input("Press Enter to exit...")
        return False
    
    if version.minor < 9:
        print()
        print("[WARNING] Python 3.9+ is recommended for optimal performance")
        print("          Current version may work but is not guaranteed")
        print()
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            return False
    
    print("[OK] Python version is compatible")
    return True

def detect_venv_path():
    """Detect the virtual environment path"""
    print_section("2/5")
    print("Detecting virtual environment location...")
    
    # Use a location accessible to all users (no admin rights required)
    if platform.system() == "Windows":
        # Windows: Use user's AppData folder
        venv_path = Path.home() / "AppData" / "Local" / "TradingAlgorithm" / "venv"
    else:
        # Unix/Linux: Use user's home directory
        venv_path = Path.home() / ".trading_algorithm" / "venv"
    
    print(f"    Virtual environment will be created at:")
    print(f"    {venv_path}")
    print()
    print("    This location:")
    print("    • Requires no administrator privileges")
    print("    • Is accessible to all users")
    print("    • Will be automatically managed")
    
    return venv_path

def remove_existing_venv(venv_path):
    """Remove existing virtual environment if it exists"""
    if venv_path.exists():
        print()
        print(f"Found existing virtual environment at: {venv_path}")
        print("Removing it to ensure a clean setup...")
        
        try:
            import shutil
            shutil.rmtree(venv_path)
            print("[OK] Existing virtual environment removed")
        except OSError as e:
            print()
            print("[ERROR] Could not remove existing virtual environment")
            print(f"        This may be due to insufficient permissions.")
            print(f"        Please manually delete: {venv_path}")
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
    
    # Install GUI dependencies
    print("    Installing GUI dependencies...")
    dependencies = [
        "numpy==1.23.5",
        "scikit-learn==1.3.0", 
        "PyQt5==5.15.9",
        "pyqtgraph==0.13.3",
        "pandas==2.0.3",
        "yfinance==0.2.63",
        "qt-material==2.14",
        "tensorflow==2.13.0"
    ]
    
    try:
        for dep in dependencies:
            print(f"        Installing {dep}...")
            subprocess.run([str(python_exe), "-m", "pip", "install", dep], 
                          check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print()
        print("[ERROR] Setup failed!")
        print()
        print("Common solutions:")
        print("   * Ensure you have a stable internet connection")
        print("   * Make sure Python 3.9 is properly installed")
        print("   * Check if antivirus is blocking the installation")
        print(f"   * Error: {e}")
        print()
        input("Press Enter to exit...")
        return False
    
    # Install Financial Modeler dependencies
    print("    Installing Financial Modeler dependencies...")
    financial_deps = [
        "openpyxl==3.1.2",  # Excel file handling for financial models
        "sec-api==0.0.1"    # SEC API for filing data
    ]
    
    try:
        for dep in financial_deps:
            print(f"        Installing {dep}...")
            subprocess.run([str(python_exe), "-m", "pip", "install", dep], 
                          check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print()
        print("[WARNING] Some Financial Modeler dependencies failed to install")
        print("          The trading system will still work, but financial modeling features may be limited")
        print(f"          Error: {e}")
        print()
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            return False
    
    print("[OK] Environment setup completed successfully")
    return True

def launch_gui(venv_path):
    """Launch the GUI application"""
    print()
    print("[4/5] Launching Neural Network Trading System & Financial Modeler...")
    print()
    print("=" * 80)
    print(" NOTES:")
    print(" * The GUI will open in a new window")
    print(" * Use tabs to switch between Trading System and Financial Modeler")
    print(" * Close the GUI window to exit the application")
    print(" * To deactivate the virtual environment after stopping, run: deactivate")
    print(f" * Virtual environment location: {venv_path}")
    print("=" * 80)
    print()
    
    # Determine Python executable in virtual environment
    if platform.system() == "Windows":
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"
    
    # Path to main.py
    script_dir = Path(__file__).parent
    main_py = script_dir.parent.parent / "_3_Networking_and_User_Input" / "local_gui" / "main.py"
    
    if not main_py.exists():
        print()
        print("[ERROR] GUI application file not found")
        print(f"        Expected location: {main_py}")
        print()
        input("Press Enter to exit...")
        return False
    
    try:
        subprocess.run([str(python_exe), str(main_py)], check=True)
    except subprocess.CalledProcessError as e:
        print()
        print("[ERROR] Failed to start GUI application")
        print(f"        Error: {e}")
        print()
        input("Press Enter to exit...")
        return False
    
    return True

def main():
    """Main function"""
    # Clear screen and set console width for better formatting
    if platform.system() == "Windows":
        os.system('cls')
        os.system('mode con: cols=100 lines=40')
    else:
        os.system('clear')
    
    print_header()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Detect virtual environment path
    venv_path = detect_venv_path()
    print(f"Virtual Environment Location: {venv_path}")
    print("(This location is accessible to all users and requires no admin rights)")
    print()
    print("-" * 80)
    print()
    
    # Remove existing virtual environment
    if not remove_existing_venv(venv_path):
        return 1
    
    # Create virtual environment directory
    if not create_venv_directory(venv_path):
        return 1
    
    # Set up environment
    if not setup_environment(venv_path):
        return 1
    
    # Launch GUI
    if not launch_gui(venv_path):
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n[INFO] Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Unexpected error: {e}")
        input("Press Enter to exit...")
        sys.exit(1) 