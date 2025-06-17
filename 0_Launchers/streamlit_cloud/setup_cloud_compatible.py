#!/usr/bin/env python3
"""
Cloud-compatible virtual environment setup script
Automatically detects environment and uses appropriate paths
Python executable version of setup_cloud_compatible.bat
"""

import os
import sys
import subprocess
import shutil
import platform
import tempfile
from pathlib import Path


def print_header(title):
    """Print a formatted header"""
    print()
    print("=" * 80)
    print(f"                    {title}")
    print("=" * 80)
    print()


def print_step(step_num, total_steps, description):
    """Print a step description"""
    print(f"[{step_num}/{total_steps}] {description}...")


def print_info(message):
    """Print an info message"""
    print(f"[INFO] {message}")


def print_ok(message):
    """Print a success message"""
    print(f"[OK] {message}")


def print_warning(message):
    """Print a warning message"""
    print(f"[WARNING] {message}")


def print_error(message):
    """Print an error message"""
    print(f"[ERROR] {message}")


def detect_environment():
    """Detect the current environment type"""
    print_step(1, 5, "Detecting environment")
    
    # Detect OS
    is_windows = platform.system() == "Windows"
    if is_windows:
        print_info("Windows environment detected")
    else:
        print_info("Non-Windows environment detected (Linux/macOS/Cloud)")
    
    # Detect cloud environment
    cloud_vars = {
        'STREAMLIT_SERVER_RUNNING_ON_CLOUD': 'streamlit',
        'HEROKU_APP_NAME': 'heroku',
        'AWS_LAMBDA_FUNCTION_NAME': 'aws_lambda',
        'AZURE_FUNCTIONS_ENVIRONMENT': 'azure',
        'GCP_PROJECT': 'gcp'
    }
    
    is_cloud = False
    cloud_type = 'local'
    
    for var, cloud_name in cloud_vars.items():
        if os.environ.get(var):
            print_info(f"{cloud_name.title()} environment detected")
            is_cloud = True
            cloud_type = cloud_name
            break
    
    if not is_cloud:
        print_info("Local environment detected")
    
    return is_windows, is_cloud, cloud_type


def determine_venv_path(is_windows, is_cloud):
    """Determine the appropriate virtual environment path"""
    print_step(2, 5, "Determining virtual environment path")
    
    if is_cloud:
        print_info("Cloud environment detected - using temporary directory")
        if 'TMPDIR' in os.environ:
            venv_path = Path(os.environ['TMPDIR']) / "trading_algorithm_venv"
        elif 'TEMP' in os.environ:
            venv_path = Path(os.environ['TEMP']) / "trading_algorithm_venv"
        else:
            venv_path = Path("./venv")
        print_info(f"Using cloud path: {venv_path}")
        
    elif is_windows:
        print_info("Windows local environment - testing LOCALAPPDATA")
        localappdata = os.environ.get('LOCALAPPDATA')
        if localappdata:
            test_path = Path(localappdata) / "TradingAlgorithm" / "venv"
            test_file = Path(localappdata) / "write_test.tmp"
            
            try:
                # Test if we can write to LOCALAPPDATA
                test_file.write_text("test")
                test_file.unlink()  # Remove test file
                venv_path = test_path
                print_ok(f"LOCALAPPDATA is writable - using: {venv_path}")
            except (OSError, PermissionError):
                venv_path = Path("./venv")
                print_warning(f"LOCALAPPDATA not writable - using: {venv_path}")
        else:
            venv_path = Path("./venv")
            print_warning("LOCALAPPDATA not available - using: ./venv")
    else:
        print_info("Non-Windows local environment - using relative path")
        venv_path = Path("./venv")
    
    return venv_path


def check_python_installation(is_windows):
    """Check Python availability and return the appropriate command"""
    print_step(3, 5, "Checking Python installation")
    
    python_commands = []
    
    if is_windows:
        python_commands = [
            ["py", "-3.9"],
            ["python"],
            ["python3"]
        ]
    else:
        python_commands = [
            ["python3"],
            ["python"]
        ]
    
    for cmd in python_commands:
        try:
            result = subprocess.run(cmd + ["--version"], 
                                  capture_output=True, text=True, check=True)
            python_cmd = cmd
            print_ok(f"Python found via '{' '.join(cmd)}' command")
            return python_cmd
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    print_error("Python not found. Please install Python 3.9")
    sys.exit(1)


def create_virtual_environment(venv_path, python_cmd):
    """Create the virtual environment"""
    print_step(4, 5, "Creating virtual environment")
    
    # Remove existing virtual environment if it exists
    if venv_path.exists():
        print_info("Removing existing virtual environment...")
        try:
            shutil.rmtree(venv_path)
        except OSError as e:
            print_warning(f"Could not remove existing venv: {e}")
    
    # Create parent directory if needed
    parent_dir = venv_path.parent
    if not parent_dir.exists():
        print_info(f"Creating parent directory: {parent_dir}")
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print_error(f"Could not create directory: {parent_dir}")
            print_info("Trying alternative location...")
            venv_path = Path("./venv")
    
    print_info(f"Creating virtual environment at: {venv_path}")
    try:
        subprocess.run(python_cmd + ["-m", "venv", str(venv_path)], check=True)
    except subprocess.CalledProcessError as e:
        print_error("Failed to create virtual environment")
        print_error("This may be due to insufficient permissions or disk space")
        sys.exit(1)


def get_activate_script(venv_path, is_windows):
    """Get the path to the virtual environment activation script"""
    if is_windows:
        return venv_path / "Scripts" / "activate.bat"
    else:
        return venv_path / "bin" / "activate"


def install_dependencies(venv_path, python_cmd, is_windows):
    """Install the required dependencies"""
    print_step(5, 5, "Installing dependencies")
    
    # Get the Python executable from the virtual environment
    if is_windows:
        venv_python = venv_path / "Scripts" / "python.exe"
    else:
        venv_python = venv_path / "bin" / "python"
    
    # Upgrade pip
    try:
        subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(venv_python), "-m", "pip", "install", "wheel"], check=True)
    except subprocess.CalledProcessError as e:
        print_error("Failed to upgrade pip or install wheel")
        sys.exit(1)
    
    # Install dependencies based on platform
    if is_windows:
        packages = [
            "numpy==1.23.5",
            "scikit-learn==1.3.0", 
            "PyQt5==5.15.9",
            "pyqtgraph==0.13.3",
            "pandas==2.0.3",
            "yfinance==0.2.63",
            "qt-material==2.14",
            "tensorflow==2.13.0"
        ]
    else:
        packages = [
            "numpy==1.23.5",
            "scikit-learn==1.3.0",
            "pandas==2.0.3", 
            "yfinance==0.2.63",
            "tensorflow==2.13.0"
        ]
    
    try:
        subprocess.run([str(venv_python), "-m", "pip", "install"] + packages, check=True)
    except subprocess.CalledProcessError as e:
        print_error("Failed to install dependencies")
        sys.exit(1)


def main():
    """Main function"""
    print_header("CLOUD-COMPATIBLE ENVIRONMENT SETUP")
    
    # Detect environment
    is_windows, is_cloud, cloud_type = detect_environment()
    
    # Determine virtual environment path
    venv_path = determine_venv_path(is_windows, is_cloud)
    
    # Check Python installation
    python_cmd = check_python_installation(is_windows)
    
    # Create virtual environment
    create_virtual_environment(venv_path, python_cmd)
    
    # Install dependencies
    install_dependencies(venv_path, python_cmd, is_windows)
    
    # Get activation script path
    activate_script = get_activate_script(venv_path, is_windows)
    
    # Success message
    print_header("SETUP COMPLETE")
    print("Environment Details:")
    print(f"  Type: {cloud_type}")
    print(f"  Path: {venv_path}")
    print(f"  Python: {' '.join(python_cmd)}")
    print("  Status: Ready to use")
    print()
    print("For other scripts, use these environment variables:")
    print(f"  VENV_PATH={venv_path}")
    print(f"  VENV_ACTIVATE={activate_script}")
    print(f"  PYTHON_CMD={' '.join(python_cmd)}")
    print()
    print("=" * 80)
    print()
    
    # Wait for user input on Windows
    if is_windows:
        input("Press Enter to continue...")


if __name__ == "__main__":
    main() 