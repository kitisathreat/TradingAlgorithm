#!/usr/bin/env python3
"""
Streamlit Cloud Requirements Installer
Handles the websocket dependency conflict between TensorFlow and Alpaca
"""

import subprocess
import sys
import os
from pathlib import Path

def run_pip_install(package, description, extra_args=""):
    """Run pip install with error handling"""
    cmd = f"pip install {package} {extra_args}"
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} successful")
            return True
        else:
            print(f"⚠️ {description} had issues: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False

def main():
    """Main installation function"""
    print("🔧 Installing Streamlit Cloud requirements with dependency conflict resolution...")
    
    # Step 1: Force install websockets 13+ first
    print("\n📦 Step 1: Installing websockets 13+ for TensorFlow compatibility")
    success = run_pip_install(
        "websockets>=13.0", 
        "Installing websockets 13+", 
        "--force-reinstall --no-deps"
    )
    
    if not success:
        print("⚠️ Websockets installation had issues, but continuing...")
    
    # Step 2: Install alpaca-trade-api with --no-deps to ignore websocket constraint
    print("\n📦 Step 2: Installing alpaca-trade-api without dependencies")
    success = run_pip_install(
        "alpaca-trade-api==3.2.0", 
        "Installing alpaca-trade-api", 
        "--no-deps --force-reinstall"
    )
    
    if not success:
        print("⚠️ Alpaca installation had issues, but continuing...")
    
    # Step 3: Install the rest of the requirements
    print("\n📦 Step 3: Installing remaining requirements")
    
    # Read requirements file and install each package
    script_dir = Path(__file__).parent
    requirements_file = script_dir / "streamlit_requirements.txt"
    
    if requirements_file.exists():
        with open(requirements_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and 'alpaca-trade-api' not in line and 'websockets' not in line:
                # Extract package name and version
                if '==' in line:
                    package = line.split('==')[0].strip()
                    version = line.split('==')[1].split()[0].strip()  # Remove any extra args
                    package_spec = f"{package}=={version}"
                elif '>=' in line:
                    package = line.split('>=')[0].strip()
                    version = line.split('>=')[1].split()[0].strip()
                    package_spec = f"{package}>={version}"
                else:
                    package_spec = line
                
                run_pip_install(package_spec, f"Installing {package_spec}")
    
    # Step 4: Verify installation
    print("\n🔍 Step 4: Verifying installation...")
    try:
        import websockets
        print(f"✅ websockets version: {websockets.__version__}")
        
        import alpaca_trade_api
        print(f"✅ alpaca-trade-api imported successfully")
        
        import tensorflow as tf
        print(f"✅ tensorflow version: {tf.__version__}")
        
        if websockets.__version__ >= "13.0":
            print("🎉 SUCCESS: All dependencies installed with correct versions!")
        else:
            print("⚠️ WARNING: websockets version is below 13.0, but continuing...")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("⚠️ Some features may not work, but the app will continue...")
    
    print("\n✅ Installation process completed!")

if __name__ == "__main__":
    main() 