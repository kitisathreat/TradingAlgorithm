#!/usr/bin/env python3
"""
Streamlit Cloud Setup Script
Handles dependency installation with overrides for compatibility issues.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return None

def install_dependencies_with_overrides():
    """Install dependencies with overrides to handle conflicts."""
    
    print("🚀 Installing dependencies with websockets override...")
    
    # Method 1: Install websockets 13.0 first to establish the version
    run_command("pip install websockets==13.0", "Installing websockets 13.0")
    
    # Method 2: Install alpaca-trade-api without its websockets dependency
    run_command("pip install alpaca-trade-api==3.2.0 --no-deps", "Installing alpaca-trade-api without dependencies")
    
    # Method 3: Install remaining requirements using constraints
    run_command("pip install -r streamlit_requirements.txt --constraint constraints.txt", "Installing requirements with constraints")
    
    # Method 4: Force reinstall websockets to ensure version 13.0 is used
    run_command("pip install websockets==13.0 --force-reinstall", "Forcing websockets 13.0")

def verify_installation():
    """Verify that the correct versions are installed."""
    print("🔍 Verifying installation...")
    
    try:
        import websockets
        print(f"✅ websockets version: {websockets.__version__}")
        
        import alpaca_trade_api
        print(f"✅ alpaca-trade-api version: {alpaca_trade_api.__version__}")
        
        if websockets.__version__ == "13.0":
            print("✅ Websockets override successful!")
        else:
            print("⚠️ Warning: Websockets version is not 13.0")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")

def main():
    """Main setup function."""
    print("🔧 Streamlit Cloud Setup Script")
    print("=" * 50)
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Install dependencies with overrides
    install_dependencies_with_overrides()
    
    # Verify installation
    verify_installation()
    
    print("✅ Setup completed!")
    print("🚀 Ready to run Streamlit app")

if __name__ == "__main__":
    main() 