#!/usr/bin/env python3
"""
Streamlit Cloud Setup Script
Handles dependency conflicts and ensures compatibility
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"⚠️ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Starting Streamlit Cloud setup...")
    
    # Step 1: Upgrade pip
    run_command("pip install --upgrade pip", "Upgrading pip")
    
    # Step 2: Install websockets with compatible version first
    print("🔧 Installing compatible websockets version...")
    run_command("pip install 'websockets>=9.0,<11.0'", "Installing compatible websockets")
    
    # Step 3: Install alpaca-trade-api
    run_command("pip install alpaca-trade-api==3.2.0", "Installing Alpaca Trade API")
    
    # Step 4: Install TensorFlow with compatible version
    run_command("pip install tensorflow==2.13.0", "Installing TensorFlow")
    
    # Step 5: Install remaining requirements
    script_dir = Path(__file__).parent
    requirements_file = script_dir / "requirements.txt"
    
    if requirements_file.exists():
        run_command(f"pip install -r {requirements_file}", "Installing remaining requirements")
    else:
        print("⚠️ Requirements file not found")
    
    print("✅ Streamlit Cloud setup completed!")

if __name__ == "__main__":
    main() 