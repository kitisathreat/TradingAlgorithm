#!/usr/bin/env python3
"""
Streamlit Cloud Requirements Installer
Handles the websockets/alpaca-trade-api conflict by installing in the correct order.
"""

import subprocess
import sys
import os

def install_with_override():
    """Install requirements with websockets override for alpaca-trade-api compatibility."""
    
    print("🔧 Installing requirements with websockets override...")
    
    # Step 1: Install websockets 13.0 first (this establishes the version we want)
    print("📦 Step 1: Installing websockets==13.0...")
    subprocess.run([sys.executable, "-m", "pip", "install", "websockets==13.0"], check=True)
    
    # Step 2: Install alpaca-trade-api without its websockets dependency
    print("📦 Step 2: Installing alpaca-trade-api==3.2.0 without websockets dependency...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "alpaca-trade-api==3.2.0"], check=True)
    
    # Step 3: Install the rest of the requirements (excluding the conflicting ones)
    print("📦 Step 3: Installing remaining requirements...")
    
    # Read requirements.txt and filter out the conflicting packages
    with open("requirements.txt", "r") as f:
        requirements = f.readlines()
    
    # Filter out alpaca-trade-api and websockets (we already installed them)
    filtered_requirements = []
    for req in requirements:
        req = req.strip()
        if req and not req.startswith("#"):
            if not any(pkg in req for pkg in ["alpaca-trade-api", "websockets"]):
                filtered_requirements.append(req)
    
    # Install filtered requirements
    for req in filtered_requirements:
        try:
            print(f"📦 Installing {req}...")
            subprocess.run([sys.executable, "-m", "pip", "install", req], check=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Warning: Failed to install {req}: {e}")
    
    # Step 4: Force reinstall websockets to ensure version 13.0 is used
    print("📦 Step 4: Ensuring websockets==13.0 is active...")
    subprocess.run([sys.executable, "-m", "pip", "install", "websockets==13.0", "--force-reinstall"], check=True)
    
    print("✅ Requirements installation completed with websockets override!")

if __name__ == "__main__":
    install_with_override() 