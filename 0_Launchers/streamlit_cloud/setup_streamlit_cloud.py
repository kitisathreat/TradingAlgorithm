#!/usr/bin/env python3
"""
Streamlit Cloud Setup Script
Handles Alpaca websocket version conflicts during deployment
"""

import subprocess
import sys
import os

def install_with_override():
    """Install packages with version overrides to avoid conflicts"""
    
    print("🔧 Setting up Streamlit Cloud with Alpaca websocket override...")
    
    # Step 1: Install websockets first with the version we want
    print("📦 Installing websockets==13.0...")
    subprocess.run([sys.executable, "-m", "pip", "install", "websockets==13.0"], check=True)
    
    # Step 2: Install Alpaca without dependencies
    print("📦 Installing alpaca-trade-api without dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "alpaca-trade-api==3.2.0"], check=True)
    
    # Step 3: Install remaining dependencies
    print("📦 Installing remaining dependencies...")
    
    packages = [
        "tensorflow==2.13.0",
        "numpy==1.24.3", 
        "pandas==2.0.3",
        "scikit-learn==1.3.0",
        "joblib==1.3.0",
        "streamlit==1.31.0",
        "fastapi==0.100.0",
        "python-dotenv==1.0.0",
        "requests==2.31.0",
        "httpx==0.24.0",
        "yfinance==0.2.63",
        "lxml==4.9.1",
        "tqdm==4.65.0",
        "plotly==5.18.0",
        "matplotlib==3.7.0",
        "seaborn==0.12.0",
        "onnxruntime==1.15.0",
        "opencv-python-headless==4.7.0.72",
        "vaderSentiment==3.3.2",
        "textblob==0.17.1",
        "psutil==5.9.0",
        "tenacity==8.2.2",
        "streamlit-plotly-events==0.0.6",
        "streamlit-option-menu==0.3.12",
        "streamlit-extras==0.3.5"
    ]
    
    for package in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"⚠️ Failed to install {package}, trying without dependencies...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", package], check=True)
                print(f"✅ Installed {package} without dependencies")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package} even without dependencies")
    
    print("✅ Streamlit Cloud setup completed!")

if __name__ == "__main__":
    install_with_override() 