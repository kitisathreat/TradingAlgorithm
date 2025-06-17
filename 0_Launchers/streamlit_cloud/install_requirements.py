#!/usr/bin/env python3
"""
Custom installation script for Streamlit Cloud
Forces installation of packages without dependency conflicts
Specifically handles Alpaca websocket version requirements
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors gracefully"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"⚠️ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} error: {e}")
        return False

def install_package_without_deps(package, description):
    """Install a package without its dependencies"""
    cmd = f"{sys.executable} -m pip install --no-deps {package}"
    return run_command(cmd, f"Installing {description} without dependencies")

def install_package_force(package, description):
    """Force install a package, ignoring conflicts"""
    cmd = f"{sys.executable} -m pip install --force-reinstall --no-deps {package}"
    return run_command(cmd, f"Force installing {description}")

def main():
    print("🚀 Starting custom package installation for Streamlit Cloud...")
    
    # Core packages that need to be installed first
    core_packages = [
        ("tensorflow==2.13.0", "TensorFlow"),
        ("numpy==1.24.3", "NumPy"),
        ("pandas==2.0.3", "Pandas"),
        ("streamlit==1.31.0", "Streamlit"),
    ]
    
    # Install core packages first
    for package, description in core_packages:
        if not install_package_without_deps(package, description):
            print(f"⚠️ Failed to install {description}, trying force install...")
            install_package_force(package, description)
    
    # Install Alpaca with specific websocket version override
    print("🔧 Installing Alpaca with websocket version override...")
    
    # First, install the specific websocket version we want
    run_command(f"{sys.executable} -m pip install websockets==13.0", "Installing websockets 13.0")
    
    # Then install alpaca-trade-api without dependencies
    if not install_package_without_deps("alpaca-trade-api==3.2.0", "Alpaca Trade API"):
        print("⚠️ Alpaca installation failed, trying force install...")
        install_package_force("alpaca-trade-api==3.2.0", "Alpaca Trade API")
    
    # Install remaining packages
    remaining_packages = [
        ("scikit-learn==1.3.0", "Scikit-learn"),
        ("joblib==1.3.0", "Joblib"),
        ("fastapi==0.100.0", "FastAPI"),
        ("python-dotenv==1.0.0", "Python-dotenv"),
        ("requests==2.31.0", "Requests"),
        ("httpx==0.24.0", "HTTPX"),
        ("yfinance==0.2.63", "YFinance"),
        ("lxml==4.9.1", "LXML"),
        ("tqdm==4.65.0", "TQDM"),
        ("plotly==5.18.0", "Plotly"),
        ("matplotlib==3.7.0", "Matplotlib"),
        ("seaborn==0.12.0", "Seaborn"),
        ("onnxruntime==1.15.0", "ONNX Runtime"),
        ("opencv-python-headless==4.7.0.72", "OpenCV Headless"),
        ("vaderSentiment==3.3.2", "VADER Sentiment"),
        ("textblob==0.17.1", "TextBlob"),
        ("psutil==5.9.0", "PSUtil"),
        ("tenacity==8.2.2", "Tenacity"),
        ("streamlit-plotly-events==0.0.6", "Streamlit Plotly Events"),
        ("streamlit-option-menu==0.3.12", "Streamlit Option Menu"),
        ("streamlit-extras==0.3.5", "Streamlit Extras"),
    ]
    
    for package, description in remaining_packages:
        install_package_without_deps(package, description)
    
    print("✅ Custom installation completed!")
    print("📝 Note: Some dependency warnings may appear, but packages should work correctly.")

if __name__ == "__main__":
    main() 