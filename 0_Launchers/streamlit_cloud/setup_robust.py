#!/usr/bin/env python3
"""
Streamlit Cloud Setup Script - Robust Multi-Approach
Handles dependency installation with multiple fallback approaches for compatibility issues.
"""

import subprocess
import sys
import os
import logging
from pathlib import Path
from typing import Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(cmd, description, check=True):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True, result.stdout
        else:
            print(f"❌ {description} failed:")
            print(f"Error: {result.stderr}")
            return False, result.stderr
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False, e.stderr

def try_install_with_constraints() -> bool:
    """Try installing with constraints file approach."""
    try:
        script_dir = Path(__file__).parent
        constraints_file = script_dir / "constraints.txt"
        requirements_file = script_dir / "streamlit_requirements.txt"
        
        if not constraints_file.exists() or not requirements_file.exists():
            return False
        
        print("🔄 Attempting installation with constraints file...")
        cmd = f"pip install -r {requirements_file} --constraint {constraints_file} --force-reinstall"
        success, _ = run_command(cmd, "Installing with constraints")
        return success
        
    except Exception as e:
        print(f"⚠️ Constraints approach failed: {e}")
        return False

def try_install_with_no_deps() -> bool:
    """Try installing with --no-deps approach."""
    try:
        print("🔄 Attempting installation with --no-deps approach...")
        
        # First install websockets 13+
        success1, _ = run_command("pip install websockets>=13.0 --force-reinstall", "Installing websockets 13+")
        if not success1:
            return False
        
        # Then install alpaca with --no-deps
        success2, _ = run_command("pip install alpaca-trade-api==3.2.0 --no-deps", "Installing alpaca-trade-api without dependencies")
        if not success2:
            return False
        
        # Finally install the rest
        script_dir = Path(__file__).parent
        requirements_file = script_dir / "streamlit_requirements.txt"
        
        success3, _ = run_command(f"pip install -r {requirements_file}", "Installing remaining requirements")
        return success3
        
    except Exception as e:
        print(f"⚠️ --no-deps approach failed: {e}")
        return False

def try_install_with_override() -> bool:
    """Try installing with override requirements file."""
    try:
        script_dir = Path(__file__).parent
        override_file = script_dir / "streamlit_requirements_override.txt"
        
        if not override_file.exists():
            return False
        
        print("🔄 Attempting installation with override requirements...")
        
        # Install websockets first
        success1, _ = run_command("pip install websockets>=13.0", "Installing websockets 13+")
        if not success1:
            return False
        
        # Install alpaca with --no-deps
        success2, _ = run_command("pip install alpaca-trade-api==3.2.0 --no-deps", "Installing alpaca-trade-api without dependencies")
        if not success2:
            return False
        
        # Install the rest from override file (excluding alpaca line)
        with open(override_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#') and 'alpaca-trade-api' not in line]
        
        for line in lines:
            if line and not line.startswith('#'):
                success, _ = run_command(f"pip install {line}", f"Installing {line}")
                if not success:
                    return False
        
        return True
        
    except Exception as e:
        print(f"⚠️ Override approach failed: {e}")
        return False

def try_install_with_force() -> bool:
    """Try installing with --force-reinstall and ignore conflicts."""
    try:
        print("🔄 Attempting installation with force approach...")
        
        # Force install websockets 13+
        success1, _ = run_command("pip install websockets>=13.0 --force-reinstall --no-deps", "Force installing websockets 13+")
        if not success1:
            return False
        
        # Force install alpaca ignoring dependencies
        success2, _ = run_command("pip install alpaca-trade-api==3.2.0 --force-reinstall --no-deps", "Force installing alpaca-trade-api")
        if not success2:
            return False
        
        # Install the rest normally
        script_dir = Path(__file__).parent
        requirements_file = script_dir / "streamlit_requirements.txt"
        
        with open(requirements_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#') and 'alpaca-trade-api' not in line]
        
        for line in lines:
            if line and not line.startswith('#'):
                success, _ = run_command(f"pip install {line} --force-reinstall", f"Force installing {line}")
                if not success:
                    return False
        
        return True
        
    except Exception as e:
        print(f"⚠️ Force approach failed: {e}")
        return False

def install_dependencies_robust():
    """Install dependencies using multiple approaches with fallbacks."""
    
    print("🚀 Installing dependencies with robust multi-approach method...")
    
    approaches = [
        ("Constraints", try_install_with_constraints),
        ("No-deps", try_install_with_no_deps),
        ("Override", try_install_with_override),
        ("Force", try_install_with_force)
    ]
    
    for name, approach in approaches:
        print(f"\n🔄 Trying {name} approach...")
        if approach():
            print(f"✅ Successfully installed requirements using {name} approach")
            return True
    
    print("❌ All installation approaches failed")
    return False

def verify_installation():
    """Verify that the correct versions are installed."""
    print("\n🔍 Verifying installation...")
    
    try:
        import websockets
        print(f"✅ websockets version: {websockets.__version__}")
        
        import alpaca_trade_api
        print(f"✅ alpaca-trade-api version: {alpaca_trade_api.__version__}")
        
        if websockets.__version__ >= "13.0":
            print("✅ Websockets override successful!")
        else:
            print("⚠️ Warning: Websockets version is below 13.0")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

def main():
    """Main setup function."""
    print("🔧 Streamlit Cloud Setup Script - Robust Multi-Approach")
    print("=" * 60)
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Install dependencies with robust approach
    if install_dependencies_robust():
        # Verify installation
        if verify_installation():
            print("\n✅ Setup completed successfully!")
            print("🚀 Ready to run Streamlit app")
        else:
            print("\n⚠️ Setup completed with warnings")
            print("🚀 App may have limited functionality")
    else:
        print("\n❌ Setup failed")
        print("🚀 App may not function properly")
        sys.exit(1)

if __name__ == "__main__":
    main() 