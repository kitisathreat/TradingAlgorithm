import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def try_install_with_constraints() -> bool:
    """Try installing with constraints file approach."""
    try:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        constraints_file = project_root / "streamlit_cloud" / "constraints.txt"
        requirements_file = project_root / "requirements.txt"
        
        if not constraints_file.exists() or not requirements_file.exists():
            return False
        
        logger.info("Attempting installation with constraints file...")
        cmd = [
            sys.executable, "-m", "pip", "install",
            "-r", str(requirements_file),
            "--constraint", str(constraints_file),
            "--force-reinstall"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Successfully installed with constraints")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.warning(f"Constraints approach failed: {e.stderr}")
        return False

def try_install_with_no_deps() -> bool:
    """Try installing with --no-deps approach."""
    try:
        logger.info("Attempting installation with --no-deps approach...")
        
        # First install websockets 13+
        cmd1 = [sys.executable, "-m", "pip", "install", "websockets>=13.0", "--force-reinstall"]
        subprocess.run(cmd1, capture_output=True, text=True, check=True)
        
        # Then install alpaca with --no-deps
        cmd2 = [sys.executable, "-m", "pip", "install", "alpaca-trade-api==3.2.0", "--no-deps"]
        subprocess.run(cmd2, capture_output=True, text=True, check=True)
        
        # Finally install the rest
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        requirements_file = project_root / "requirements.txt"
        
        cmd3 = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        subprocess.run(cmd3, capture_output=True, text=True, check=True)
        
        logger.info("Successfully installed with --no-deps approach")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.warning(f"--no-deps approach failed: {e}")
        return False

def try_install_with_override() -> bool:
    """Try installing with override requirements file."""
    try:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        override_file = project_root / "streamlit_cloud" / "streamlit_requirements_override.txt"
        
        if not override_file.exists():
            return False
        
        logger.info("Attempting installation with override requirements...")
        
        # Install websockets first
        cmd1 = [sys.executable, "-m", "pip", "install", "websockets>=13.0"]
        subprocess.run(cmd1, capture_output=True, text=True, check=True)
        
        # Install alpaca with --no-deps
        cmd2 = [sys.executable, "-m", "pip", "install", "alpaca-trade-api==3.2.0", "--no-deps"]
        subprocess.run(cmd2, capture_output=True, text=True, check=True)
        
        # Install the rest from override file (excluding alpaca line)
        with open(override_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#') and 'alpaca-trade-api' not in line]
        
        for line in lines:
            if line and not line.startswith('#'):
                cmd = [sys.executable, "-m", "pip", "install", line]
                subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        logger.info("Successfully installed with override approach")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.warning(f"Override approach failed: {e}")
        return False

def try_install_with_force() -> bool:
    """Try installing with --force-reinstall and ignore conflicts."""
    try:
        logger.info("Attempting installation with force approach...")
        
        # Force install websockets 13+
        cmd1 = [sys.executable, "-m", "pip", "install", "websockets>=13.0", "--force-reinstall", "--no-deps"]
        subprocess.run(cmd1, capture_output=True, text=True, check=True)
        
        # Force install alpaca ignoring dependencies
        cmd2 = [sys.executable, "-m", "pip", "install", "alpaca-trade-api==3.2.0", "--force-reinstall", "--no-deps"]
        subprocess.run(cmd2, capture_output=True, text=True, check=True)
        
        # Install the rest normally
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        requirements_file = project_root / "requirements.txt"
        
        with open(requirements_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#') and 'alpaca-trade-api' not in line]
        
        for line in lines:
            if line and not line.startswith('#'):
                cmd = [sys.executable, "-m", "pip", "install", line, "--force-reinstall"]
                subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        logger.info("Successfully installed with force approach")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.warning(f"Force approach failed: {e}")
        return False

def install_requirements() -> None:
    """Try multiple approaches to install requirements."""
    approaches = [
        ("Constraints", try_install_with_constraints),
        ("No-deps", try_install_with_no_deps),
        ("Override", try_install_with_override),
        ("Force", try_install_with_force)
    ]
    
    for name, approach in approaches:
        logger.info(f"Trying {name} approach...")
        if approach():
            logger.info(f"Successfully installed requirements using {name} approach")
            return
    
    logger.error("All installation approaches failed")
    sys.exit(1)

def run_streamlit_app() -> None:
    """Launch the Streamlit app in Streamlit Cloud."""
    # Get the path to the Streamlit app
    project_root = Path(__file__).parent.parent
    app_path = project_root / "streamlit_app.py"

    if not app_path.exists():
        raise FileNotFoundError(f"Streamlit app not found at {app_path}")

    logger.info("Launching Streamlit app...")
    
    # Run Streamlit with cloud-specific settings
    cmd = [
        sys.executable,
        "-m", "streamlit", "run",
        str(app_path),
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false"
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("\nStreamlit server stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Streamlit server: {e}")
        sys.exit(1)

def main():
    """Main entry point for Streamlit Cloud deployment with robust dependency resolution."""
    try:
        # Install requirements using multiple approaches
        install_requirements()
        
        # Launch Streamlit app
        run_streamlit_app()
        
    except Exception as e:
        logger.error(f"Error during setup and launch: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 