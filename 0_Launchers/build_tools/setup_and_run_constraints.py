import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def install_with_constraints() -> bool:
    """Install requirements using constraints to override dependency conflicts."""
    try:
        # Get the directory of this script
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        constraints_file = project_root / "streamlit_cloud" / "constraints.txt"
        requirements_file = project_root / "requirements.txt"
        
        if not constraints_file.exists():
            logger.error(f"Constraints file not found: {constraints_file}")
            return False
            
        if not requirements_file.exists():
            logger.error(f"Requirements file not found: {requirements_file}")
            return False
        
        logger.info("Installing requirements with constraints to resolve websocket conflict...")
        
        # Use pip install with constraints to override dependency conflicts
        cmd = [
            sys.executable, "-m", "pip", "install",
            "-r", str(requirements_file),
            "--constraint", str(constraints_file),
            "--force-reinstall"  # Force reinstall to ensure constraints are applied
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Successfully installed requirements with constraints")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install with constraints: {e.stderr}")
        return False

def install_requirements() -> None:
    """Install requirements using constraints for Streamlit Cloud."""
    if not install_with_constraints():
        logger.error("Failed to install requirements with constraints")
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
    """Main entry point for Streamlit Cloud deployment using constraints."""
    try:
        # Install requirements using constraints
        install_requirements()
        
        # Launch Streamlit app
        run_streamlit_app()
        
    except Exception as e:
        logger.error(f"Error during setup and launch: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 