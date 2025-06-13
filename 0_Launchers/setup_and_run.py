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

def install_requirements() -> None:
    """Install requirements using the package manager for Streamlit Cloud."""
    # Get the directory of this script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Import package manager from root directory
    sys.path.append(str(project_root))
    from package_manager import PackageManager

    # Initialize package manager
    manager = PackageManager()

    # Always use requirements.txt for Streamlit Cloud deployment
    cloud_reqs = project_root / "requirements.txt"
    if not cloud_reqs.exists():
        logger.error(f"Streamlit Cloud requirements file not found: {cloud_reqs}")
        sys.exit(1)

    logger.info(f"Installing Streamlit Cloud requirements from {cloud_reqs}")
    failed = manager.install_requirements(str(cloud_reqs))
    if failed:
        logger.error(f"Failed to install some requirements: {failed}")
        sys.exit(1)

def run_streamlit_app() -> None:
    """Launch the Streamlit app in Streamlit Cloud."""
    # Get the path to the Streamlit app
    project_root = Path(__file__).parent.parent
    app_path = project_root / "_3_Networking_and_User_Input" / "web_interface" / "streamlit_app.py"

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
    """Main entry point for Streamlit Cloud deployment."""
    try:
        # Install requirements (always use requirements.txt for Streamlit Cloud)
        install_requirements()
        
        # Launch Streamlit app
        run_streamlit_app()
        
    except Exception as e:
        logger.error(f"Error during setup and launch: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 