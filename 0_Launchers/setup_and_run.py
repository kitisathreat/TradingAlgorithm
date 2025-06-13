import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Optional
import platform

def is_streamlit_cloud() -> bool:
    """
    Detect if running in Streamlit Cloud using multiple checks.
    Returns True if any of the checks indicate Streamlit Cloud.
    """
    # Check 1: Platform processor (empty string in Streamlit Cloud)
    if platform.processor() == '':
        return True
        
    # Check 2: Environment variable
    if os.environ.get('STREAMLIT_SERVER_RUNNING_ON_CLOUD', '').lower() == 'true':
        return True
        
    # Check 3: Mount path (exists in Streamlit Cloud)
    if os.path.exists('/mount/src'):
        return True
        
    # Check 4: Home directory (exists in Streamlit Cloud)
    if os.path.exists('/home/adminuser'):
        return True
        
    return False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_virtual_env() -> Optional[str]:
    """
    Set up virtual environment if running locally.
    Returns the path to the virtual environment's Python executable if created,
    None if running in cloud or if using system Python.
    """
    if is_streamlit_cloud():
        logger.info("Running in Streamlit Cloud - using system Python")
        return None

    # Get the project root directory (parent of 0_Launchers)
    project_root = Path(__file__).parent.parent
    venv_path = project_root / "venv"
    
    # Check if venv already exists
    if venv_path.exists():
        logger.info("Removing existing virtual environment...")
        if sys.platform == "win32":
            subprocess.run(["rmdir", "/s", "/q", str(venv_path)], check=True)
        else:
            subprocess.run(["rm", "-rf", str(venv_path)], check=True)

    logger.info("Creating new virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)

    # Get the path to the virtual environment's Python executable
    if sys.platform == "win32":
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        python_path = venv_path / "bin" / "python"

    if not python_path.exists():
        raise RuntimeError(f"Virtual environment Python not found at {python_path}")

    return str(python_path)

def install_requirements(python_path: Optional[str] = None) -> None:
    """Install requirements using the package manager."""
    if python_path is None:
        python_path = sys.executable

    # Get the directory of this script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Import package manager from root directory
    sys.path.append(str(project_root))
    from package_manager import PackageManager

    # Initialize package manager
    manager = PackageManager()

    # Install root requirements first
    root_reqs = project_root / "root_requirements.txt"
    logger.info(f"Installing root requirements from {root_reqs}")
    failed = manager.install_requirements(str(root_reqs))
    if failed:
        logger.error(f"Failed to install some root requirements: {failed}")
        sys.exit(1)

    # Install ML requirements if running locally
    if not is_streamlit_cloud():
        ml_reqs = project_root / "2_Orchestrator_And_ML_Python" / "ml_requirements.txt"
        if ml_reqs.exists():
            logger.info(f"Installing ML requirements from {ml_reqs}")
            failed = manager.install_requirements(str(ml_reqs))
            if failed:
                logger.error(f"Failed to install some ML requirements: {failed}")
                sys.exit(1)

def run_streamlit_app(python_path: Optional[str] = None) -> None:
    """Launch the Streamlit app."""
    if python_path is None:
        python_path = sys.executable

    # Get the path to the Streamlit app
    project_root = Path(__file__).parent.parent
    app_path = project_root / "3_Networking_and_User_Input" / "web_interface" / "app.py"

    if not app_path.exists():
        raise FileNotFoundError(f"Streamlit app not found at {app_path}")

    logger.info("Launching Streamlit app...")
    
    # Use the appropriate Python executable to run Streamlit
    cmd = [
        python_path,
        "-m", "streamlit", "run",
        str(app_path),
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ]

    # Add cloud-specific settings if running in Streamlit Cloud
    if is_streamlit_cloud():
        cmd.extend([
            "--server.enableCORS=false",
            "--server.enableXsrfProtection=false"
        ])

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("\nStreamlit server stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Streamlit server: {e}")
        sys.exit(1)

def main():
    """Main entry point for the launcher script."""
    try:
        # Set up virtual environment if running locally
        python_path = setup_virtual_env()
        
        # Install requirements
        install_requirements(python_path)
        
        # Launch Streamlit app
        run_streamlit_app(python_path)
        
    except Exception as e:
        logger.error(f"Error during setup and launch: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 