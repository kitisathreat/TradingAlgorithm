"""
Trading Algorithm Training Interface
Main entry point for Streamlit Cloud deployment.
"""

import os
import sys
from pathlib import Path

# Add the orchestrator directory to Python path
REPO_ROOT = Path(__file__).parent
ORCHESTRATOR_PATH = REPO_ROOT / "_2_Orchestrator_And_ML_Python"
sys.path.append(str(ORCHESTRATOR_PATH))

# Import the training interface
from _2_Orchestrator_And_ML_Python.interactive_training_app.backend.model_trainer import ModelTrainer
from _3_Networking_and_User_Input.web_interface.streamlit_training import main

if __name__ == "__main__":
    # Set environment variables for Streamlit Cloud
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    
    # Run the main Streamlit app
    main() 