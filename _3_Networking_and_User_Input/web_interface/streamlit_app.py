"""
Trading Algorithm Training Interface
Main entry point for Streamlit Cloud deployment.
For troubleshooting, see docs/streamlit_cloud_troubleshooting.md
"""

import os
import sys
from pathlib import Path
import streamlit as st
import platform
import logging
import json
from datetime import datetime

# Configure logging with file rotation
logging.basicConfig(
    filename="streamlit_app.log",
    filemode="w",  # Overwrite log file on each app start
    level=logging.INFO,  # Set to INFO to see debug messages
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True  # Force reconfiguration of logging
)

# Log environment information
logger = logging.getLogger(__name__)
logger.info("=== Streamlit App Environment Check ===")
logger.info(f"Python Version: {platform.python_version()}")
logger.info(f"Platform Processor: {platform.processor()}")
logger.info(f"Current Working Directory: {os.getcwd()}")
logger.info(f"Directory Contents: {os.listdir('.')}")
logger.info("Environment Variables:")
for var in ['STREAMLIT_SERVER_RUNNING_ON_CLOUD', 'STREAMLIT_SERVER_PORT', 
            'STREAMLIT_SERVER_HEADLESS', 'STREAMLIT_BROWSER_GATHER_USAGE_STATS']:
    logger.info(f"{var}: {os.environ.get(var)}")
logger.info(f"Mount path exists: {os.path.exists('/mount/src')}")
logger.info(f"Home directory exists: {os.path.exists('/home/adminuser')}")

# Detect if running in Streamlit Cloud
is_streamlit_cloud = platform.processor() == ''
logger.info(f"Running in Streamlit Cloud: {is_streamlit_cloud}")
logger.info("=====================================")

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="Trading Algorithm Training Interface",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add the orchestrator directory to Python path
REPO_ROOT = Path(__file__).parent.parent.parent
ORCHESTRATOR_PATH = REPO_ROOT / "_2_Orchestrator_And_ML_Python"
sys.path.append(str(ORCHESTRATOR_PATH))
# Add the networking directory to Python path
NETWORKING_PATH = REPO_ROOT / "_3_Networking_and_User_Input"
sys.path.append(str(NETWORKING_PATH))

# Constants for model state
MODEL_STATE_FILE = "model_state.json"
TRAINING_THRESHOLD = 5  # Minimum examples needed for training

def get_model_state():
    """Get the current state of the model"""
    try:
        if os.path.exists(MODEL_STATE_FILE):
            with open(MODEL_STATE_FILE, 'r') as f:
                state = json.load(f)
                return {
                    'is_trained': state.get('is_trained', False),
                    'training_examples': state.get('training_examples', 0),
                    'last_training_date': state.get('last_training_date', None),
                    'model_accuracy': state.get('model_accuracy', 0.0)
                }
    except Exception as e:
        logging.error(f"Error reading model state: {e}")
    return {
        'is_trained': False,
        'training_examples': 0,
        'last_training_date': None,
        'model_accuracy': 0.0
    }

def update_model_state(is_trained=False, training_examples=0, model_accuracy=0.0):
    """Update the model state file"""
    try:
        state = {
            'is_trained': is_trained,
            'training_examples': training_examples,
            'last_training_date': datetime.now().isoformat(),
            'model_accuracy': model_accuracy
        }
        with open(MODEL_STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logging.error(f"Error updating model state: {e}")

# Check Python version
python_version = platform.python_version_tuple()
if python_version[0] != '3' or python_version[1] != '9':
    st.error(f"""
    ❌ Incompatible Python version: {platform.python_version()}
    
    This app requires Python 3.9.x for compatibility with TensorFlow and other dependencies.
    Please update your environment to use Python 3.9.x:
    
    For local development:
    1. Install Python 3.9.x from https://www.python.org/downloads/release/python-3913/
    2. Run setup_local_env.bat to create a new virtual environment
    
    For Streamlit Cloud:
    1. Go to your app settings
    2. Under 'Python version', select Python 3.9
    3. Redeploy your app
    """)
    st.stop()

# Check for required dependencies
try:
    import pandas as pd
    import plotly.graph_objects as go
    import yfinance as yf
    import numpy as np
    from dotenv import load_dotenv
except ImportError as e:
    st.error(f"""
    ❌ Missing required dependencies: {str(e)}
    
    Please ensure all dependencies are installed:
    1. Check that web_requirements.txt is properly configured
    2. Try restarting the app
    3. If the issue persists, contact support
    """)
    st.stop()

# Try to import ML dependencies with graceful fallback
try:
    import tensorflow as tf
    ML_AVAILABLE = True
    st.sidebar.success(f"TensorFlow {tf.__version__} loaded successfully")
except ImportError as e:
    st.warning(f"""
    ⚠️ TensorFlow not available: {str(e)}
    
    The app will run in basic mode without ML capabilities.
    This is expected if you're running on Streamlit Cloud without proper Python version.
    """)
    ML_AVAILABLE = False

# Try to import the training interface with graceful fallback
try:
    from _2_Orchestrator_And_ML_Python.interactive_training_app.backend.model_trainer import ModelTrainer
    from _3_Networking_and_User_Input.web_interface.streamlit_training import main
    TRAINING_AVAILABLE = True
except ImportError as e:
    st.warning(f"""
    ⚠️ Training interface not available: {str(e)}
    
    The app will run in basic mode without training capabilities.
    This is expected if you're running on Streamlit Cloud.
    """)
    TRAINING_AVAILABLE = False

# Set environment variables for Streamlit Cloud
os.environ["STREAMLIT_SERVER_PORT"] = "8501"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

# Load environment variables
load_dotenv()

def check_environment():
    """Check if all required environment variables are set"""
    required_vars = [
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
        "APCA_BASE_URL",
        "FMP_API_KEY",
        "POLYGON_API_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        st.warning(f"""
        ⚠️ Missing environment variables: {', '.join(missing_vars)}
        
        Please add these to your Streamlit Cloud secrets:
        1. Go to your app settings
        2. Click on 'Secrets'
        3. Add the missing variables
        """)
        return False
    return True

def check_model_availability():
    """Check if a trained model is available and provide appropriate messaging"""
    model_state = get_model_state()
    
    if not ML_AVAILABLE:
        st.error("""
        ❌ TensorFlow is not available.
        
        This is required for model training and predictions.
        Please ensure you're using Python 3.9 in Streamlit Cloud settings.
        See docs/streamlit_cloud_troubleshooting.md for more information.
        """)
        return False
        
    if not TRAINING_AVAILABLE:
        st.error("""
        ❌ Training interface is not available.
        
        This is required for model training and predictions.
        Please check that all required modules are properly installed.
        """)
        return False
    
    if not model_state['is_trained']:
        if model_state['training_examples'] > 0:
            st.warning(f"""
            ⚠️ Model needs training
            
            You have {model_state['training_examples']} training examples.
            Need at least {TRAINING_THRESHOLD} examples to train the model.
            
            Please continue training to enable predictions.
            """)
        else:
            st.info("""
            ℹ️ No trained model available
            
            To get started:
            1. Use the training interface to provide examples
            2. Train the model with at least 5 examples
            3. Once trained, you can use the model for predictions
            """)
        return False
    
    st.success(f"""
    ✅ Model is trained and ready
    
    - Last trained: {model_state['last_training_date']}
    - Training examples: {model_state['training_examples']}
    - Model accuracy: {model_state['model_accuracy']:.2%}
    """)
    return True

if __name__ == "__main__":
    # Check environment first
    if not check_environment():
        st.stop()
    
    # Run the main Streamlit app
    if TRAINING_AVAILABLE and ML_AVAILABLE:
        # Check model availability before running main interface
        model_ready = check_model_availability()
        if model_ready:
            main()
        else:
            # Show training interface even if model isn't ready
            st.title("Trading Algorithm Training Interface")
            st.write("Please train the model to enable predictions.")
            # Import and show training interface
            from _2_Orchestrator_And_ML_Python.interactive_training_app.backend.model_trainer import ModelTrainer
            trainer = ModelTrainer()
            # Show training interface components
            st.subheader("Training Interface")
            # Add your training interface components here
    else:
        # Basic mode without ML
        st.title("Trading Algorithm Training Interface")
        
        if not ML_AVAILABLE:
            st.error("""
            ❌ TensorFlow is not available.
            Please ensure you're using Python 3.9 in Streamlit Cloud settings.
            """)
        
        if not TRAINING_AVAILABLE:
            st.warning("""
            ⚠️ Training interface is not available.
            Running in basic mode with limited functionality.
            """)
        
        # Add basic functionality here
        st.write("Basic trading interface coming soon...") 