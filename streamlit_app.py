"""
Trading Algorithm Training Interface
Main entry point for Streamlit Cloud deployment.
"""

import os
import sys
from pathlib import Path
import streamlit as st
import platform

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="Trading Algorithm Training Interface",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add the orchestrator directory to Python path
REPO_ROOT = Path(__file__).parent
ORCHESTRATOR_PATH = REPO_ROOT / "_2_Orchestrator_And_ML_Python"
sys.path.append(str(ORCHESTRATOR_PATH))
# Add the networking directory to Python path
NETWORKING_PATH = REPO_ROOT / "networking_and_user_input"
sys.path.append(str(NETWORKING_PATH))

# Check Python version
python_version = platform.python_version_tuple()
if int(python_version[0]) > 3 or (int(python_version[0]) == 3 and int(python_version[1]) > 9):
    st.error(f"""
    ❌ Incompatible Python version: {platform.python_version()}
    
    This app requires Python 3.9.x for TensorFlow compatibility.
    Please update your Streamlit Cloud settings to use Python 3.9.
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
    1. Check that streamlit_requirements.txt is properly configured
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
    from networking_and_user_input.web_interface.streamlit_training import main
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

if __name__ == "__main__":
    # Check environment first
    if not check_environment():
        st.stop()
    
    # Run the main Streamlit app
    if TRAINING_AVAILABLE and ML_AVAILABLE:
        main()
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