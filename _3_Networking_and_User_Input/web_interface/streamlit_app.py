"""
Trading Algorithm Training Interface
Main entry point for Streamlit Cloud deployment.
Enhanced with structural dependency management.
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
logger.info("=== Advanced Neural Network Trading System ===")
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

# Detect if running in Streamlit Cloud using multiple checks
is_cloud = is_streamlit_cloud()
logger.info(f"Running in Streamlit Cloud: {is_cloud}")
logger.info("=====================================")

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="🧠 Neural Network Trading Algorithm",
    page_icon="🧠",
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
TRAINING_THRESHOLD = 10  # Minimum examples needed for neural network training

def check_dependencies():
    """
    Check if all required dependencies are available with helpful error messages
    """
    missing_deps = []
    ml_deps_available = True
    
    # Check core dependencies
    try:
        import pandas as pd
        import plotly.graph_objects as go
        import numpy as np
        from dotenv import load_dotenv
    except ImportError as e:
        missing_deps.append(f"Core dependencies: {str(e)}")
    
    # Check yfinance (optional - has fallback)
    try:
        import yfinance as yf
        yf_available = True
    except ImportError:
        yf_available = False
    
    # Check ML dependencies
    try:
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        ml_deps_available = True
        tf_version = tf.__version__
    except ImportError as e:
        missing_deps.append(f"ML dependencies: {str(e)}")
        ml_deps_available = False
        tf_version = "Not installed"
    
    return missing_deps, ml_deps_available, yf_available, tf_version

def display_setup_instructions():
    """Display setup instructions when dependencies are missing"""
    st.error("🚨 **Missing Dependencies for Neural Network Training System**")
    
    st.markdown("""
    ### 🔧 **Setup Required**
    
    It looks like the required dependencies for the **Advanced Neural Network Trading System** are not installed.
    
    #### **For Local Development:**
    1. **Close this browser tab**
    2. **Open Command Prompt/PowerShell in your project directory**
    3. **Run the setup command:**
       ```bash
       0_Launchers\\setup_and_run.bat
       ```
    4. **Wait for installation to complete (may take 5-10 minutes)**
    5. **The Streamlit app will start automatically**
    
    #### **What Gets Installed:**
    - 🧠 **TensorFlow 2.13.0** - Neural network engine
    - 📊 **VADER Sentiment Analysis** - Analyzes your trading reasoning
    - 📈 **Advanced Technical Indicators** - RSI, MACD, Bollinger Bands
    - 🎯 **Interactive Charting** - Candlestick charts with 25 years of data
    - 🔗 **All Supporting Libraries** - Pandas, NumPy, Plotly, etc.
    
    #### **For Streamlit Cloud:**
    The system will automatically use the `web_requirements.txt` file which includes all necessary dependencies.
    """)
    
    # Show what's missing
    with st.expander("🔍 **Detailed Dependency Status**"):
        missing_deps, ml_available, yf_available, tf_version = check_dependencies()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Core Status:**")
            if not missing_deps:
                st.success("✅ Basic dependencies available")
            else:
                for dep in missing_deps:
                    st.error(f"❌ {dep}")
        
        with col2:
            st.write("**ML Status:**")
            st.write(f"TensorFlow: {tf_version}")
            if ml_available:
                st.success("✅ Neural network ready")
            else:
                st.error("❌ Neural network not available")
            
            if yf_available:
                st.success("✅ Real market data available")
            else:
                st.warning("⚠️ Using synthetic data (YFinance not available)")

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
    ❌ **Incompatible Python Version: {platform.python_version()}**
    
    This advanced neural network system requires **Python 3.9.x** for optimal compatibility 
    with TensorFlow 2.13.0 and other ML dependencies.
    
    **For Local Development:**
    1. Install Python 3.9.x from https://www.python.org/downloads/release/python-3913/
    2. Run: `0_Launchers\\setup_and_run.bat`
    
    **For Streamlit Cloud:**
    1. Go to your app settings
    2. Under 'Python version', select Python 3.9
    3. Redeploy your app
    """)
    st.stop()

# Check for required dependencies
missing_deps, ml_available, yf_available, tf_version = check_dependencies()

if missing_deps:
    display_setup_instructions()
    st.stop()

# Success! All dependencies are available
st.success(f"✅ **Neural Network Training System Ready** (TensorFlow {tf_version})")

# Try to import the training interface
try:
    from _2_Orchestrator_And_ML_Python.interactive_training_app.backend.model_trainer import ModelTrainer
    from _3_Networking_and_User_Input.web_interface.streamlit_training import main
    TRAINING_AVAILABLE = True
    st.sidebar.success("🧠 Neural Network Engine: Ready")
except ImportError as e:
    st.error(f"""
    ⚠️ **Training Interface Import Error: {str(e)}**
    
    The neural network training system is not properly set up.
    Please run the setup script: `0_Launchers\\setup_and_run.bat`
    """)
    TRAINING_AVAILABLE = False

# Set environment variables for Streamlit Cloud
os.environ["STREAMLIT_SERVER_PORT"] = "8501"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

# Load environment variables
from dotenv import load_dotenv
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
        st.sidebar.warning(f"""
        ⚠️ **Missing API Keys:** {', '.join(missing_vars)}
        
        These are optional for training but required for live trading.
        Add them to Streamlit Cloud secrets or your .env file.
        """)
        return False
    return True

def check_model_availability():
    """Check if a trained model is available and provide appropriate messaging"""
    model_state = get_model_state()
        
    if not TRAINING_AVAILABLE:
        st.error("""
        ❌ **Neural Network Training Interface Not Available**
        
        Please run the setup script: `0_Launchers\\setup_and_run.bat`
        """)
        return False
    
    if not model_state['is_trained']:
        if model_state['training_examples'] > 0:
            st.info(f"""
            🔄 **Model Ready for Training**
            
            You have {model_state['training_examples']} training examples.
            Need at least {TRAINING_THRESHOLD} examples to train the neural network.
            
            Continue adding examples to enable neural network training.
            """)
        else:
            st.info("""
            🎯 **Welcome to Neural Network Training**
            
            This system learns to mimic your trading decisions by analyzing both:
            - **Technical indicators** (RSI, MACD, etc.)  
            - **Your reasoning** (sentiment analysis of your explanations)
            
            Start by providing some training examples!
            """)
        return False
    
    st.success(f"""
    ✅ **Neural Network Trained and Ready**
    
    - **Last trained:** {model_state['last_training_date']}
    - **Training examples:** {model_state['training_examples']}
    - **Model accuracy:** {model_state['model_accuracy']:.2%}
    """)
    return True

if __name__ == "__main__":
    # Check environment first
    check_environment()
    
    # Run the main neural network training interface
    if TRAINING_AVAILABLE:
            main()
    else:
        # Show setup instructions
        st.title("🧠 Advanced Neural Network Trading Algorithm")
        st.markdown("**Intelligent Trading Decision System with Sentiment Analysis**")
        
        display_setup_instructions()
        
        st.markdown("---")
        st.markdown("""
        ### 🎯 **What This System Does:**
        
        1. **📊 Presents Random S&P100 Stocks** - Shows interactive charts with 25 years of data
        2. **🧠 Collects Your Trading Decisions** - You analyze and decide: BUY, SELL, or HOLD
        3. **📝 Analyzes Your Reasoning** - VADER sentiment analysis extracts keywords and sentiment
        4. **🤖 Trains Neural Network** - TensorFlow neural network learns your trading psychology
        5. **🔮 Makes Predictions** - AI mimics your decision-making process on new stocks
        
        This combines **technical analysis** with **sentiment analysis** to create an AI that 
        thinks like you do when making trading decisions.
        """) 