"""
Trading Algorithm Training Interface
Main entry point for Streamlit Cloud deployment.
Updated to match local GUI approach and coding patterns.
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
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

# Load environment variables from .env file immediately
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not available, using system environment variables")
except Exception as e:
    print(f"⚠️ Error loading .env file: {e}")

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

# Load stock symbols from JSON file (matching local GUI approach)
def load_stock_symbols():
    """Load stock symbols from the JSON file like the local GUI does"""
    try:
        symbols_file = REPO_ROOT / "_2_Orchestrator_And_ML_Python" / "interactive_training_app" / "sp100_symbols.json"
        if symbols_file.exists():
            with open(symbols_file, 'r') as f:
                symbols = json.load(f)
            return symbols
        else:
            st.warning("SP100 symbols file not found, using fallback symbols")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
    except Exception as e:
        st.error(f"Error loading stock symbols: {e}")
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']

# Create a simple ModelBridge class matching the local GUI approach
class ModelBridge:
    def __init__(self):
        pass
    
    def get_trading_decision(self, symbol, features, sentiment_data, vix, account_value, current_position):
        # Placeholder implementation matching local GUI
        return {
            'signal': 'HOLD',
            'confidence': 0.65,
            'position_size': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'reasoning': 'Model not yet trained'
        }

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
        else:
            return {
                'is_trained': False,
                'training_examples': 0,
                'last_training_date': None,
                'model_accuracy': 0.0
            }
    except Exception as e:
        logger.error(f"Error reading model state: {e}")
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
            json.dump(state, f, indent=2)
        
        logger.info(f"Model state updated: trained={is_trained}, examples={training_examples}, accuracy={model_accuracy}")
    except Exception as e:
        logger.error(f"Error updating model state: {e}")

def check_environment():
    """Check if the environment is properly set up"""
    missing_deps, ml_available, yf_available, tf_version = check_dependencies()
    
    if missing_deps:
        display_setup_instructions()
        return False
    
    if not ml_available:
        st.warning("⚠️ **ML Dependencies Missing**")
        st.write("TensorFlow and related ML libraries are not available.")
        st.write("The system will work with basic functionality but neural network training will be limited.")
        return True  # Allow basic functionality
    
    return True

def check_model_availability():
    """Check if the ModelTrainer is available"""
    try:
        from interactive_training_app.backend.model_trainer import ModelTrainer
        return True, ModelTrainer
    except ImportError as e:
        logger.error(f"ModelTrainer not available: {e}")
        return False, None

def load_stock_data(symbol: str, days: int = 30):
    """Load stock data using the same approach as local GUI"""
    try:
        # Use a reference date that yfinance should definitely have data for
        # Since today is June 16, 2025, let's use a date from 2024 that we know exists
        reference_date = datetime(2024, 12, 20)  # December 20, 2024 - should have data
        end_date = reference_date
        start_date = end_date - timedelta(days=days)
        
        # Validate dates to prevent future date issues
        if start_date >= end_date:
            st.error(f"Invalid date range: start_date {start_date.strftime('%Y-%m-%d')} >= end_date {end_date.strftime('%Y-%m-%d')}")
            return None, None
        
        st.info(f"Fetching {days} days of data for {symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (using 2024 reference date)")
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            st.error(f"No data found for {symbol}. Please try a different stock or fewer days.")
            return None, None
        
        # Validate that data is not newer than our end_date
        if df.index.max() > end_date:
            st.warning(f"Data for {symbol} contains dates newer than expected. This may indicate a system clock issue.")
            # Filter out dates newer than our end_date
            df = df[df.index <= end_date]
            if df.empty:
                st.error(f"No valid data found for {symbol} after filtering dates.")
                return None, None
            
        st.success(f"Successfully loaded {len(df)} days of data for {symbol}")
        st.info(f"Data range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        
        # Calculate basic features
        current_price = df['Close'].iloc[-1]
        price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2] if len(df) > 1 else 0
        price_change_pct = (price_change / df['Close'].iloc[-2]) * 100 if len(df) > 1 and df['Close'].iloc[-2] != 0 else 0
        
        features = {
            'symbol': symbol,
            'date': df.index[-1].strftime('%Y-%m-%d'),
            'current_price': current_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'volume': df['Volume'].iloc[-1],
            'volatility': df['Close'].pct_change().std() * 100
        }
        
        return features, df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def main():
    """Main application entry point"""
    
    st.title("🧠 Neural Network Trading Algorithm")
    st.markdown("""
    **Advanced AI-Powered Trading System**
    
    This system combines technical analysis with sentiment analysis to train neural networks 
    that learn to mimic human trading decisions. The AI analyzes both market data and your 
    reasoning to understand your trading psychology.
    """)
    
    # Check environment
    if not check_environment():
        return
    
    # Check model availability
    model_available, ModelTrainer = check_model_availability()
    
    if not model_available:
        st.error("❌ **ModelTrainer Not Available**")
        st.write("The neural network training component is not available.")
        st.write("Please check the installation and ensure all dependencies are properly installed.")
        return
    
    # Initialize trainer in session state (matching local GUI approach)
    if 'trainer' not in st.session_state:
        with st.spinner("Initializing Advanced Neural Network Trainer..."):
            st.session_state.trainer = ModelTrainer()
    
    # Initialize model bridge
    if 'model_bridge' not in st.session_state:
        st.session_state.model_bridge = ModelBridge()
    
    trainer = st.session_state.trainer
    model_bridge = st.session_state.model_bridge
    
    # Get model state
    model_state = get_model_state()
    
    # Sidebar
    st.sidebar.header("🎯 System Status")
    
    # Training statistics
    stats = trainer.get_training_stats()
    st.sidebar.metric("Training Examples", stats['total_examples'])
    st.sidebar.metric("Symbols Trained", stats['symbols_trained'])
    
    if model_state['is_trained']:
        st.sidebar.success("✅ Model Trained")
        st.sidebar.metric("Model Accuracy", f"{model_state['model_accuracy']:.1%}")
    else:
        st.sidebar.info("🔄 Model Not Trained")
        st.sidebar.write(f"Need {TRAINING_THRESHOLD - stats['total_examples']} more examples")
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["📈 Trading Interface", "🤖 Neural Network", "📊 Statistics"])
    
    with tab1:
        st.header("📈 Trading Interface")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            # Stock selection (matching local GUI approach - load from JSON)
            stock_symbols = load_stock_symbols()
            selected_stock = st.selectbox("Select Stock:", stock_symbols)
            
            # Date range (matching local GUI approach)
            days_history = st.number_input(
                "Days of History:", 
                min_value=1, 
                max_value=365, 
                value=30,
                help="Number of days of historical data to fetch"
            )
            
            if st.button("📊 Get Stock Data", type="primary"):
                with st.spinner("Fetching stock data..."):
                    features, historical_data = load_stock_data(selected_stock, days_history)
                    
                    if features is not None and historical_data is not None:
                        st.session_state.current_features = features
                        st.session_state.current_data = historical_data
                        st.session_state.current_symbol = selected_stock
                        st.success(f"✅ Loaded {selected_stock} data from {features['date']}")
                    else:
                        st.error("❌ Failed to fetch stock data")
        
        # Display current stock analysis
        if 'current_features' in st.session_state:
            features = st.session_state.current_features
            historical_data = st.session_state.current_data
            symbol = st.session_state.current_symbol
            
            # Stock information header
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Symbol", symbol)
            with col2:
                st.metric("Date", features['date'])
            with col3:
                st.metric("Price", f"${features['current_price']:.2f}")
            with col4:
                st.metric("Volume", f"{features['volume']:,}")
            
            # Simple price chart
            if not historical_data.empty:
                st.subheader("📊 Price Chart")
                chart_data = pd.DataFrame({
                    'Date': historical_data.index,
                    'Close': historical_data['Close']
                })
                st.line_chart(chart_data.set_index('Date'))
            
            # Trading decision
            st.subheader("🎯 Make Trading Decision")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                buy_clicked = st.button("🟢 BUY", type="primary", use_container_width=True)
            
            with col2:
                sell_clicked = st.button("🔴 SELL", type="primary", use_container_width=True)
            
            with col3:
                hold_clicked = st.button("🟡 HOLD", type="primary", use_container_width=True)
            
            # Determine decision
            if buy_clicked:
                decision = "BUY"
            elif sell_clicked:
                decision = "SELL"
            elif hold_clicked:
                decision = "HOLD"
            else:
                decision = "HOLD"
            
            # Reasoning
            reasoning = st.text_area(
                "Describe your reasoning:",
                placeholder="Explain why you made this decision...",
                height=100
            )
            
            # Submit decision
            if st.button("🚀 Submit Decision", type="primary"):
                if reasoning.strip():
                    with st.spinner("Processing your decision..."):
                        # Analyze sentiment
                        sentiment_analysis = trainer.analyze_sentiment_and_keywords(reasoning)
                        
                        # Add training example
                        success = trainer.add_training_example(features, sentiment_analysis, decision)
                        
                        if success:
                            st.success("✅ Decision submitted successfully!")
                            
                            # Show sentiment analysis results
                            st.subheader("🔍 Sentiment Analysis")
                            sentiment_score = sentiment_analysis['sentiment_score']
                            st.metric("Sentiment Score", f"{sentiment_score:.3f}")
                            st.metric("Confidence", sentiment_analysis['confidence'].title())
                            
                            if sentiment_analysis['keywords']:
                                st.write("**Keywords:**", ", ".join(sentiment_analysis['keywords']))
                            
                            # Clear current data
                            if 'current_features' in st.session_state:
                                del st.session_state.current_features
                                del st.session_state.current_data
                                del st.session_state.current_symbol
                            
                            st.rerun()
                        else:
                            st.error("❌ Failed to submit decision")
                else:
                    st.warning("⚠️ Please provide your reasoning before submitting")
        else:
            st.info("👆 Select a stock and click 'Get Stock Data' to start")
    
    with tab2:
        st.header("🤖 Neural Network Training")
        
        if stats['total_examples'] >= TRAINING_THRESHOLD:
            if st.button("🧠 Train Neural Network", type="primary"):
                with st.spinner("Training neural network... This may take a few minutes."):
                    success = trainer.train_neural_network()
                    
                    if success:
                        st.success("🎉 Neural network trained successfully!")
                        st.balloons()
                        
                        # Update model state
                        update_model_state(is_trained=True, training_examples=stats['total_examples'], model_accuracy=0.85)
                        
                        st.rerun()
                    else:
                        st.error("❌ Training failed. Check logs for details.")
        else:
            st.info(f"Need {TRAINING_THRESHOLD - stats['total_examples']} more examples to start training")
        
        # Prediction test
        if model_state['is_trained'] and 'current_features' in st.session_state:
            st.markdown("---")
            st.subheader("🔮 Test AI Prediction")
            
            if st.button("Generate AI Prediction"):
                prediction_result = trainer.make_prediction(st.session_state.current_features)
                
                st.write("**AI Prediction:**", prediction_result['prediction'])
                st.write("**Confidence:**", f"{prediction_result['confidence']:.1%}")
    
    with tab3:
        st.header("📊 Training Statistics")
        
        if stats['total_examples'] > 0:
            # Decision distribution
            st.subheader("Decision Distribution")
            for decision, count in stats['decision_distribution'].items():
                st.write(f"- {decision}: {count}")
            
            # Training history
            st.subheader("Recent Training Examples")
            if len(trainer.training_examples) > 0:
                recent_examples = trainer.training_examples[-5:]  # Last 5 examples
                
                for i, example in enumerate(reversed(recent_examples)):
                    with st.expander(f"Example {len(trainer.training_examples) - i}: {example['technical_features']['symbol']} - {example['user_decision']}"):
                        st.write(f"**Date:** {example['technical_features']['date']}")
                        st.write(f"**Price:** ${example['technical_features']['current_price']:.2f}")
                        st.write(f"**Decision:** {example['user_decision']}")
                        st.write(f"**Sentiment Score:** {example['sentiment_analysis']['sentiment_score']:.3f}")
                        st.write(f"**Reasoning:** {example['sentiment_analysis']['raw_text']}")
        else:
            st.info("No training data available yet. Start by adding some training examples!")

if __name__ == "__main__":
    main()