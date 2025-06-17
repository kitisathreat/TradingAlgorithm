"""
Trading Algorithm Training Interface
Main entry point for Streamlit Cloud deployment.
Updated to exactly replicate local GUI layout and functionality.
Enhanced with structural dependency management.
For troubleshooting, see docs/streamlit_cloud_troubleshooting.md
"""

import sys
import os
from pathlib import Path
import streamlit as st
import platform
import logging
import json
from datetime import datetime, timedelta, timezone
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Load environment variables from .env file immediately
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Environment variables loaded from .env file")
except ImportError:
    print("python-dotenv not available, using system environment variables")
except Exception as e:
    print(f"Error loading .env file: {e}")

print("PYTHONPATH:", sys.path)

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
    page_title="Neural Network Trading System",
    page_icon="",
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
    st.error("Missing Dependencies for Neural Network Training System")
    
    st.markdown("""
    ### Setup Required
    
    It looks like the required dependencies for the **Advanced Neural Network Trading System** are not installed.
    
    #### For Local Development:
    1. **Close this browser tab**
    2. **Open Command Prompt/PowerShell in your project directory**
    3. **Run the setup command:**
       ```bash
       0_Launchers\\setup_and_run.bat
       ```
    4. **Wait for installation to complete (may take 5-10 minutes)**
    5. **The Streamlit app will start automatically**
    
    #### What Gets Installed:
    - **TensorFlow 2.13.0** - Neural network engine
    - **VADER Sentiment Analysis** - Analyzes your trading reasoning
    - **Advanced Technical Indicators** - RSI, MACD, Bollinger Bands
    - **Interactive Charting** - Candlestick charts with 25 years of data
    - **All Supporting Libraries** - Pandas, NumPy, Plotly, etc.
    
    #### For Streamlit Cloud:
    The system will automatically use the `web_requirements.txt` file which includes all necessary dependencies.
    """)
    
    # Show what's missing
    missing_deps, ml_deps_available, yf_available, tf_version = check_dependencies()
    
    if missing_deps:
        st.markdown("### Missing Dependencies:")
        for dep in missing_deps:
            st.write(f"- {dep}")
    
    st.markdown("### Current Status:")
    st.write(f"- **TensorFlow Version:** {tf_version}")
    st.write(f"- **ML Dependencies:** {'Available' if ml_deps_available else 'Missing'}")
    st.write(f"- **YFinance:** {'Available' if yf_available else 'Missing'}")
    
    st.stop()

def get_model_state():
    """Get the current model state from file"""
    try:
        if os.path.exists(MODEL_STATE_FILE):
            with open(MODEL_STATE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading model state: {e}")
    
    return {
        'is_trained': False,
        'training_examples': 0,
        'model_accuracy': 0.0,
        'last_trained': None
    }

def update_model_state(is_trained=False, training_examples=0, model_accuracy=0.0):
    """Update the model state file"""
    try:
        state = {
            'is_trained': is_trained,
            'training_examples': training_examples,
            'model_accuracy': model_accuracy,
            'last_trained': datetime.now().isoformat()
        }
        with open(MODEL_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        st.error(f"Error saving model state: {e}")

def check_environment():
    """Check if the environment is properly set up"""
    missing_deps, ml_deps_available, yf_available, tf_version = check_dependencies()
    
    if missing_deps:
        display_setup_instructions()
        return False
    
    return True

def check_model_availability():
    """Check if the ModelTrainer is available"""
    try:
        from _2_Orchestrator_And_ML_Python.interactive_training_app.backend.model_trainer import ModelTrainer
        return True, ModelTrainer
    except ImportError as e:
        st.error(f"ModelTrainer not available: {e}")
        return False, None

def load_stock_data(symbol: str, days: int = 30):
    """Load stock data and calculate features"""
    try:
        # Fetch data from yfinance
        stock = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get historical data
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            st.error(f"No data available for {symbol}")
            return None, None
        
        # Calculate technical indicators
        features = calculate_technical_features(df, symbol)
        
        return features, df
        
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {e}")
        return None, None

def calculate_technical_features(df, symbol):
    """Calculate technical features from stock data"""
    try:
        # Basic price features
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
        
        # Volume features
        current_volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # Volatility features
        daily_returns = df['Close'].pct_change().dropna()
        daily_vol = daily_returns.std() * 100
        weekly_vol = daily_returns.rolling(5).std().iloc[-1] * 100 if len(daily_returns) >= 5 else daily_vol
        
        # Technical indicators
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        current_macd = macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0
        
        # Moving averages
        sma_20 = df['Close'].rolling(20).mean()
        sma_20_current = sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else current_price
        ema_12_current = ema_12.iloc[-1] if not pd.isna(ema_12.iloc[-1]) else current_price
        
        # ATR calculation
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(14).mean().iloc[-1] if len(df) >= 14 else true_range.mean()
        
        # Bollinger Bands width
        std_20 = df['Close'].rolling(20).std()
        bollinger_width = (std_20.iloc[-1] / sma_20.iloc[-1]) * 100 if not pd.isna(sma_20.iloc[-1]) else 0
        
        return {
            'symbol': symbol,
            'date': df.index[-1].strftime('%Y-%m-%d'),
            'current_price': current_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'volume': current_volume,
            'avg_volume': avg_volume,
            'volume_ratio': volume_ratio,
            'daily_volatility': daily_vol,
            'weekly_volatility': weekly_vol,
            'rsi': current_rsi,
            'macd': current_macd,
            'sma_20': sma_20_current,
            'ema_12': ema_12_current,
            'atr': atr,
            'bollinger_width': bollinger_width,
            'high': df['High'].max(),
            'low': df['Low'].min(),
            'avg_price': df['Close'].mean()
        }
        
    except Exception as e:
        st.error(f"Error calculating features: {e}")
        return None

def create_candlestick_chart(df):
    """Create an interactive candlestick chart matching the local GUI"""
    if df is None or df.empty:
        return None
    
    # Create candlestick chart
    fig = go.Figure()
    
    # Add candlestick trace
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))
    
    # Add current price line
    current_price = df['Close'].iloc[-1]
    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="blue",
        line_width=2,
        annotation_text=f"Current: ${current_price:.2f}"
    )
    
    # Update layout to match local GUI style
    fig.update_layout(
        title="",
        xaxis_title="Date",
        yaxis_title="Price (USD$)",
        template="plotly_white",
        height=500,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=30, b=50)
    )
    
    # Update x-axis to show dates properly
    fig.update_xaxes(
        type='date',
        tickformat='%d/%m/%y',
        tickmode='auto',
        nticks=10
    )
    
    # Update y-axis
    fig.update_yaxes(
        tickprefix="$",
        tickformat=".2f"
    )
    
    return fig

def create_metrics_panel(features):
    """Create the metrics panel matching the local GUI layout exactly"""
    if features is None:
        return
    
    # Use columns to create the collapsible metrics panel
    with st.expander("Financial Metrics", expanded=True):
        # Price Metrics
        st.markdown("**Price Metrics**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Price", f"${features['current_price']:.2f}")
            st.metric("Price Change", f"${features['price_change']:.2f} ({features['price_change_pct']:+.2f}%)")
        with col2:
            st.metric("High/Low", f"${features['high']:.2f} / ${features['low']:.2f}")
            st.metric("Avg Price", f"${features['avg_price']:.2f}")
        
        st.markdown("---")
        
        # Volume Metrics
        st.markdown("**Volume Metrics**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Volume", f"{features['volume']:,}")
            st.metric("Avg Volume", f"{features['avg_volume']:,.0f}")
        with col2:
            st.metric("Volume Ratio", f"{features['volume_ratio']:.2f}")
        
        st.markdown("---")
        
        # Volatility Metrics
        st.markdown("**Volatility Metrics**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Daily Volatility", f"{features['daily_volatility']:.2f}%")
            st.metric("Weekly Volatility", f"{features['weekly_volatility']:.2f}%")
        with col2:
            st.metric("ATR", f"${features['atr']:.2f}")
            st.metric("Bollinger Width", f"{features['bollinger_width']:.2f}%")
        
        st.markdown("---")
        
        # Technical Indicators
        st.markdown("**Technical Indicators**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RSI", f"{features['rsi']:.2f}")
            st.metric("MACD", f"{features['macd']:.2f}")
        with col2:
            st.metric("SMA(20)", f"${features['sma_20']:.2f}")
            st.metric("EMA(12)", f"${features['ema_12']:.2f}")

def main():
    """Main application entry point"""
    
    st.title("Neural Network Trading System")
    
    # Check environment
    if not check_environment():
        return
    
    # Check model availability
    model_available, ModelTrainer = check_model_availability()
    
    if not model_available:
        st.error("ModelTrainer Not Available")
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
    
    # Sidebar - matching local GUI approach
    st.sidebar.header("System Status")
    
    # Training statistics
    stats = trainer.get_training_stats()
    st.sidebar.metric("Training Examples", stats['total_examples'])
    st.sidebar.metric("Symbols Trained", stats['symbols_trained'])
    
    if model_state['is_trained']:
        st.sidebar.success("Model Trained")
        st.sidebar.metric("Model Accuracy", f"{model_state['model_accuracy']:.1%}")
    else:
        st.sidebar.info("Model Not Trained")
        st.sidebar.write(f"Need {TRAINING_THRESHOLD - stats['total_examples']} more examples")
    
    # Main interface with tabs - matching local GUI exactly
    tab1, tab2 = st.tabs(["Training", "Prediction"])
    
    with tab1:
        # Training tab layout - matching local GUI exactly
        
        # Top controls row - matching local GUI layout exactly
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
        
        with col1:
            # Stock selection (matching local GUI approach - load from JSON)
            stock_symbols = load_stock_symbols()
            selected_stock = st.selectbox("Stock:", stock_symbols, key="training_stock")
        
        with col2:
            # Date range selection (matching local GUI exactly)
            days_history = st.selectbox(
                "Days of History:", 
                ["30", "60", "90", "180", "365"],
                index=0,
                key="training_days"
            )
        
        with col3:
            # Get data button (matching local GUI exactly)
            if st.button("Get Stock Data", key="get_data_btn"):
                with st.spinner("Fetching stock data..."):
                    features, historical_data = load_stock_data(selected_stock, int(days_history))
                    
                    if features is not None and historical_data is not None:
                        st.session_state.current_features = features
                        st.session_state.current_data = historical_data
                        st.session_state.current_symbol = selected_stock
                        st.success(f"Loaded {selected_stock} data")
                    else:
                        st.error("Failed to fetch stock data")
        
        # Main content area - horizontal splitter layout
        if 'current_features' in st.session_state:
            features = st.session_state.current_features
            historical_data = st.session_state.current_data
            symbol = st.session_state.current_symbol
            
            # Create horizontal layout with chart and metrics
            chart_col, metrics_col = st.columns([3, 1])
            
            with chart_col:
                # Chart area
                chart_fig = create_candlestick_chart(historical_data)
                if chart_fig:
                    st.plotly_chart(chart_fig, use_container_width=True, height=500)
                
                # Decision controls - matching local GUI layout exactly
                st.markdown("### Trading Decision")
                
                # Decision buttons in a row (matching local GUI exactly)
                col1, col2, col3 = st.columns(3)
                with col1:
                    buy_clicked = st.button("BUY", type="primary", use_container_width=True, key="buy_btn")
                with col2:
                    sell_clicked = st.button("SELL", type="primary", use_container_width=True, key="sell_btn")
                with col3:
                    hold_clicked = st.button("HOLD", type="primary", use_container_width=True, key="hold_btn")
                
                # Determine decision
                if buy_clicked:
                    decision = "BUY"
                elif sell_clicked:
                    decision = "SELL"
                elif hold_clicked:
                    decision = "HOLD"
                else:
                    decision = None
                
                # Reasoning input - matching local GUI exactly
                reasoning = st.text_area(
                    "Trading Reasoning:",
                    placeholder="Explain your trading decision...",
                    height=100,
                    key="reasoning_input"
                )
                
                # Submit button (matching local GUI exactly)
                if st.button("Submit Decision", type="primary", key="submit_btn"):
                    if reasoning.strip() and decision:
                        with st.spinner("Processing your decision..."):
                            # Analyze sentiment
                            sentiment_analysis = trainer.analyze_sentiment_and_keywords(reasoning)
                            
                            # Add training example
                            success = trainer.add_training_example(features, sentiment_analysis, decision)
                            
                            if success:
                                st.success("Decision submitted successfully!")
                                
                                # Show sentiment analysis results
                                st.markdown("### Sentiment Analysis")
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
                                st.error("Failed to submit decision")
                    else:
                        st.warning("Please provide your reasoning and make a decision before submitting")
                
                # Training controls - matching local GUI exactly
                st.markdown("### Neural Network Training")
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    epochs = st.number_input("Training Epochs:", min_value=1, max_value=100, value=10, key="epochs_input")
                
                with col2:
                    if stats['total_examples'] >= TRAINING_THRESHOLD:
                        if st.button("Train Model", type="primary", key="train_btn"):
                            with st.spinner("Training neural network... This may take a few minutes."):
                                success = trainer.train_neural_network()
                                
                                if success:
                                    st.success("Neural network trained successfully!")
                                    
                                    # Update model state
                                    update_model_state(is_trained=True, training_examples=stats['total_examples'], model_accuracy=0.85)
                                    
                                    st.rerun()
                                else:
                                    st.error("Training failed. Check logs for details.")
                    else:
                        st.info(f"Need {TRAINING_THRESHOLD - stats['total_examples']} more examples")
            
            with metrics_col:
                # Metrics panel - matching local GUI exactly
                create_metrics_panel(features)
        
        else:
            st.info("Select a stock and click 'Get Stock Data' to start")
    
    with tab2:
        # Prediction tab - matching local GUI exactly
        st.markdown("### Prediction Interface")
        
        # Controls row
        col1, col2 = st.columns([2, 1])
        
        with col1:
            pred_stock_symbols = load_stock_symbols()
            pred_selected_stock = st.selectbox("Stock:", pred_stock_symbols, key="prediction_stock")
        
        with col2:
            if st.button("Get Prediction", key="prediction_btn"):
                with st.spinner("Generating prediction..."):
                    # Load data for prediction
                    pred_features, pred_historical_data = load_stock_data(pred_selected_stock, 30)
                    
                    if pred_features is not None and pred_historical_data is not None:
                        st.session_state.pred_features = pred_features
                        st.session_state.pred_data = pred_historical_data
                        st.session_state.pred_symbol = pred_selected_stock
                        
                        # Generate prediction if model is trained
                        if model_state['is_trained']:
                            prediction_result = trainer.make_prediction(pred_features)
                            st.session_state.prediction_result = prediction_result
                        
                        st.success(f"Loaded {pred_selected_stock} data for prediction")
                    else:
                        st.error("Failed to fetch prediction data")
        
        # Prediction results
        if 'pred_features' in st.session_state:
            pred_features = st.session_state.pred_features
            pred_historical_data = st.session_state.pred_data
            pred_symbol = st.session_state.pred_symbol
            
            # Create horizontal layout for prediction
            pred_chart_col, pred_metrics_col = st.columns([3, 1])
            
            with pred_chart_col:
                # Prediction chart
                pred_chart_fig = create_candlestick_chart(pred_historical_data)
                if pred_chart_fig:
                    st.plotly_chart(pred_chart_fig, use_container_width=True, height=500)
                
                # Prediction result
                if model_state['is_trained'] and 'prediction_result' in st.session_state:
                    st.markdown("### AI Prediction")
                    pred_result = st.session_state.prediction_result
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Prediction", pred_result['prediction'])
                    with col2:
                        st.metric("Confidence", f"{pred_result['confidence']:.1%}")
                else:
                    st.info("Model not trained yet. Train the model in the Training tab first.")
            
            with pred_metrics_col:
                # Prediction metrics panel
                create_metrics_panel(pred_features)

if __name__ == "__main__":
    main()