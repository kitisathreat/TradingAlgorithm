import streamlit as st
import sys
import os
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import plotly.express as px

# Add the orchestrator directory to Python path
REPO_ROOT = Path(__file__).parent.parent.parent
ORCHESTRATOR_PATH = REPO_ROOT / "_2_Orchestrator_And_ML_Python"
sys.path.append(str(ORCHESTRATOR_PATH))

try:
    from deployment_services.live_trader import LiveTrader
    from deployment_services.market_analyzer import MarketAnalyzer
    from interactive_training_app.backend.model_trainer import ModelTrainer
except ImportError as e:
    st.error(f"❌ Failed to import modules: {str(e)}")
    LiveTrader = None
    MarketAnalyzer = None
    ModelTrainer = None

# Page config
st.set_page_config(
    page_title="Trading Algorithm Interface",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'trader' not in st.session_state:
    st.session_state.trader = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = ModelTrainer() if ModelTrainer else None
if 'selected_symbols' not in st.session_state:
    st.session_state.selected_symbols = []
if 'training_count' not in st.session_state:
    st.session_state.training_count = 0
if 'current_features' not in st.session_state:
    st.session_state.current_features = None
if 'current_feature_vector' not in st.session_state:
    st.session_state.current_feature_vector = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = []

def initialize_components():
    """Initialize trading components if not already initialized"""
    if st.session_state.trader is None and LiveTrader is not None:
        try:
            st.session_state.trader = LiveTrader()
            st.session_state.analyzer = MarketAnalyzer()
        except Exception as e:
            st.error(f"❌ Failed to initialize trading components: {str(e)}")

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["Trading Dashboard", "Model Training", "Settings"])

with tab1:
    st.title("📈 Trading Dashboard")
    
    # Market Overview
    st.header("Market Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Market Status", value="Open", delta="Active")
    with col2:
        st.metric(label="Active Positions", value="0", delta="0")
    with col3:
        st.metric(label="Current Balance", value="$1000.00", delta="+$50.00")
    
    # Trading Chart
    st.subheader("Live Trading Chart")
    if st.session_state.selected_symbols:
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        prices = pd.Series(index=dates, data=[100 + i + i**1.5 for i in range(len(dates))])
        
        fig = go.Figure(data=[go.Candlestick(
            x=dates,
            open=prices,
            high=prices + 2,
            low=prices - 2,
            close=prices + 1
        )])
        
        fig.update_layout(
            title=f"Price Chart for {', '.join(st.session_state.selected_symbols)}",
            yaxis_title="Price ($)",
            xaxis_title="Date",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Please select symbols in the settings tab to view the trading chart")

with tab2:
    st.title("🎯 Model Training Interface")
    
    if st.session_state.model_trainer is None:
        st.error("❌ Model trainer not initialized. Please check dependencies.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Training Example")
            
            if st.button("Get New Training Example"):
                features, feature_vector, actual_result = st.session_state.model_trainer.get_random_stock_data()
                if features:
                    st.session_state.current_features = features
                    st.session_state.current_feature_vector = feature_vector
                    
                    # Display stock information
                    st.write("### Stock Information")
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric("Symbol", features['Symbol'])
                        st.metric("Current Price", f"${features['Current Price']:.2f}")
                    with metrics_col2:
                        st.metric("Price Change", f"{features['Price Change %']:.2f}%")
                        st.metric("RSI", f"{features['RSI']:.2f}")
                    with metrics_col3:
                        st.metric("Volume", f"{features['Volume']:,}")
                        st.metric("Volume Change", f"{features['Volume Change %']:.2f}%")
                    
                    # Technical indicators
                    st.write("### Technical Indicators")
                    st.write(f"- SMA(5): ${features['SMA_5']:.2f}")
                    st.write(f"- SMA(20): ${features['SMA_20']:.2f}")
                    
                    # User input
                    st.write("### Your Trading Decision")
                    st.write("Based on this information, what would you do? (Type your decision and explanation)")
                    user_input = st.text_area("Your decision", height=100, 
                                            help="Example: 'Buy because the price is above both SMAs and RSI shows oversold conditions'")
                    
                    if st.button("Submit Decision"):
                        if user_input:
                            decision = st.session_state.model_trainer.parse_user_decision(user_input)
                            st.session_state.model_trainer.add_training_example(feature_vector, decision)
                            st.session_state.training_count += 1
                            st.session_state.training_history.append({
                                'Symbol': features['Symbol'],
                                'Date': features['Date'],
                                'Decision': ['Buy', 'Sell', 'Hold'][decision],
                                'Actual': 'Up' if actual_result else 'Down'
                            })
                            st.success(f"✅ Training example #{st.session_state.training_count} added!")
                        else:
                            st.warning("⚠️ Please enter your decision first")
                else:
                    st.error("❌ Failed to fetch stock data. Please try again.")
        
        with col2:
            st.subheader("Training Progress")
            st.write(f"Total examples: {st.session_state.training_count}")
            
            if st.session_state.training_count >= 5:
                if st.button("Train Model"):
                    with st.spinner("Training model..."):
                        results = st.session_state.model_trainer.train_model()
                        st.write("### Training Results")
                        st.write(f"- Loss: {results['loss']:.4f}")
                        st.write(f"- Accuracy: {results['accuracy']:.2%}")
                        if results['val_accuracy']:
                            st.write(f"- Validation Accuracy: {results['val_accuracy']:.2%}")
                
                if st.button("Save Model"):
                    model_path = os.path.join(REPO_ROOT, "_2_Orchestrator_And_ML_Python", "models", "trading_model.h5")
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    st.session_state.model_trainer.save_model(model_path)
                    st.success(f"✅ Model saved to {model_path}")
            else:
                st.info(f"⚠️ Need at least 5 examples to train (have {st.session_state.training_count})")
            
            if st.session_state.training_history:
                st.write("### Training History")
                history_df = pd.DataFrame(st.session_state.training_history)
                st.dataframe(history_df, use_container_width=True)

with tab3:
    st.title("⚙️ Settings")
    
    # Authentication section
    st.subheader("🔑 Authentication")
    api_key = st.text_input("API Key", type="password")
    api_secret = st.text_input("API Secret", type="password")
    
    if st.button("Connect"):
        if api_key and api_secret:
            initialize_components()
            if st.session_state.trader:
                try:
                    st.success("✅ Successfully connected!")
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")
        else:
            st.warning("⚠️ Please enter both API key and secret")
    
    # Trading parameters
    st.subheader("📊 Trading Parameters")
    risk_level = st.slider("Risk Level", 1, 10, 5)
    investment_amount = st.number_input("Investment Amount ($)", min_value=0.0, value=1000.0)
    
    # Symbol selection
    st.subheader("🎯 Symbol Selection")
    symbols = st.multiselect(
        "Select Trading Symbols",
        ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "BTC-USD", "ETH-USD"],
        default=st.session_state.selected_symbols
    )
    st.session_state.selected_symbols = symbols

# Footer
st.markdown("---")
st.markdown("*Trading Algorithm Interface - Powered by Streamlit*") 