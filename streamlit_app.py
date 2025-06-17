#!/usr/bin/env python3
"""
Advanced Neural Network Trading System - Streamlit Interface
Main entry point for both local and Streamlit Cloud deployment
"""

import streamlit as st
import sys
import os
from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Add the orchestrator path to sys.path
REPO_ROOT = Path(__file__).parent
ORCHESTRATOR_PATH = REPO_ROOT / "_2_Orchestrator_And_ML_Python"
sys.path.append(str(ORCHESTRATOR_PATH))

# Import the ModelTrainer
try:
    from interactive_training_app.backend.model_trainer import ModelTrainer
    MODEL_TRAINER_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import ModelTrainer: {e}")
    MODEL_TRAINER_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="Advanced Neural Network Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .control-panel {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
    .decision-button {
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_model_trainer():
    """Initialize the ModelTrainer instance"""
    if not MODEL_TRAINER_AVAILABLE:
        return None
    
    try:
        trainer = ModelTrainer()
        return trainer
    except Exception as e:
        st.error(f"Failed to initialize ModelTrainer: {e}")
        return None

def display_header():
    """Display the main header"""
    st.markdown('<h1 class="main-header">🧠 Advanced Neural Network Trading System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Learn from human trading intuition using AI-powered sentiment analysis and technical indicators
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_model_status(trainer):
    """Display current model status"""
    if trainer is None:
        st.error("Model trainer not available")
        return
    
    st.subheader("📊 Model Status")
    
    # Get model state
    model_state = trainer.get_model_state()
    training_stats = trainer.get_training_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Training Examples",
            value=training_stats.get('total_examples', 0),
            help="Number of training examples collected"
        )
    
    with col2:
        st.metric(
            label="Model Trained",
            value="✅ Yes" if model_state.get('is_trained', False) else "❌ No",
            help="Whether the neural network has been trained"
        )
    
    with col3:
        accuracy = model_state.get('model_accuracy', 0.0)
        st.metric(
            label="Model Accuracy",
            value=f"{accuracy:.2%}" if accuracy > 0 else "N/A",
            help="Current model accuracy"
        )
    
    with col4:
        st.metric(
            label="Last Updated",
            value=model_state.get('last_updated', 'Never'),
            help="Last time the model was updated"
        )

def display_data_controls(trainer):
    """Display data loading controls similar to local GUI"""
    if trainer is None:
        st.error("Model trainer not available")
        return
    
    st.subheader("📈 Stock Data Controls")
    
    with st.container():
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        
        # Stock selection
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            stock_symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 
                'AMD', 'INTC', 'CRM', 'ADBE', 'PYPL', 'UBER', 'SHOP', 'SQ',
                'SPY', 'QQQ', 'IWM', 'VTI'  # ETFs for more reliable data
            ]
            selected_stock = st.selectbox("Stock Symbol:", stock_symbols, index=0)
        
        with col2:
            # Date range selection
            date_range_options = ["30", "60", "90", "180", "365", "Custom"]
            selected_date_range = st.selectbox("Days of History:", date_range_options, index=0)
        
        with col3:
            # Custom days input (only visible when "Custom" is selected)
            if selected_date_range == "Custom":
                custom_days = st.number_input("Custom Days:", min_value=1, max_value=365, value=30)
                days_to_fetch = custom_days
            else:
                days_to_fetch = int(selected_date_range)
                st.write(f"Days: {days_to_fetch}")
        
        with col4:
            st.write("")  # Spacer
            if st.button("📊 Get Stock Data", type="primary"):
                return selected_stock, days_to_fetch
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return None, None

def load_stock_data(trainer, symbol, days):
    """Load stock data using the same logic as local GUI"""
    try:
        # Import the date range utilities
        from date_range_utils import find_available_data_range, validate_date_range
        
        # Get random date range within the last 25 years
        start_date, end_date = find_available_data_range(symbol, days, max_years_back=25)
        
        # Validate the date range
        if not validate_date_range(start_date, end_date, symbol):
            st.error(f"Invalid date range generated for {symbol}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            return None, None, None
        
        with st.spinner(f"Loading {days} days of data for {symbol}..."):
            # Use the ModelTrainer's method to get data
            data = trainer.get_historical_stock_data(symbol, years_back=25)
            
            if data is not None and not data.empty:
                # Filter data to the requested date range
                data = data[(data.index >= start_date) & (data.index <= end_date)]
                
                if len(data) > 0:
                    # Create stock_info structure similar to get_random_stock_data
                    current_price = data['Close'].iloc[-1]
                    volume = data['Volume'].iloc[-1] if 'Volume' in data.columns else 0
                    
                    # Calculate technical indicators
                    technical_indicators = trainer._calculate_performance_metrics(data)
                    
                    # Create stock_info structure
                    stock_info = {
                        'current_price': current_price,
                        'volume': volume,
                        'market_cap': current_price * 1e9,  # Placeholder
                        'technical_indicators': {
                            'rsi': technical_indicators.get('rsi', 50.0),
                            'macd': technical_indicators.get('macd', 0.0),
                            'bollinger_upper': current_price * 1.02,  # Placeholder
                            'bollinger_lower': current_price * 0.98,  # Placeholder
                        }
                    }
                    
                    st.success(f"✅ Successfully loaded data for {symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                    return stock_info, data, symbol
                else:
                    st.error(f"No data available for {symbol} in the specified date range")
                    return None, None, None
            else:
                st.error(f"Failed to load data for {symbol}")
                return None, None, None
                
    except Exception as e:
        st.error(f"Error loading data: {str(e)}\n\nTry:\n- Different stock symbol\n- Fewer days of history\n- Check internet connection")
        return None, None, None

def display_stock_analysis(stock_info, stock_data, symbol):
    """Display stock analysis similar to local GUI"""
    st.markdown(f"### 📈 {symbol} Stock Analysis")
    
    # Display stock info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${stock_info['current_price']:.2f}")
    with col2:
        st.metric("Volume", f"{stock_info['volume']:,}")
    with col3:
        st.metric("Market Cap", f"${stock_info['market_cap']/1e9:.1f}B")
    
    # Display technical indicators
    st.markdown("#### Technical Indicators")
    indicators = stock_info['technical_indicators']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RSI", f"{indicators['rsi']:.2f}")
    with col2:
        st.metric("MACD", f"{indicators['macd']:.4f}")
    with col3:
        st.metric("Bollinger Upper", f"${indicators['bollinger_upper']:.2f}")
    with col4:
        st.metric("Bollinger Lower", f"${indicators['bollinger_lower']:.2f}")
    
    # Display price chart
    st.markdown("#### Price Chart")
    if not stock_data.empty:
        # Create a simple line chart
        chart_data = stock_data[['Close']].copy()
        chart_data.index = pd.to_datetime(chart_data.index)
        st.line_chart(chart_data)

def display_trading_decision_interface(trainer, stock_info, stock_data, symbol):
    """Display trading decision interface similar to local GUI"""
    st.markdown("#### 🤔 What would you do?")
    
    # Trading decision buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        buy_clicked = st.button("🟢 BUY", type="primary", use_container_width=True)
    with col2:
        sell_clicked = st.button("🔴 SELL", type="primary", use_container_width=True)
    with col3:
        hold_clicked = st.button("🟡 HOLD", type="primary", use_container_width=True)
    
    # Determine selected decision
    if buy_clicked:
        selected_decision = "BUY"
    elif sell_clicked:
        selected_decision = "SELL"
    elif hold_clicked:
        selected_decision = "HOLD"
    else:
        selected_decision = None
    
    # Trading description
    trading_description = st.text_area(
        "Trading Reasoning:",
        placeholder="Explain your trading decision...",
        height=100
    )
    
    # Submit decision
    if st.button("📚 Submit Decision", type="primary"):
        if selected_decision and trading_description.strip():
            with st.spinner("Processing training example..."):
                try:
                    # Get features from the current stock data
                    features = trainer._calculate_performance_metrics(stock_data)
                    
                    # Analyze sentiment
                    sentiment_analysis = trainer.analyze_sentiment_and_keywords(trading_description)
                    
                    # Add training example
                    success = trainer.add_training_example(features, sentiment_analysis, selected_decision)
                    
                    if success:
                        st.success("✅ Training example added successfully!")
                        st.balloons()
                        
                        # Clear session state to get new data
                        if 'stock_info' in st.session_state:
                            del st.session_state.stock_info
                        if 'stock_data' in st.session_state:
                            del st.session_state.stock_data
                        if 'symbol' in st.session_state:
                            del st.session_state.symbol
                    else:
                        st.error("❌ Failed to add training example")
                        
                except Exception as e:
                    st.error(f"Error processing training example: {e}")
        else:
            st.warning("Please select a decision and provide reasoning")

def display_training_controls(trainer):
    """Display training controls"""
    if trainer is None:
        st.error("Model trainer not available")
        return
    
    st.subheader("🏋️ Model Training")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        epochs = st.number_input("Training Epochs:", min_value=1, max_value=100, value=10)
    
    with col2:
        if st.button("🧠 Train Neural Network", type="primary"):
            if trainer.training_examples:
                with st.spinner("Training neural network..."):
                    try:
                        success = trainer.train_neural_network()
                        if success:
                            st.success("✅ Neural network trained successfully!")
                            st.balloons()
                        else:
                            st.error("❌ Failed to train neural network")
                    except Exception as e:
                        st.error(f"Error during training: {e}")
            else:
                st.warning("No training examples available. Please add some training examples first.")
    
    with col3:
        if st.button("📊 View Training Statistics"):
            if trainer.training_examples:
                stats = trainer.get_training_stats()
                
                st.markdown("#### Training Statistics")
                
                # Decision distribution
                st.markdown("**Decision Distribution:**")
                decision_dist = stats.get('decision_distribution', {})
                for decision, count in decision_dist.items():
                    st.write(f"- {decision}: {count}")
                
                # Sentiment distribution
                st.markdown("**Sentiment Distribution:**")
                sentiment_dist = stats.get('sentiment_distribution', {})
                for sentiment, count in sentiment_dist.items():
                    st.write(f"- {sentiment}: {count}")
            else:
                st.info("No training data available")

def display_prediction_interface(trainer):
    """Display prediction interface"""
    if trainer is None:
        st.error("Model trainer not available")
        return
    
    st.subheader("🔮 Make Predictions")
    
    # Check if model is trained
    model_state = trainer.get_model_state()
    if not model_state.get('is_trained', False):
        st.warning("Model needs to be trained before making predictions. Please add training examples and train the model.")
        return
    
    # Get stock data for prediction
    if st.button("🎯 Get Stock for Prediction"):
        with st.spinner("Fetching stock data for prediction..."):
            try:
                stock_info, data, symbol = trainer.get_random_stock_data()
                
                # Store in session state
                st.session_state.prediction_stock_info = stock_info
                st.session_state.prediction_stock_data = data
                st.session_state.prediction_symbol = symbol
                
                st.success(f"✅ Fetched data for {symbol}")
                
            except Exception as e:
                st.error(f"Failed to fetch stock data: {e}")
    
    # Display prediction if data is available
    if hasattr(st.session_state, 'prediction_stock_data') and st.session_state.prediction_stock_data is not None:
        st.markdown(f"### 📊 {st.session_state.prediction_symbol} Prediction")
        
        # Display stock info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${st.session_state.prediction_stock_info['current_price']:.2f}")
        with col2:
            st.metric("Volume", f"{st.session_state.prediction_stock_info['volume']:,}")
        with col3:
            st.metric("Market Cap", f"${st.session_state.prediction_stock_info['market_cap']/1e9:.1f}B")
        
        # Make prediction
        if st.button("🔮 Make Prediction", type="primary"):
            with st.spinner("Making prediction..."):
                try:
                    # Get features
                    features = trainer._calculate_performance_metrics(st.session_state.prediction_stock_data)
                    
                    # Make prediction
                    prediction_result = trainer.make_prediction(features)
                    
                    # Display prediction
                    st.markdown("#### 🤖 AI Prediction")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        decision = prediction_result.get('predicted_decision', 'UNKNOWN')
                        confidence = prediction_result.get('confidence', 0.0)
                        
                        if decision == 'BUY':
                            st.markdown('<div class="success-message">', unsafe_allow_html=True)
                            st.markdown(f"**Decision: {decision}**")
                            st.markdown(f"Confidence: {confidence:.2%}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        elif decision == 'SELL':
                            st.markdown('<div class="error-message">', unsafe_allow_html=True)
                            st.markdown(f"**Decision: {decision}**")
                            st.markdown(f"Confidence: {confidence:.2%}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="warning-message">', unsafe_allow_html=True)
                            st.markdown(f"**Decision: {decision}**")
                            st.markdown(f"Confidence: {confidence:.2%}")
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**Technical Analysis:**")
                        st.write(f"RSI: {features.get('rsi', 'N/A'):.2f}")
                        st.write(f"MACD: {features.get('macd', 'N/A'):.4f}")
                        st.write(f"Volume Ratio: {features.get('volume_ratio', 'N/A'):.2f}")
                    
                    with col3:
                        st.markdown("**Market Conditions:**")
                        st.write(f"Trend: {features.get('trend', 'N/A')}")
                        st.write(f"Volatility: {features.get('volatility', 'N/A'):.2f}")
                        st.write(f"Price Change: {features.get('price_change', 'N/A'):.2%}")
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")

def main():
    """Main function"""
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    
    # Display header
    display_header()
    
    # Initialize model trainer
    if not st.session_state.initialized:
        with st.spinner("Initializing model trainer..."):
            trainer = initialize_model_trainer()
            st.session_state.trainer = trainer
            st.session_state.initialized = True
    
    trainer = st.session_state.get('trainer')
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Status", "📈 Data & Training", "🏋️ Model", "🔮 Predictions"])
    
    with tab1:
        display_model_status(trainer)
    
    with tab2:
        # Data controls
        stock_symbol, days = display_data_controls(trainer)
        
        # Load data if requested
        if stock_symbol and days:
            stock_info, stock_data, symbol = load_stock_data(trainer, stock_symbol, days)
            if stock_info and stock_data is not None:
                # Store in session state
                st.session_state.stock_info = stock_info
                st.session_state.stock_data = stock_data
                st.session_state.symbol = symbol
        
        # Display stock analysis if data is available
        if hasattr(st.session_state, 'stock_data') and st.session_state.stock_data is not None:
            display_stock_analysis(
                st.session_state.stock_info,
                st.session_state.stock_data,
                st.session_state.symbol
            )
            
            # Display trading decision interface
            display_trading_decision_interface(
                trainer,
                st.session_state.stock_info,
                st.session_state.stock_data,
                st.session_state.symbol
            )
    
    with tab3:
        display_training_controls(trainer)
    
    with tab4:
        display_prediction_interface(trainer)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>Advanced Neural Network Trading System | Built with Streamlit and TensorFlow</p>
        <p>Learn from human intuition, predict with AI precision</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 