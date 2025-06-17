"""
Trading Algorithm Streamlit App
Main interface for the trading algorithm system.
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
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    
    /* Enhanced styling for expandable boxes */
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 0.5rem !important;
        font-weight: bold !important;
        color: #495057 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #ffffff !important;
        border: 1px solid #dee2e6 !important;
        border-top: none !important;
        border-radius: 0 0 0.5rem 0.5rem !important;
        padding: 1rem !important;
    }
    
    /* Chart container styling */
    .element-container {
        border-radius: 0.5rem;
        overflow: hidden;
    }
    
    /* Metric styling */
    .metric-container {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    
    /* Two-column layout improvements */
    .row-widget.stHorizontal {
        gap: 1rem;
    }
    
    /* Custom button styling */
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
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
    
    # Get model state and metadata
    model_state = trainer.get_model_state()
    training_stats = trainer.get_training_stats()
    model_metadata = trainer.get_model_metadata()
    
    # Display model metadata
    st.markdown("#### 📋 Model Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Model Name",
            value=model_metadata.get('name', 'Default Model'),
            help="Name of the current model"
        )
    
    with col2:
        st.metric(
            label="Author",
            value=model_metadata.get('author', 'Unknown'),
            help="Author of the model"
        )
    
    with col3:
        st.metric(
            label="Model Type",
            value=model_state.get('model_type', 'Unknown'),
            help="Type of neural network architecture"
        )
    
    with col4:
        st.metric(
            label="Min Epochs",
            value=model_metadata.get('min_epochs', 50),
            help="Minimum recommended training epochs"
        )
    
    # Show training subjects
    training_subjects = model_metadata.get('training_subjects', [])
    if training_subjects:
        st.markdown("**Training Subjects:**")
        for subject in training_subjects:
            st.write(f"• {subject}")
    
    st.markdown("---")
    
    # Display training metrics
    st.markdown("#### 📈 Training Metrics")
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
    """Display data loading controls with enhanced stock selection"""
    if trainer is None:
        st.error("Model trainer not available")
        return
    
    st.subheader("📈 Stock Data Controls")
    
    # Import stock selection utilities
    try:
        sys.path.append(str(ORCHESTRATOR_PATH))
        from stock_selection_utils import StockSelectionManager
        stock_manager = StockSelectionManager()
    except Exception as e:
        st.error(f"Error loading stock selection utilities: {e}")
        return None, None
    
    with st.container():
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        
        # Enhanced stock selection
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            # Get stock options for dropdown
            stock_options = stock_manager.get_stock_options()
            stock_display_names = [option['name'] for option in stock_options]
            
            selected_stock_index = st.selectbox(
                "Stock Selection:", 
                range(len(stock_display_names)),
                format_func=lambda x: stock_display_names[x],
                help="Select from S&P100 stocks, random pick, optimized pick, or custom ticker"
            )
            
            selected_stock_option = stock_options[selected_stock_index]
            selected_stock_symbol = selected_stock_option['symbol']
            
            # Show description
            st.caption(f"ℹ️ {selected_stock_option['description']}")
            
            # Handle special stock selection cases
            if selected_stock_symbol == "random":
                if st.button("🎲 Get Random Stock", type="secondary"):
                    symbol, name = stock_manager.get_random_stock()
                    st.success(f"🎲 Random pick: {symbol} ({name})")
                    selected_stock_symbol = symbol
                    
            elif selected_stock_symbol == "optimized":
                if st.button("🚀 Get Optimized Pick", type="secondary"):
                    symbol, name = stock_manager.get_optimized_pick(trainer)
                    st.success(f"🚀 Optimized pick: {symbol} ({name})")
                    selected_stock_symbol = symbol
                    
            elif selected_stock_symbol == "custom":
                # Custom ticker input
                custom_ticker = st.text_input(
                    "Enter Stock Ticker:",
                    placeholder="e.g., AAPL, TSLA, GOOGL",
                    help="Enter any valid stock ticker symbol"
                )
                
                if custom_ticker:
                    # Validate the custom ticker
                    is_valid, message, company_name = stock_manager.validate_custom_ticker(custom_ticker)
                    
                    if is_valid:
                        st.success(message)
                        selected_stock_symbol = custom_ticker.upper()
                    else:
                        st.error(message)
                        selected_stock_symbol = None
                        
                    # Show validation result
                    if company_name:
                        st.info(f"Company: {company_name}")
        
        with col2:
            # Date range selection
            date_range_options = ["30", "60", "90", "180", "365", "Custom"]
            selected_date_range = st.selectbox("Days of History:", date_range_options, index=0, help="Number of trading days to analyze")
        
        with col3:
            # Custom days input (only visible when "Custom" is selected)
            if selected_date_range == "Custom":
                custom_days = st.number_input("Custom Days:", min_value=1, max_value=365, value=30, help="Enter custom number of days")
                days_to_fetch = custom_days
            else:
                days_to_fetch = int(selected_date_range)
                st.write(f"Days: {days_to_fetch}")
        
        with col4:
            st.write("")  # Spacer
            if st.button("📊 Get Stock Data", type="primary", use_container_width=True):
                # Ensure we have a valid symbol
                if selected_stock_symbol and selected_stock_symbol not in ["random", "optimized", "custom"]:
                    return selected_stock_symbol, days_to_fetch
                else:
                    st.error("Please select a valid stock symbol first")
                    return None, None
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return None, None

def load_stock_data(trainer, symbol, days):
    """Load stock data using the same logic as local GUI"""
    try:
        # Validate symbol parameter
        if symbol is None or not isinstance(symbol, str) or not symbol.strip():
            st.error("Invalid stock symbol provided")
            return None, None, None
        
        symbol = symbol.strip().upper()
        
        # Import the date range utilities from the orchestrator directory
        sys.path.append(str(ORCHESTRATOR_PATH))
        from date_range_utils import find_available_data_range, validate_date_range
        import yfinance as yf
        
        # Get random date range with no limit on how far back we can look
        start_date, end_date = find_available_data_range(symbol, days, max_years_back=None)
        
        # Validate the date range
        if not validate_date_range(start_date, end_date, symbol):
            st.error(f"Invalid date range generated for {symbol}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            return None, None, None
        
        with st.spinner(f"Loading {days} days of data for {symbol}..."):
            # Use yfinance directly like the local GUI does
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data is not None and not data.empty:
                # Ensure df.index is timezone-aware (UTC) like the local GUI
                if data.index.tz is None:
                    data.index = data.index.tz_localize('UTC')
                else:
                    data.index = data.index.tz_convert('UTC')
                
                # Validate that data is not newer than our end_date
                if data.index.max() > end_date:
                    data = data[data.index <= end_date]
                    if data.empty:
                        st.error(f"No valid data found for {symbol} after filtering dates")
                        return None, None, None
                
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
    """Display stock analysis with candlestick chart and financial metrics"""
    st.markdown(f"### 📈 {symbol} Stock Analysis")
    
    # Create two-column layout: chart on left, metrics on right
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create candlestick chart using Plotly
        if not stock_data.empty:
            fig = create_candlestick_chart(stock_data, symbol)
            st.plotly_chart(fig, use_container_width=True, height=500)
    
    with col2:
        # Financial metrics in expandable boxes
        display_financial_metrics(stock_data, stock_info)

def create_candlestick_chart(df, symbol):
    """Create a professional candlestick chart using Plotly"""
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Price Chart', 'Volume'),
        row_width=[0.7, 0.3]
    )
    
    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC',
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350',
            increasing_fillcolor='#26A69A',
            decreasing_fillcolor='#EF5350'
        ),
        row=1, col=1
    )
    
    # Add volume bars
    colors = ['#26A69A' if close >= open else '#EF5350' 
              for close, open in zip(df['Close'], df['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Add moving averages
    if len(df) >= 20:
        sma_20 = df['Close'].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=sma_20,
                name='SMA(20)',
                line=dict(color='#FF9800', width=2)
            ),
            row=1, col=1
        )
    
    if len(df) >= 12:
        ema_12 = df['Close'].ewm(span=12).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=ema_12,
                name='EMA(12)',
                line=dict(color='#2196F3', width=2)
            ),
            row=1, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Stock Analysis',
        yaxis_title='Price (USD)',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    # Update axes
    fig.update_xaxes(gridcolor='lightgray', gridwidth=0.5)
    fig.update_yaxes(gridcolor='lightgray', gridwidth=0.5)
    
    return fig

def display_financial_metrics(df, stock_info):
    """Display financial metrics in expandable boxes"""
    if df is None or df.empty:
        st.warning("No data available for metrics")
        return
    
    try:
        # Calculate metrics
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
        
        high_price = df['High'].max()
        low_price = df['Low'].min()
        avg_price = df['Close'].mean()
        
        current_volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # Volatility metrics
        daily_returns = df['Close'].pct_change().dropna()
        daily_vol = daily_returns.std() * 100
        weekly_vol = daily_returns.rolling(5).std().iloc[-1] * 100 if len(daily_returns) >= 5 else daily_vol
        
        # ATR calculation
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(14).mean().iloc[-1] if len(df) >= 14 else true_range.mean()
        
        # Bollinger Bands width
        sma_20 = df['Close'].rolling(20).mean()
        std_20 = df['Close'].rolling(20).std()
        bollinger_width = (std_20.iloc[-1] / sma_20.iloc[-1]) * 100 if not pd.isna(sma_20.iloc[-1]) else 0
        
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
        sma_20_current = sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else current_price
        ema_12_current = ema_12.iloc[-1] if not pd.isna(ema_12.iloc[-1]) else current_price
        
        # Price Metrics
        with st.expander("💰 Price Metrics", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
                st.metric("High", f"${high_price:.2f}")
            with col2:
                st.metric("Price Change", f"${price_change:.2f}", f"{price_change_pct:+.2f}%")
                st.metric("Low", f"${low_price:.2f}")
            
            st.metric("Average Price", f"${avg_price:.2f}")
        
        # Volume Metrics
        with st.expander("📊 Volume Metrics", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Volume", f"{current_volume:,}")
            with col2:
                st.metric("Volume Ratio", f"{volume_ratio:.2f}")
            
            st.metric("Avg Volume", f"{avg_volume:,.0f}")
        
        # Volatility Metrics
        with st.expander("📈 Volatility Metrics", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Daily Volatility", f"{daily_vol:.2f}%")
                st.metric("ATR", f"${atr:.2f}")
            with col2:
                st.metric("Weekly Volatility", f"{weekly_vol:.2f}%")
                st.metric("Bollinger Width", f"{bollinger_width:.2f}%")
        
        # Technical Indicators
        with st.expander("🔧 Technical Indicators", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RSI", f"{current_rsi:.2f}")
                st.metric("SMA(20)", f"${sma_20_current:.2f}")
            with col2:
                st.metric("MACD", f"{current_macd:.4f}")
                st.metric("EMA(12)", f"${ema_12_current:.2f}")
            
            # RSI interpretation
            if current_rsi > 70:
                st.warning("RSI indicates overbought conditions")
            elif current_rsi < 30:
                st.info("RSI indicates oversold conditions")
            else:
                st.success("RSI in neutral range")
        
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")

def display_trading_decision_interface(trainer, stock_info, stock_data, symbol):
    """Display trading decision interface with improved layout"""
    st.markdown("#### 🤔 What would you do?")
    
    # Initialize session state for tracking decisions
    if 'selected_decision' not in st.session_state:
        st.session_state.selected_decision = None
    
    # Create a container for the decision interface
    with st.container():
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        
        # Trading decision buttons in a row
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if st.button("🟢 BUY", type="primary", use_container_width=True):
                st.session_state.selected_decision = "BUY"
                st.rerun()
        with col2:
            if st.button("🔴 SELL", type="primary", use_container_width=True):
                st.session_state.selected_decision = "SELL"
                st.rerun()
        with col3:
            if st.button("🟡 HOLD", type="primary", use_container_width=True):
                st.session_state.selected_decision = "HOLD"
                st.rerun()
        with col4:
            # Spacer column for better alignment
            st.write("")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show selected decision
        if st.session_state.selected_decision:
            st.success(f"Selected Decision: {st.session_state.selected_decision}")
        
        # Trading description in an expandable section
        with st.expander("📝 Trading Reasoning", expanded=True):
            trading_description = st.text_area(
                "Explain your trading decision:",
                placeholder="Describe your analysis, technical indicators, market sentiment, risk factors, and reasoning for your decision...",
                height=120,
                help="Be detailed about your analysis. Include technical indicators, market conditions, risk assessment, and your overall strategy."
            )
            
            # Submit decision button
            if st.button("📚 Submit Decision", type="primary", use_container_width=True):
                if st.session_state.selected_decision and trading_description.strip():
                    with st.spinner("Processing training example..."):
                        try:
                            # Get features from the current stock data
                            features = trainer._calculate_performance_metrics(stock_data)
                            
                            # Analyze sentiment
                            sentiment_analysis = trainer.analyze_sentiment_and_keywords(trading_description)
                            
                            # Add training example
                            success = trainer.add_training_example(features, sentiment_analysis, st.session_state.selected_decision)
                            
                            if success:
                                st.success("✅ Training example added successfully!")
                                st.balloons()
                                
                                # Clear session state
                                st.session_state.selected_decision = None
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
                elif not st.session_state.selected_decision:
                    st.warning("Please select a trading decision (BUY/SELL/HOLD)")
                else:
                    st.warning("Please provide reasoning for your trading decision")

def display_training_controls(trainer):
    """Display training controls with model selection and additive training info"""
    if trainer is None:
        st.error("Model trainer not available")
        return
    
    st.subheader("🏋️ Model Training")
    
    # Show additive training information
    st.info(f"📚 **Additive Training Enabled**: Your training data is preserved across sessions. Current total: {len(trainer.training_examples)} examples")
    
    with st.container():
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        
        # Model selection section
        st.markdown("#### 🧠 Neural Network Model Selection")
        
        # Get available models
        available_models = trainer.get_available_models()
        current_model_info = trainer.get_current_model_info()
        
        # Create model selection interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Model selection dropdown
            model_options = {info['name']: model_type for model_type, info in available_models.items()}
            selected_model_name = st.selectbox(
                "Select Neural Network Model:",
                options=list(model_options.keys()),
                index=list(model_options.values()).index(current_model_info.get('model_type', 'standard')),
                help="Choose the neural network architecture for training"
            )
            
            selected_model_type = model_options[selected_model_name]
            
            # Show model information
            model_info = available_models[selected_model_type]
            st.markdown(f"**Description:** {model_info['description']}")
            st.markdown(f"**Best for:** {model_info['best_for']}")
            st.markdown(f"**Training time:** {model_info['training_time']}")
            st.markdown(f"**Complexity:** {model_info['complexity']}")
        
        with col2:
            # Change model button
            if st.button("🔄 Change Model", type="secondary", use_container_width=True):
                if trainer.change_model_type(selected_model_type):
                    st.success(f"✅ Model changed to {selected_model_name}")
                    st.rerun()
                else:
                    st.error("❌ Failed to change model")
            
            # Show current model info
            st.markdown("**Current Model:**")
            st.write(f"Type: {current_model_info.get('model_type', 'Unknown')}")
            st.write(f"Trained: {'✅ Yes' if current_model_info.get('is_trained', False) else '❌ No'}")
            st.write(f"Parameters: {current_model_info.get('total_params', 0):,}")
        
        st.markdown("---")
        
        # Training controls section
        st.markdown("#### 🚀 Training Controls")
        
        # Get model metadata
        model_metadata = trainer.get_model_metadata()
        current_model_info = trainer.get_current_model_info()
        
        # Show model metadata
        st.markdown("**Current Model Details:**")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.write(f"**Name:** {model_metadata.get('name', 'Default Model')}")
            st.write(f"**Author:** {model_metadata.get('author', 'Unknown')}")
        
        with col_b:
            st.write(f"**Type:** {current_model_info.get('model_type', 'Unknown')}")
            st.write(f"**Min Epochs:** {model_metadata.get('min_epochs', 50)}")
        
        with col_c:
            st.write(f"**Trained:** {'✅ Yes' if current_model_info.get('is_trained', False) else '❌ No'}")
            st.write(f"**Parameters:** {current_model_info.get('total_params', 0):,}")
        
        # Show training subjects
        training_subjects = model_metadata.get('training_subjects', [])
        if training_subjects:
            st.markdown("**Training Subjects:**")
            for subject in training_subjects:
                st.write(f"• {subject}")
        
        st.markdown("---")
        
        # Training controls in a grid layout
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # Use minimum epochs from model metadata
            min_epochs = model_metadata.get('min_epochs', 50)
            max_epochs = 500  # Reasonable maximum
            
            epochs = st.number_input(
                "Training Epochs:", 
                min_value=min_epochs, 
                max_value=max_epochs, 
                value=min_epochs, 
                help=f"Minimum recommended: {min_epochs} epochs for this model type"
            )
        
        with col2:
            if st.button("🧠 Train Neural Network", type="primary", use_container_width=True):
                if trainer.training_examples:
                    with st.spinner(f"Training {selected_model_name} neural network..."):
                        try:
                            success = trainer.train_neural_network(epochs=epochs)
                            if success:
                                st.success("✅ Neural network trained successfully!")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error("❌ Failed to train neural network")
                        except Exception as e:
                            st.error(f"Error during training: {e}")
                else:
                    st.warning("No training examples available. Please add some training examples first.")
        
        with col3:
            if st.button("📊 View Training Statistics", type="secondary", use_container_width=True):
                if trainer.training_examples:
                    stats = trainer.get_training_stats()
                    
                    st.markdown("#### 📈 Training Statistics")
                    
                    # Basic stats
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Total Examples", stats.get('total_examples', 0))
                        st.metric("Unique Symbols", stats.get('symbols_trained', 0))
                    
                    with col_b:
                        model_state = stats.get('model_state', {})
                        st.metric("Model Accuracy", f"{model_state.get('model_accuracy', 0):.1%}")
                        st.metric("Model Type", model_state.get('model_type', 'Unknown'))
                    
                    # Decision distribution
                    st.markdown("**Decision Distribution:**")
                    decision_dist = stats.get('decision_distribution', {})
                    if decision_dist:
                        for decision, count in decision_dist.items():
                            percentage = (count / stats.get('total_examples', 1)) * 100
                            st.write(f"- {decision}: {count} ({percentage:.1f}%)")
                    else:
                        st.write("No decision data available")
                    
                    # Sentiment distribution
                    st.markdown("**Sentiment Analysis:**")
                    sentiment_dist = stats.get('sentiment_distribution', {})
                    if sentiment_dist:
                        st.write(f"- Mean sentiment: {sentiment_dist.get('mean_sentiment', 0):.3f}")
                        st.write(f"- Sentiment std: {sentiment_dist.get('std_sentiment', 0):.3f}")
                    else:
                        st.write("No sentiment data available")
                else:
                    st.info("No training data available")
        
        st.markdown('</div>', unsafe_allow_html=True)

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

def display_model_creation_interface(trainer):
    """Display model creation interface"""
    if trainer is None:
        st.error("Model trainer not available")
        return
    
    st.subheader("🛠️ Create New Model")
    
    # Get current model metadata
    current_metadata = trainer.get_model_metadata()
    
    # Show current model info
    st.info(f"**Current Model:** {current_metadata.get('name', 'Default Model')} by {current_metadata.get('author', 'Unknown')}")
    
    with st.expander("📋 Model Creation Form", expanded=True):
        # Get available models
        available_models = trainer.get_available_models()
        
        # Create form for model creation
        with st.form("model_creation_form"):
            st.markdown("#### Model Configuration")
            
            # Model type selection
            col1, col2 = st.columns(2)
            
            with col1:
                model_type = st.selectbox(
                    "Model Type:",
                    options=list(available_models.keys()),
                    format_func=lambda x: available_models[x]['name'],
                    help="Choose the neural network architecture"
                )
                
                # Show model information
                model_info = available_models[model_type]
                st.markdown(f"**Description:** {model_info['description']}")
                st.markdown(f"**Best for:** {model_info['best_for']}")
                st.markdown(f"**Training time:** {model_info['training_time']}")
                st.markdown(f"**Complexity:** {model_info['complexity']}")
            
            with col2:
                # Model metadata inputs
                model_name = st.text_input(
                    "Model Name:",
                    value=f"My {model_info['name']}",
                    help="Give your model a descriptive name"
                )
                
                author = st.text_input(
                    "Author:",
                    value=current_metadata.get('author', ''),
                    help="Your name or identifier"
                )
                
                min_epochs = st.number_input(
                    "Minimum Training Epochs:",
                    min_value=model_info['min_epochs'],
                    max_value=model_info['max_epochs'],
                    value=model_info['min_epochs'],
                    help=f"Recommended: {model_info['min_epochs']}-{model_info['max_epochs']} epochs for {model_info['name']}"
                )
            
            # Training subjects
            st.markdown("#### Training Subjects")
            st.markdown("Specify who this model will be trained on (e.g., 'John Doe', 'Trading Team A', 'Myself')")
            
            # Allow multiple training subjects
            training_subjects_input = st.text_area(
                "Training Subjects (one per line):",
                value="\n".join(current_metadata.get('training_subjects', [])),
                height=100,
                help="Enter the names of people this model will learn from, one per line"
            )
            
            # Parse training subjects
            training_subjects = [subject.strip() for subject in training_subjects_input.split('\n') if subject.strip()]
            
            # Validation
            if not model_name.strip():
                st.error("Please provide a model name")
                return
            
            if not author.strip():
                st.error("Please provide an author name")
                return
            
            if not training_subjects:
                st.error("Please specify at least one training subject")
                return
            
            # Submit button
            submitted = st.form_submit_button("🚀 Create Model", type="primary")
            
            if submitted:
                with st.spinner(f"Creating {model_name}..."):
                    try:
                        success = trainer.create_model(
                            model_type=model_type,
                            model_name=model_name,
                            author=author,
                            training_subjects=training_subjects,
                            min_epochs=min_epochs
                        )
                        
                        if success:
                            st.success(f"✅ Model '{model_name}' created successfully!")
                            st.balloons()
                            
                            # Show model details
                            st.markdown("#### 📊 Model Details")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Model Type", model_info['name'])
                                st.metric("Author", author)
                            
                            with col2:
                                st.metric("Training Subjects", len(training_subjects))
                                st.metric("Min Epochs", min_epochs)
                            
                            with col3:
                                st.metric("Complexity", model_info['complexity'])
                                st.metric("Training Time", model_info['training_time'])
                            
                            # Show training subjects
                            st.markdown("**Training Subjects:**")
                            for subject in training_subjects:
                                st.write(f"• {subject}")
                            
                            st.rerun()
                        else:
                            st.error("❌ Failed to create model")
                            
                    except Exception as e:
                        st.error(f"Error creating model: {e}")

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Status", "🛠️ Create Model", "📈 Data & Training", "🏋️ Model", "🔮 Predictions"])
    
    with tab1:
        display_model_status(trainer)
    
    with tab2:
        display_model_creation_interface(trainer)
    
    with tab3:
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
    
    with tab4:
        display_training_controls(trainer)
    
    with tab5:
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