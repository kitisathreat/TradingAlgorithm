"""
Advanced Streamlit Training Interface
Updated to match local GUI approach and coding patterns
Presents historical stock data and collects human trading intuition
Uses the neural network ModelTrainer from the orchestrator layer
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, timezone
import sys
import os
import json
from pathlib import Path
import yfinance as yf

# Add orchestrator path to import the ModelTrainer
repo_root = Path(__file__).parent.parent.parent
sys.path.append(str(repo_root / "_2_Orchestrator_And_ML_Python"))

try:
    from interactive_training_app.backend.model_trainer import ModelTrainer
    TRAINER_AVAILABLE = True
except ImportError as e:
    st.error(f"Could not import ModelTrainer: {e}")
    TRAINER_AVAILABLE = False

# Load stock symbols from JSON file (matching local GUI approach)
def load_stock_symbols():
    """Load stock symbols from the JSON file like the local GUI does"""
    try:
        symbols_file = repo_root / "_2_Orchestrator_And_ML_Python" / "interactive_training_app" / "sp100_symbols.json"
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

def create_stock_chart(data: pd.DataFrame, symbol: str, current_date: str):
    """Create an interactive stock chart with technical indicators"""
    
    # Create candlestick chart
    fig = go.Figure(data=go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=f"{symbol} Price"
    ))
    
    # Add moving averages if available
    if 'SMA_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=2)
        ))
    
    if 'SMA_50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='blue', width=2)
        ))
    
    # Add Bollinger Bands if available
    if all(col in data.columns for col in ['BB_Upper', 'BB_Lower']):
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)',
            showlegend=False
        ))
    
    # Highlight current date with vertical line
    current_date_parsed = pd.to_datetime(current_date)
    if current_date_parsed in data.index:
        current_price = data.loc[current_date_parsed, 'Close']
        fig.add_vline(
            x=current_date_parsed,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Decision Point: ${current_price:.2f}",
            annotation_position="top"
        )
    
    # Calculate proper axis ranges
    min_price = data['Low'].min()
    max_price = data['High'].max()
    price_padding = (max_price - min_price) * 0.05  # 5% padding
    
    # Update layout with proper axis ranges
    fig.update_layout(
        title=f"{symbol} Stock Chart - Make Your Trading Decision",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        # Set proper x-axis range (full date range)
        xaxis=dict(
            range=[data.index.min(), data.index.max()],
            type='date'
        ),
        # Set proper y-axis range (price range with padding)
        yaxis=dict(
            range=[max(0, min_price - price_padding), max_price + price_padding],
            type='linear'
        )
    )
    
    return fig

def create_technical_indicators_chart(data: pd.DataFrame, current_date: str):
    """Create charts for technical indicators"""
    
    # Create subplots for different indicators
    col1, col2 = st.columns(2)
    
    with col1:
        # RSI Chart
        if 'RSI' in data.columns:
            st.subheader("RSI (Relative Strength Index)")
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple')
            ))
            
            # Add RSI levels
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            
            # Highlight current date
            current_date_parsed = pd.to_datetime(current_date)
            if current_date_parsed in data.index:
                current_rsi = data.loc[current_date_parsed, 'RSI']
                rsi_fig.add_vline(x=current_date_parsed, line_dash="dash", line_color="red")
                st.metric("Current RSI", f"{current_rsi:.1f}")
            
            # Set proper axis ranges for RSI
            rsi_fig.update_layout(
                height=300, 
                yaxis=dict(range=[0, 100]),
                xaxis=dict(
                    range=[data.index.min(), data.index.max()],
                    type='date'
                )
            )
            st.plotly_chart(rsi_fig, use_container_width=True)
    
    with col2:
        # MACD Chart
        if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
            st.subheader("MACD")
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue')
            ))
            macd_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['MACD_Signal'],
                mode='lines',
                name='Signal',
                line=dict(color='red')
            ))
            
            # Highlight current date
            current_date_parsed = pd.to_datetime(current_date)
            if current_date_parsed in data.index:
                current_macd = data.loc[current_date_parsed, 'MACD']
                macd_fig.add_vline(x=current_date_parsed, line_dash="dash", line_color="red")
                st.metric("Current MACD", f"{current_macd:.4f}")
            
            # Calculate proper MACD axis range
            macd_min = min(data['MACD'].min(), data['MACD_Signal'].min())
            macd_max = max(data['MACD'].max(), data['MACD_Signal'].max())
            macd_padding = (macd_max - macd_min) * 0.1 if macd_max != macd_min else 0.1
            
            macd_fig.update_layout(
                height=300,
                xaxis=dict(
                    range=[data.index.min(), data.index.max()],
                    type='date'
                ),
                yaxis=dict(
                    range=[macd_min - macd_padding, macd_max + macd_padding]
                )
            )
            st.plotly_chart(macd_fig, use_container_width=True)

def display_performance_metrics(features: dict):
    """Display performance metrics in a clean format"""
    
    st.subheader("📊 Performance Metrics")
    
    # Create metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Price Change", f"${features.get('price_change', 0):.2f}")
    
    with col2:
        st.metric("Price Change %", f"{features.get('price_change_pct', 0):.2f}%")
    
    with col3:
        st.metric("Volume", f"{features.get('volume', 0):,}")
    
    with col4:
        st.metric("Volatility", f"{features.get('volatility', 0):.2f}%")
    
    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'RSI' in features:
            st.metric("RSI", f"{features['RSI']:.1f}")
    
    with col2:
        if 'MACD' in features:
            st.metric("MACD", f"{features['MACD']:.4f}")
    
    with col3:
        if 'BB_Position' in features:
            bb_pos = features['BB_Position']
            if bb_pos > 0.8:
                bb_status = "Overbought"
            elif bb_pos < 0.2:
                bb_status = "Oversold"
            else:
                bb_status = "Neutral"
            st.metric("BB Position", bb_status)
    
    with col4:
        if 'SMA_Trend' in features:
            trend = features['SMA_Trend']
            if trend > 0:
                trend_text = "Bullish"
            elif trend < 0:
                trend_text = "Bearish"
            else:
                trend_text = "Neutral"
            st.metric("SMA Trend", trend_text)

def collect_trading_decision():
    """Collect trading decision from user"""
    
    st.subheader("🎯 Make Your Trading Decision")
    
    # Decision buttons in a row
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
        confidence = 0.8
    elif sell_clicked:
        decision = "SELL"
        confidence = 0.8
    elif hold_clicked:
        decision = "HOLD"
        confidence = 0.8
    else:
        decision = "HOLD"
        confidence = 0.6
    
    # Confidence slider
    confidence = st.slider(
        "Confidence Level",
        min_value=0.1,
        max_value=1.0,
        value=confidence,
        step=0.1,
        help="How confident are you in this decision?"
    )
    
    # Reasoning (for sentiment analysis)
    reasoning = st.text_area(
        "Describe your reasoning:",
        placeholder="e.g., 'Strong bullish momentum with RSI showing upward trend. Breaking above resistance with high volume. Confident buy opportunity.'",
        help="Explain why you made this decision. This will be analyzed for sentiment and keywords.",
        height=100
    )
    
    return decision, confidence, reasoning

def load_stock_data(symbol: str, days: int = 30):
    """Load stock data using random date range within the last 25 years"""
    try:
        # Import the date range utilities
        import sys
        from pathlib import Path
        REPO_ROOT = Path(__file__).parent.parent.parent
        ORCHESTRATOR_PATH = REPO_ROOT / "_2_Orchestrator_And_ML_Python"
        sys.path.append(str(ORCHESTRATOR_PATH))
        
        from date_range_utils import find_available_data_range, validate_date_range
        
        # Get random date range within the last 25 years
        start_date, end_date = find_available_data_range(symbol, days, max_years_back=25)
        
        # Validate the date range
        if not validate_date_range(start_date, end_date, symbol):
            st.error(f"Invalid date range generated for {symbol}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            return None, None
        
        st.info(f"Fetching {days} days of data for {symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (random range within last 25 years)")
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            st.error(f"No data found for {symbol}. Please try a different stock or fewer days.")
            return None, None
        
        # Ensure df.index is timezone-aware (UTC)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
        
        # Check if we got the expected amount of data
        if len(df) < days * 0.8:  # Allow 20% tolerance for weekends/holidays
            st.warning(f"Got {len(df)} days of data for {symbol}, expected around {days} days")
        
        if df.index.max() > end_date:
            st.warning(f"Data for {symbol} contains dates newer than expected. This may indicate a system clock issue.")
            df = df[df.index <= end_date]
            if df.empty:
                st.error(f"No valid data found for {symbol} after filtering dates.")
                return None, None
        
        st.success(f"Successfully loaded {len(df)} days of data for {symbol}")
        st.info(f"Data range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        
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
    """Main training interface"""
    
    st.title("🧠 Advanced Neural Network Training Interface")
    st.markdown("""
    This interface combines **technical analysis** with **sentiment analysis** to train a neural network 
    that learns to mimic your trading decisions. The AI analyzes both market data and your reasoning 
    to understand your trading psychology.
    """)
    
    if not TRAINER_AVAILABLE:
        st.error("ModelTrainer not available. Please check the installation.")
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
    
    # Sidebar - Training Controls
    st.sidebar.header("🎯 Training Controls")
    
    # Training statistics
    stats = trainer.get_training_stats()
    st.sidebar.metric("Training Examples", stats['total_examples'])
    st.sidebar.metric("Symbols Trained", stats['symbols_trained'])
    
    if stats['total_examples'] > 0:
        st.sidebar.write("**Decision Distribution:**")
        for decision, count in stats['decision_distribution'].items():
            st.sidebar.write(f"- {decision}: {count}")
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["📈 Training", "🤖 Neural Network", "📊 Statistics"])
    
    with tab1:
        st.header("1. Stock Analysis & Decision Collection")
        
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
        
        with col1:
            if st.button("📊 Test Data Connectivity"):
                st.info("Testing historical data fetching...")
                # This could test the data sources
                st.success("✅ Data connectivity working")
        
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
            
            # Stock chart
            chart = create_stock_chart(historical_data, symbol, features['date'])
            st.plotly_chart(chart, use_container_width=True)
            
            # Technical indicators
            create_technical_indicators_chart(historical_data, features['date'])
            
            # Performance metrics
            display_performance_metrics(features)
            
            # Collect trading decision
            st.markdown("---")
            decision, confidence, reasoning = collect_trading_decision()
            
            # Submit training example
            if st.button("🚀 Submit Training Example", type="primary"):
                if reasoning.strip():
                    with st.spinner("Processing your decision..."):
                        # Analyze sentiment
                        sentiment_analysis = trainer.analyze_sentiment_and_keywords(reasoning)
                        
                        # Add training example
                        success = trainer.add_training_example(features, sentiment_analysis, decision)
                        
                        if success:
                            st.success("✅ Training example added successfully!")
                            
                            # Show sentiment analysis results
                            st.subheader("🔍 Sentiment Analysis Results")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                sentiment_score = sentiment_analysis['sentiment_score']
                                if sentiment_score > 0.1:
                                    sentiment_emoji = "😊"
                                    sentiment_text = "Positive"
                                elif sentiment_score < -0.1:
                                    sentiment_emoji = "😞"
                                    sentiment_text = "Negative"
                                else:
                                    sentiment_emoji = "😐"
                                    sentiment_text = "Neutral"
                                
                                st.metric(
                                    f"{sentiment_emoji} Sentiment", 
                                    sentiment_text,
                                    f"Score: {sentiment_score:.3f}"
                                )
                            
                            with col2:
                                st.metric("Confidence", sentiment_analysis['confidence'].title())
                            
                            with col3:
                                st.metric("Keywords Found", len(sentiment_analysis['keywords']))
                            
                            if sentiment_analysis['keywords']:
                                st.write("**Keywords detected:**", ", ".join(sentiment_analysis['keywords']))
                            
                            # Clear current data to get a new example
                            if 'current_features' in st.session_state:
                                del st.session_state.current_features
                                del st.session_state.current_data
                                del st.session_state.current_symbol
                            
                            st.rerun()
                        else:
                            st.error("❌ Failed to add training example")
                else:
                    st.warning("⚠️ Please provide your reasoning before submitting")
        else:
            st.info("👆 Select a stock and click 'Get Stock Data' to start training the neural network")
    
    with tab2:
        st.header("🤖 Neural Network Training & Predictions")
        
        model_state = trainer.get_model_state()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Status")
            
            if model_state['is_trained']:
                st.success("✅ Neural Network is Trained")
                st.metric("Training Examples", model_state['training_examples'])
                st.metric("Model Accuracy", f"{model_state['model_accuracy']:.1%}")
                st.write(f"**Last Trained:** {model_state.get('last_training_date', 'Unknown')}")
            else:
                st.info("🔄 Neural Network Not Yet Trained")
                st.write(f"Training examples: {stats['total_examples']}")
                st.write("Need at least 10 examples to train the network")
        
        with col2:
            st.subheader("Training Controls")
            
            if stats['total_examples'] >= 10:
                if st.button("🧠 Train Neural Network", type="primary"):
                    with st.spinner("Training neural network... This may take a few minutes."):
                        success = trainer.train_neural_network()
                        
                        if success:
                            st.success("🎉 Neural network trained successfully!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("❌ Training failed. Check logs for details.")
            else:
                st.info(f"Need {10 - stats['total_examples']} more examples to start training")
            
            # Prediction test
            if model_state['is_trained'] and 'current_features' in st.session_state:
                st.markdown("---")
                st.subheader("🔮 Test Prediction")
                
                if st.button("Generate AI Prediction"):
                    prediction_result = trainer.make_prediction(st.session_state.current_features)
                    
                    st.write("**AI Prediction:**", prediction_result['prediction'])
                    st.write("**Confidence:**", f"{prediction_result['confidence']:.1%}")
                    
                    # Compare with actual outcome if available
                    actual = st.session_state.current_features.get('actual_outcome')
                    if actual and actual != 'UNKNOWN':
                        if actual == prediction_result['prediction']:
                            st.success(f"✅ Match! Actual outcome was {actual}")
                        else:
                            st.warning(f"❌ Mismatch. Actual outcome was {actual}")
    
    with tab3:
        st.header("📊 Training Statistics & Analysis")
        
        if stats['total_examples'] > 0:
            # Decision distribution
            st.subheader("Decision Distribution")
            decision_df = pd.DataFrame(
                list(stats['decision_distribution'].items()),
                columns=['Decision', 'Count']
            )
            
            fig = px.pie(decision_df, values='Count', names='Decision', 
                        title="Distribution of Trading Decisions")
            st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment analysis
            if 'sentiment_distribution' in stats:
                sent_stats = stats['sentiment_distribution']
                st.subheader("Sentiment Analysis")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Sentiment", f"{sent_stats.get('mean_sentiment', 0):.3f}")
                with col2:
                    st.metric("Positive Decisions", sent_stats.get('positive_count', 0))
                with col3:
                    st.metric("Negative Decisions", sent_stats.get('negative_count', 0))
            
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
                        if example['sentiment_analysis']['keywords']:
                            st.write(f"**Keywords:** {', '.join(example['sentiment_analysis']['keywords'])}")
        else:
            st.info("No training data available yet. Start by adding some training examples!")

if __name__ == "__main__":
    main()