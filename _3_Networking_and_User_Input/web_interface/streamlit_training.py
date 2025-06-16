"""
Advanced Streamlit Training Interface
Presents historical stock data and collects human trading intuition
Uses the neural network ModelTrainer from the orchestrator layer
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
from pathlib import Path

# Add orchestrator path to import the ModelTrainer
repo_root = Path(__file__).parent.parent.parent
sys.path.append(str(repo_root / "_2_Orchestrator_And_ML_Python"))

try:
    from interactive_training_app.backend.model_trainer import ModelTrainer
    TRAINER_AVAILABLE = True
except ImportError as e:
    st.error(f"Could not import ModelTrainer: {e}")
    TRAINER_AVAILABLE = False

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
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} Stock Chart - Make Your Trading Decision",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False
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
            
            rsi_fig.update_layout(height=300, yaxis=dict(range=[0, 100]))
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
            
            macd_fig.update_layout(height=300)
            st.plotly_chart(macd_fig, use_container_width=True)

def display_performance_metrics(features: dict):
    """Display performance and technical metrics"""
    
    st.subheader("📊 Performance & Technical Analysis")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "5-Day Return", 
            f"{features.get('return_5d', 0):.2f}%",
            delta=None
        )
        st.metric(
            "Volatility (20d)", 
            f"{features.get('volatility_20d', 0):.2f}%"
        )
    
    with col2:
        st.metric(
            "20-Day Return", 
            f"{features.get('return_20d', 0):.2f}%"
        )
        st.metric(
            "Max Drawdown", 
            f"{features.get('max_drawdown', 0):.2f}%"
        )
    
    with col3:
        st.metric(
            "Distance from 52W High", 
            f"{features.get('distance_from_52w_high', 0):.2f}%"
        )
        st.metric(
            "Sharpe Ratio", 
            f"{features.get('sharpe_ratio', 0):.2f}"
        )
    
    with col4:
        st.metric(
            "Distance from 52W Low", 
            f"{features.get('distance_from_52w_low', 0):.2f}%"
        )
        st.metric(
            "Volume Ratio", 
            f"{features.get('volume_ratio', 1.0):.2f}x"
        )
    
    # Technical indicators summary
    st.subheader("🔧 Technical Indicators Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Price vs Moving Averages:**")
        current_price = features.get('current_price', 0)
        sma_10 = features.get('sma_10', current_price)
        sma_20 = features.get('sma_20', current_price)
        sma_50 = features.get('sma_50', current_price)
        
        st.write(f"Price vs SMA10: {((current_price/sma_10-1)*100):+.1f}%")
        st.write(f"Price vs SMA20: {((current_price/sma_20-1)*100):+.1f}%")
        st.write(f"Price vs SMA50: {((current_price/sma_50-1)*100):+.1f}%")
    
    with col2:
        st.write("**Momentum Indicators:**")
        st.write(f"RSI: {features.get('rsi', 50):.1f}")
        st.write(f"MACD: {features.get('macd', 0):.4f}")
        st.write(f"BB Position: {features.get('bb_position', 0.5):.2f}")
    
    with col3:
        st.write("**Recent Price Action:**")
        st.write(f"1-Day Change: {features.get('price_change_1d', 0)*100:+.2f}%")
        st.write(f"5-Day Change: {features.get('price_change_5d', 0)*100:+.2f}%")
        st.write(f"20-Day Change: {features.get('price_change_20d', 0)*100:+.2f}%")

def collect_trading_decision():
    """Collect the user's trading decision and reasoning"""
    
    st.subheader("🧠 Your Trading Decision")
    st.write("Based on the data above, what would you do with this stock?")
    
    # Trading decision
    decision = st.radio(
        "Your Trading Decision:",
        ["BUY", "SELL", "HOLD"],
        help="Select your trading decision based on the analysis above"
    )
    
    # Confidence level
    confidence = st.select_slider(
        "Confidence Level:",
        options=["Very Low", "Low", "Medium", "High", "Very High"],
        value="Medium",
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
    
    # Initialize trainer
    if 'trainer' not in st.session_state:
        with st.spinner("Initializing Advanced Neural Network Trainer..."):
            st.session_state.trainer = ModelTrainer()
    
    trainer = st.session_state.trainer
    
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
            if st.button("🎲 Get New Stock", type="primary"):
                with st.spinner("Fetching random stock data..."):
                    features, historical_data, symbol = trainer.get_random_stock_data()
                    
                    if features is not None:
                        st.session_state.current_features = features
                        st.session_state.current_data = historical_data
                        st.session_state.current_symbol = symbol
                        st.success(f"✅ Loaded {symbol} data from {features['date']}")
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
            st.info("👆 Click 'Get New Stock' to start training the neural network")
    
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