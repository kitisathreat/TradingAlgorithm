"""
Streamlit Interface for Model Training
This module provides a user-friendly interface for training the trading model
using the ModelTrainer from the orchestrator module.
"""

import streamlit as st
import sys
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Add the orchestrator directory to Python path
REPO_ROOT = Path(__file__).parent.parent.parent
ORCHESTRATOR_PATH = REPO_ROOT / "_2_Orchestrator_And_ML_Python"
sys.path.append(str(ORCHESTRATOR_PATH))

try:
    from interactive_training_app.backend.model_trainer import ModelTrainer
except ImportError as e:
    st.error(f"❌ Failed to import ModelTrainer: {str(e)}")
    ModelTrainer = None

# Initialize session state
if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = ModelTrainer() if ModelTrainer else None
if 'current_features' not in st.session_state:
    st.session_state.current_features = None
if 'current_feature_vector' not in st.session_state:
    st.session_state.current_feature_vector = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'training_count' not in st.session_state:
    st.session_state.training_count = 0

def plot_technical_indicators(features):
    """Create an interactive plot of technical indicators"""
    fig = go.Figure()
    
    # Add price and moving averages
    fig.add_trace(go.Scatter(
        y=[features['Current Price']],
        name='Current Price',
        mode='lines+markers',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        y=[features['SMA_20']],
        name='SMA(20)',
        mode='lines+markers',
        line=dict(color='orange', width=1, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        y=[features['SMA_50']],
        name='SMA(50)',
        mode='lines+markers',
        line=dict(color='green', width=1, dash='dash')
    ))
    
    # Add Bollinger Bands
    fig.add_trace(go.Scatter(
        y=[features['BB_Upper']],
        name='BB Upper',
        mode='lines+markers',
        line=dict(color='gray', width=1, dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        y=[features['BB_Lower']],
        name='BB Lower',
        mode='lines+markers',
        line=dict(color='gray', width=1, dash='dot'),
        fill='tonexty'
    ))
    
    # Update layout
    fig.update_layout(
        title='Technical Indicators',
        yaxis_title='Price',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def main():
    st.title("Trading Model Training Interface")
    st.markdown("""
    This interface helps train our trading model by collecting your trading decisions
    based on historical market data. For each prompt, you'll see:
    1. A random stock from the S&P 100
    2. Technical indicators and market context
    3. Your task is to provide your trading decision based on the data shown
    """)
    
    # Sidebar for controls and progress
    with st.sidebar:
        st.header("Training Controls")
        
        if st.button("Get New Training Example"):
            if st.session_state.model_trainer:
                features, feature_vector, actual_result = st.session_state.model_trainer.get_random_stock_data()
                if features:
                    st.session_state.current_features = features
                    st.session_state.current_feature_vector = feature_vector
                    st.session_state.current_actual_result = actual_result
                else:
                    st.error("Failed to fetch stock data. Please try again.")
            else:
                st.error("Model trainer not initialized")
        
        st.markdown("---")
        st.markdown("### Training Progress")
        st.metric("Training Examples", st.session_state.training_count)
        
        if st.session_state.training_count >= 5:
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    results = st.session_state.model_trainer.train_model()
                    if 'error' in results:
                        st.error(results['error'])
                    else:
                        st.success("Model trained successfully!")
                        st.write("Training Results:")
                        st.write(f"- Loss: {results['loss']:.4f}")
                        st.write(f"- Accuracy: {results['accuracy']:.2%}")
                        if results['val_accuracy']:
                            st.write(f"- Validation Accuracy: {results['val_accuracy']:.2%}")
            
            if st.button("Save Model"):
                model_path = os.path.join(ORCHESTRATOR_PATH, "models", "trading_model.h5")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                st.session_state.model_trainer.save_model(model_path)
                st.success(f"✅ Model saved to {model_path}")
        else:
            st.info(f"⚠️ Need at least 5 examples to train (have {st.session_state.training_count})")
    
    # Main content area
    if st.session_state.current_features:
        # Display stock information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Symbol", st.session_state.current_features['Symbol'])
            st.metric("Date", st.session_state.current_features['Date'])
            st.metric("Current Price", f"${st.session_state.current_features['Current Price']:.2f}")
        
        with col2:
            st.metric("RSI", f"{st.session_state.current_features['RSI']:.2f}")
            st.metric("MACD", f"{st.session_state.current_features['MACD']:.4f}")
            st.metric("MACD Signal", f"{st.session_state.current_features['MACD_Signal']:.4f}")
        
        with col3:
            st.metric("VIX", f"{st.session_state.current_features['VIX']:.2%}")
            st.metric("Market Return", f"{st.session_state.current_features['Market_Return']:.2%}")
            st.metric("Volatility", f"{st.session_state.current_features['Volatility']:.2%}")
        
        # Technical indicators plot
        st.plotly_chart(plot_technical_indicators(st.session_state.current_features), use_container_width=True)
        
        # User input section
        st.markdown("---")
        st.subheader("Your Trading Decision")
        
        decision = st.radio(
            "What would you do?",
            ["Buy", "Sell", "Hold"],
            horizontal=True
        )
        
        reasoning = st.text_area(
            "Explain your reasoning:",
            height=100,
            help="Example: 'Buy because price is above both SMAs and RSI shows oversold conditions'"
        )
        
        if st.button("Submit Decision"):
            if reasoning:
                # Convert decision to model format (0=Buy, 1=Sell, 2=Hold)
                decision_map = {"Buy": 0, "Sell": 1, "Hold": 2}
                model_decision = decision_map[decision]
                
                # Add training example
                st.session_state.model_trainer.add_training_example(
                    st.session_state.current_feature_vector,
                    model_decision
                )
                
                # Update training history
                st.session_state.training_history.append({
                    'Symbol': st.session_state.current_features['Symbol'],
                    'Date': st.session_state.current_features['Date'],
                    'Decision': decision,
                    'Actual': 'Up' if st.session_state.current_actual_result else 'Down',
                    'Reasoning': reasoning
                })
                
                st.session_state.training_count += 1
                st.success(f"✅ Training example #{st.session_state.training_count} added!")
                
                # Clear current example
                st.session_state.current_features = None
                st.session_state.current_feature_vector = None
                st.experimental_rerun()
            else:
                st.warning("⚠️ Please provide your reasoning")
    
    # Display training history
    if st.session_state.training_history:
        st.markdown("---")
        st.subheader("Training History")
        history_df = pd.DataFrame(st.session_state.training_history)
        st.dataframe(history_df, use_container_width=True)

if __name__ == "__main__":
    main() 