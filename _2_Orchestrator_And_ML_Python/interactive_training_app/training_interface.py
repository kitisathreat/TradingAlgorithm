"""
Interactive Training Interface for Market Pattern Analysis Model
This Streamlit app provides an interface for users to train the model through
interactive prompts and historical market data analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from typing import Tuple, Dict, List
import json
import os

# Local imports
import sys
sys.path.append('..')
from train_model import MarketPatternAnalyzer
from sentiment_analyzer import SentimentAnalyzer

# Constants
SP500_SYMBOLS_FILE = "sp500_symbols.json"
TRAINING_DATA_DIR = "training_data"
POSITION_CATEGORIES = [
    "Strong Buy", "Buy", "Weak Buy", "Hold", "Weak Hold",
    "Neutral", "Weak Sell", "Sell", "Strong Sell", "Extreme Sell"
]

def load_sp500_symbols() -> List[str]:
    """Load S&P 500 symbols from file or fetch if not available"""
    if os.path.exists(SP500_SYMBOLS_FILE):
        with open(SP500_SYMBOLS_FILE, 'r') as f:
            return json.load(f)
    
    # If file doesn't exist, fetch from Wikipedia
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = tables[0]
        symbols = df['Symbol'].tolist()
        
        # Save for future use
        with open(SP500_SYMBOLS_FILE, 'w') as f:
            json.dump(symbols, f)
        
        return symbols
    except Exception as e:
        st.error(f"Error loading S&P 500 symbols: {e}")
        return []

def get_random_training_point() -> Tuple[str, datetime]:
    """Get a random stock and date for training"""
    symbols = load_sp500_symbols()
    if not symbols:
        st.error("No S&P 500 symbols available")
        return None, None
    
    symbol = random.choice(symbols)
    end_date = datetime.now() - timedelta(days=30)  # Don't use very recent data
    start_date = end_date - timedelta(days=25*365)  # 25 years ago
    random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    
    return symbol, random_date

def fetch_historical_data(symbol: str, date: datetime, lookback_days: int = 100) -> pd.DataFrame:
    """Fetch historical data for the given symbol and date"""
    end_date = date + timedelta(days=1)
    start_date = date - timedelta(days=lookback_days)
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            st.error(f"No data available for {symbol} on {date}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def plot_historical_data(df: pd.DataFrame, target_date: datetime) -> None:
    """Create an interactive plot of historical data"""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # Add volume bars
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        yaxis='y2',
        opacity=0.3
    ))
    
    # Add vertical line for target date
    fig.add_vline(
        x=target_date,
        line_dash="dash",
        line_color="red",
        annotation_text="Target Date",
        annotation_position="top right"
    )
    
    # Update layout
    fig.update_layout(
        title='Historical Price Data',
        yaxis_title='Price',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right'
        ),
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def save_training_data(symbol: str, date: datetime, user_response: str, 
                      categorized_position: str) -> None:
    """Save the training data to a file"""
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    
    data = {
        'symbol': symbol,
        'date': date.isoformat(),
        'user_response': user_response,
        'categorized_position': categorized_position,
        'timestamp': datetime.now().isoformat()
    }
    
    filename = f"{TRAINING_DATA_DIR}/training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    st.set_page_config(
        page_title="Trading Model Training Interface",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("Trading Model Training Interface")
    st.markdown("""
    This interface helps train our trading model by collecting your trading decisions
    based on historical market data. For each prompt, you'll see:
    1. A random stock from the S&P 500
    2. A random date from the last 25 years
    3. Historical price data up to that date
    
    Your task is to provide your trading decision based on the data shown.
    """)
    
    # Initialize session state
    if 'current_symbol' not in st.session_state:
        st.session_state.current_symbol = None
    if 'current_date' not in st.session_state:
        st.session_state.current_date = None
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Training Controls")
        if st.button("New Training Prompt"):
            st.session_state.current_symbol, st.session_state.current_date = get_random_training_point()
        
        st.markdown("---")
        st.markdown("### Training Progress")
        training_files = [f for f in os.listdir(TRAINING_DATA_DIR) if f.endswith('.json')] if os.path.exists(TRAINING_DATA_DIR) else []
        st.metric("Training Samples Collected", len(training_files))
    
    # Main content area
    if st.session_state.current_symbol and st.session_state.current_date:
        st.header(f"Training Prompt: {st.session_state.current_symbol}")
        st.subheader(f"Date: {st.session_state.current_date.strftime('%Y-%m-%d')}")
        
        # Fetch and display historical data
        df = fetch_historical_data(
            st.session_state.current_symbol,
            st.session_state.current_date
        )
        
        if df is not None:
            plot_historical_data(df, st.session_state.current_date)
            
            # User input section
            st.markdown("---")
            st.subheader("Your Trading Decision")
            
            user_response = st.text_area(
                "Based on the data shown, what would your trading decision be? "
                "Explain your reasoning:",
                height=150
            )
            
            if user_response:
                # Categorize the response
                st.subheader("Categorize Your Decision")
                categorized_position = st.selectbox(
                    "Select the category that best matches your decision:",
                    POSITION_CATEGORIES
                )
                
                if st.button("Submit Training Data"):
                    save_training_data(
                        st.session_state.current_symbol,
                        st.session_state.current_date,
                        user_response,
                        categorized_position
                    )
                    st.success("Training data saved successfully!")
                    st.session_state.current_symbol = None
                    st.session_state.current_date = None
                    st.experimental_rerun()
    else:
        st.info("Click 'New Training Prompt' in the sidebar to start training.")

if __name__ == "__main__":
    main() 