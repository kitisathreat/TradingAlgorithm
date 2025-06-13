"""
Enhanced Model Trainer for Interactive Training Interface
Incorporates advanced technical analysis and neural network architecture from train_model.py
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import random
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import joblib
import yfinance as yf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelTrainer:
    def __init__(self):
        """Initialize the model trainer with advanced features"""
        self.model = self._build_model()
        self.scaler = StandardScaler()
        self.training_data = []
        self.training_labels = []
        
        # Load S&P 100 symbols (subset of S&P 500 for faster training)
        self.sp100_symbols = self._load_sp100_symbols()
        
        # Initialize technical analysis parameters
        self.lookback_days = 100
        self.feature_columns = [
            'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Lower', 'VIX', 'Market_Return',
            'News_Sentiment', 'Volatility'
        ]
    
    def _load_sp100_symbols(self) -> List[str]:
        """Load S&P 100 symbols from file or fetch if not available"""
        symbols_file = "sp100_symbols.json"
        if os.path.exists(symbols_file):
            with open(symbols_file, 'r') as f:
                return json.load(f)
        
        # If file doesn't exist, use a subset of S&P 500
        try:
            tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            df = tables[0]
            symbols = df['Symbol'].tolist()[:100]  # Take first 100 for faster training
            
            # Save for future use
            with open(symbols_file, 'w') as f:
                json.dump(symbols, f)
            
            return symbols
        except Exception as e:
            logging.error(f"Error loading S&P 100 symbols: {e}")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']  # Fallback to major tech stocks
    
    def _build_model(self) -> keras.Model:
        """Build the enhanced neural network model"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(1, 11)),  # 11 features
            
            # LSTM layers for temporal pattern recognition
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            
            # Dense layers for feature processing
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            
            # Output layer (3 classes: Buy, Sell, Hold)
            layers.Dense(16, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the price data"""
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_Upper'] = df['SMA_20'] + (df['Close'].rolling(window=20).std() * 2)
        df['BB_Lower'] = df['SMA_20'] - (df['Close'].rolling(window=20).std() * 2)
        
        # Volatility
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
        
        return df
    
    def _get_market_context(self, start_date: datetime, end_date: datetime) -> Tuple[pd.Series, pd.Series]:
        """Get market context data (VIX and market returns)"""
        # VIX data
        vix = yf.download('^VIX', start=start_date, end=end_date)
        vix_series = vix['Close'] / 100.0  # Convert to decimal
        
        # Market returns (S&P 500)
        spy = yf.download('^GSPC', start=start_date, end=end_date)
        market_returns = spy['Close'].pct_change()
        
        return vix_series, market_returns
    
    def get_random_stock_data(self) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray], Optional[bool]]:
        """Get random stock data with enhanced features for training"""
        # Get random stock and date
        symbol = random.choice(self.sp100_symbols)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        random_date = start_date + timedelta(days=random.randint(0, self.lookback_days - 30))
        
        try:
            # Fetch stock data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(
                start=random_date - timedelta(days=30),
                end=random_date + timedelta(days=1),
                interval='1d'
            )
            
            if len(hist) < 30:
                return None, None, None
            
            # Calculate technical indicators
            hist = self._calculate_technical_indicators(hist)
            
            # Get market context
            vix, market_returns = self._get_market_context(
                random_date - timedelta(days=30),
                random_date + timedelta(days=1)
            )
            
            # Add market context to dataframe
            hist['VIX'] = vix
            hist['Market_Return'] = market_returns
            
            # Placeholder for sentiment (to be implemented)
            hist['News_Sentiment'] = 0.0
            
            # Drop NaN values
            hist = hist.dropna()
            
            if len(hist) < 20:  # Need enough data for indicators
                return None, None, None
            
            # Get the last row for features
            last_row = hist.iloc[-2]  # Use -2 to avoid lookahead bias
            
            # Create feature dictionary for display
            features = {
                'Symbol': symbol,
                'Date': random_date.strftime('%Y-%m-%d'),
                'Current Price': last_row['Close'],
                'Previous Close': hist.iloc[-3]['Close'],
                'Volume': last_row['Volume'],
                'SMA_20': last_row['SMA_20'],
                'SMA_50': last_row['SMA_50'],
                'RSI': last_row['RSI'],
                'MACD': last_row['MACD'],
                'MACD_Signal': last_row['MACD_Signal'],
                'BB_Upper': last_row['BB_Upper'],
                'BB_Lower': last_row['BB_Lower'],
                'VIX': last_row['VIX'],
                'Market_Return': last_row['Market_Return'],
                'Volatility': last_row['Volatility'],
                'Price Change %': ((last_row['Close'] - hist.iloc[-3]['Close']) / hist.iloc[-3]['Close']) * 100,
                'Volume Change %': ((last_row['Volume'] - hist.iloc[-3]['Volume']) / hist.iloc[-3]['Volume']) * 100
            }
            
            # Create feature vector for model
            feature_vector = np.array([
                last_row['SMA_20'] / last_row['Close'] - 1,
                last_row['SMA_50'] / last_row['Close'] - 1,
                last_row['RSI'] / 100,
                last_row['MACD'],
                last_row['MACD_Signal'],
                last_row['BB_Upper'] / last_row['Close'] - 1,
                last_row['BB_Lower'] / last_row['Close'] - 1,
                last_row['VIX'],
                last_row['Market_Return'],
                last_row['News_Sentiment'],
                last_row['Volatility']
            ])
            
            # Determine if price went up (for training target)
            price_went_up = hist.iloc[-1]['Close'] > last_row['Close']
            
            return features, feature_vector, price_went_up
            
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {str(e)}")
            return None, None, None
    
    def parse_user_decision(self, text: str) -> int:
        """Parse user's text input into a trading decision"""
        text = text.lower()
        if any(word in text for word in ['buy', 'long', 'purchase', 'bullish']):
            return 0  # Buy
        elif any(word in text for word in ['sell', 'short', 'bearish']):
            return 1  # Sell
        else:
            return 2  # Hold
    
    def add_training_example(self, feature_vector: np.ndarray, decision: int) -> None:
        """Add a new training example"""
        self.training_data.append(feature_vector)
        self.training_labels.append(decision)
    
    def train_model(self) -> Dict[str, float]:
        """Train the model on collected examples"""
        if len(self.training_data) < 5:
            return {
                'error': 'Need at least 5 training examples',
                'loss': 0.0,
                'accuracy': 0.0,
                'val_loss': 0.0,
                'val_accuracy': 0.0
            }
        
        # Prepare data
        X = np.array(self.training_data)
        y = np.array(self.training_labels)
        
        # Reshape for LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        
        # Scale features
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(X.shape)
        
        # Train the model
        history = self.model.fit(
            X, y,
            epochs=50,
            batch_size=min(32, len(X)),
            validation_split=0.2,
            verbose=0
        )
        
        return {
            'loss': history.history['loss'][-1],
            'accuracy': history.history['accuracy'][-1],
            'val_loss': history.history['val_loss'][-1] if 'val_loss' in history.history else None,
            'val_accuracy': history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else None
        }
    
    def save_model(self, path: str) -> None:
        """Save the trained model and scaler"""
        # Save model
        self.model.save(path)
        
        # Save scaler
        scaler_path = os.path.splitext(path)[0] + '_scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
    
    def load_model(self, path: str) -> None:
        """Load a trained model and scaler"""
        # Load model
        self.model = tf.keras.models.load_model(path)
        
        # Load scaler
        scaler_path = os.path.splitext(path)[0] + '_scaler.joblib'
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path) 