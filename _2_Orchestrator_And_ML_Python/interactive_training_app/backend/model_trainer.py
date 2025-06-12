import tensorflow as tf
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import random

class ModelTrainer:
    def __init__(self):
        self.sp100_symbols = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'BRK-B', 'JPM', 'V', 'PG',
            'UNH', 'HD', 'MA', 'BAC', 'XOM', 'CVX', 'LLY', 'AVGO', 'PFE', 'CSCO',
            'TMO', 'ABT', 'MRK', 'PEP', 'KO', 'WMT', 'DIS', 'VZ', 'CMCSA', 'ADBE'
        ]  # Shortened list for example
        self.model = self._create_model()
        self.training_data = []
        self.training_labels = []
        
    def _create_model(self):
        """Create a simple neural network for trading decisions"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(15,)),  # 15 features
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')  # Buy, Sell, Hold
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def get_random_stock_data(self):
        """Get random stock data for training"""
        # Get random stock and date
        symbol = random.choice(self.sp100_symbols)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        random_date = start_date + timedelta(days=random.randint(0, 364))
        
        try:
            # Fetch stock data
            stock = yf.Ticker(symbol)
            hist = stock.history(
                start=random_date - timedelta(days=30),
                end=random_date + timedelta(days=1),
                interval='1d'
            )
            
            if len(hist) < 30:
                return None, None, None
            
            # Calculate features
            features = {
                'Symbol': symbol,
                'Date': random_date.strftime('%Y-%m-%d'),
                'Current Price': hist['Close'].iloc[-2],
                'Previous Close': hist['Close'].iloc[-3],
                'Volume': hist['Volume'].iloc[-2],
                'SMA_5': hist['Close'].rolling(5).mean().iloc[-2],
                'SMA_20': hist['Close'].rolling(20).mean().iloc[-2],
                'RSI': self._calculate_rsi(hist['Close'])[-2],
                'Price Change %': ((hist['Close'].iloc[-2] - hist['Close'].iloc[-3]) / hist['Close'].iloc[-3]) * 100,
                'Volume Change %': ((hist['Volume'].iloc[-2] - hist['Volume'].iloc[-3]) / hist['Volume'].iloc[-3]) * 100,
                'Next Day Change %': ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
            }
            
            # Create feature vector for model
            feature_vector = self._create_feature_vector(features)
            
            return features, feature_vector, hist['Close'].iloc[-1] > hist['Close'].iloc[-2]
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None, None, None
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI for a price series"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _create_feature_vector(self, features):
        """Create normalized feature vector for model input"""
        return np.array([
            features['Current Price'] / features['Previous Close'] - 1,
            features['Volume'] / features['Volume'],
            features['SMA_5'] / features['Current Price'] - 1,
            features['SMA_20'] / features['Current Price'] - 1,
            features['RSI'] / 100,
            features['Price Change %'] / 100,
            features['Volume Change %'] / 100,
            features['Current Price'] > features['SMA_5'],
            features['Current Price'] > features['SMA_20'],
            features['Volume'] > features['Volume'],
            features['RSI'] > 70,
            features['RSI'] < 30,
            features['Price Change %'] > 0,
            features['Volume Change %'] > 0,
            1 if features['Current Price'] > features['Previous Close'] else 0
        ])
    
    def parse_user_decision(self, text):
        """Parse user's text input into a trading decision"""
        text = text.lower()
        if any(word in text for word in ['buy', 'long', 'purchase', 'bullish']):
            return 0  # Buy
        elif any(word in text for word in ['sell', 'short', 'bearish']):
            return 1  # Sell
        else:
            return 2  # Hold
    
    def add_training_example(self, feature_vector, decision):
        """Add a new training example"""
        self.training_data.append(feature_vector)
        self.training_labels.append(decision)
    
    def train_model(self):
        """Train the model on collected examples"""
        if len(self.training_data) < 5:
            return "Need at least 5 training examples"
            
        X = np.array(self.training_data)
        y = np.array(self.training_labels)
        
        # Train the model
        history = self.model.fit(
            X, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        return {
            'loss': history.history['loss'][-1],
            'accuracy': history.history['accuracy'][-1],
            'val_loss': history.history['val_loss'][-1] if 'val_loss' in history.history else None,
            'val_accuracy': history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else None
        }
    
    def save_model(self, path):
        """Save the trained model"""
        self.model.save(path)
    
    def load_model(self, path):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(path) 