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
import requests
import time

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
        self.lookback_days = 100  # Reduced from 365 to 100 for better compatibility
        self.min_required_days = 60  # Minimum days required for analysis
        self.feature_columns = [
            'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Lower', 'VIX', 'Market_Return',
            'News_Sentiment', 'Volatility'
        ]
        
        # Initialize fallback data cache
        self.data_cache = {}
        self.cache_timeout = 3600  # 1 hour cache timeout
        
        # Initialize synthetic data generator
        self.synthetic_data_enabled = True
        self.synthetic_data_quality = 0.8  # Quality factor for synthetic data
    
    def _load_sp100_symbols(self) -> List[str]:
        """Load S&P 100 symbols from file or fetch if not available"""
        # Hardcoded list of major stocks as fallback
        FALLBACK_SYMBOLS = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK-B', 'JPM', 'JNJ',
            'V', 'WMT', 'PG', 'MA', 'HD', 'BAC', 'CVX', 'KO', 'PFE', 'MRK',
            'ABBV', 'PEP', 'TMO', 'COST', 'MCD', 'CSCO', 'VZ', 'ADBE', 'CRM', 'NFLX',
            'DIS', 'INTC', 'NKE', 'PYPL', 'QCOM', 'T', 'UNH', 'UPS', 'WFC', 'XOM'
        ]
        
        try:
            symbols_file = "sp100_symbols.json"
            if os.path.exists(symbols_file):
                with open(symbols_file, 'r') as f:
                    symbols = json.load(f)
                    if len(symbols) >= 20:  # Ensure we have enough symbols
                        return symbols
            
            # If file doesn't exist or has insufficient symbols, try fetching
            try:
                tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
                df = tables[0]
                symbols = df['Symbol'].tolist()[:100]  # Take first 100 for faster training
                
                # Save for future use if possible
                try:
                    with open(symbols_file, 'w') as f:
                        json.dump(symbols, f)
                except Exception as e:
                    logging.warning(f"Could not save symbols file: {e}")
                
                return symbols
            except Exception as e:
                logging.warning(f"Error fetching S&P symbols: {e}")
                return FALLBACK_SYMBOLS
                
        except Exception as e:
            logging.error(f"Error loading S&P 100 symbols: {e}")
            return FALLBACK_SYMBOLS
    
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
    
    def _fetch_with_fallback(self, symbol: str, start_date: datetime, end_date: datetime, max_retries: int = 5) -> pd.DataFrame:
        """Fetch stock data with multiple fallback methods"""
        
        # Method 1: Try yfinance with retry logic
        for attempt in range(max_retries):
            try:
                logging.info(f"Attempt {attempt + 1}: Fetching {symbol} data with yfinance")
                
                # More conservative delay for Streamlit Cloud
                time.sleep(random.uniform(1.5, 3.0))  # Increased delay
                
                # Check cache first
                cache_key = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                cached_data = get_cached_data(cache_key)
                if cached_data is not None:
                    logging.info(f"Using cached data for {symbol}")
                    return pd.DataFrame(cached_data)
                
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date, interval='1d')
                
                if not hist.empty and len(hist) >= self.min_required_days:
                    logging.info(f"Successfully fetched {symbol} data with yfinance")
                    # Cache the successful result
                    with cache_lock:
                        cache[cache_key] = {
                            'data': hist.to_dict(),
                            'timestamp': time.time()
                        }
                    return hist
                elif not hist.empty:
                    logging.warning(f"Insufficient data for {symbol}: {len(hist)} days")
                    if self.synthetic_data_enabled:
                        logging.info(f"Generating synthetic data for {symbol}")
                        return self._generate_synthetic_data(symbol, hist)
                    
            except Exception as e:
                logging.warning(f"yfinance attempt {attempt + 1} failed for {symbol}: {str(e)}")
                if attempt < max_retries - 1:
                    # More conservative exponential backoff
                    time.sleep(3 * (2 ** attempt))  # Increased base delay
        
        # Method 2: Generate synthetic data if enabled
        if self.synthetic_data_enabled:
            logging.warning(f"All yfinance attempts failed for {symbol}, using synthetic data")
            return self._generate_synthetic_data(symbol)
        
        logging.error(f"Failed to fetch or generate data for {symbol}")
        return pd.DataFrame()

    def _generate_synthetic_data(self, symbol: str, real_data: pd.DataFrame = None) -> pd.DataFrame:
        """Generate synthetic data based on real data or market averages"""
        try:
            if real_data is not None and not real_data.empty:
                # Use real data as base and extend it
                base_price = real_data['Close'].iloc[-1]
                base_volume = real_data['Volume'].iloc[-1]
                volatility = real_data['Close'].pct_change().std()
                days_missing = self.lookback_days - len(real_data)
            else:
                # Use market averages
                base_price = 100.0  # Default base price
                base_volume = 1000000  # Default base volume
                volatility = 0.02  # Default volatility
                days_missing = self.lookback_days

            # Generate synthetic data
            dates = pd.date_range(end=datetime.now(), periods=days_missing, freq='B')
            synthetic_data = pd.DataFrame(index=dates)
            
            # Generate price series with realistic patterns
            returns = np.random.normal(0, volatility * self.synthetic_data_quality, days_missing)
            prices = base_price * (1 + returns).cumprod()
            synthetic_data['Close'] = prices
            synthetic_data['Open'] = prices * (1 + np.random.normal(0, 0.002, days_missing))
            synthetic_data['High'] = synthetic_data[['Open', 'Close']].max(axis=1) * (1 + abs(np.random.normal(0, 0.003, days_missing)))
            synthetic_data['Low'] = synthetic_data[['Open', 'Close']].min(axis=1) * (1 - abs(np.random.normal(0, 0.003, days_missing)))
            synthetic_data['Volume'] = base_volume * (1 + np.random.normal(0, 0.5, days_missing))
            
            # Combine real and synthetic data if available
            if real_data is not None and not real_data.empty:
                final_data = pd.concat([real_data, synthetic_data])
            else:
                final_data = synthetic_data
            
            # Add technical indicators
            final_data['SMA_20'] = final_data['Close'].rolling(window=20).mean()
            final_data['SMA_50'] = final_data['Close'].rolling(window=50).mean()
            final_data['RSI'] = self._calculate_rsi(final_data['Close'])
            final_data['MACD'], final_data['MACD_Signal'] = self._calculate_macd(final_data['Close'])
            final_data['BB_Upper'], final_data['BB_Lower'] = self._calculate_bollinger_bands(final_data['Close'])
            
            logging.info(f"Generated {len(final_data)} days of synthetic data for {symbol}")
            return final_data
            
        except Exception as e:
            logging.error(f"Error generating synthetic data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI technical indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD technical indicator"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band, lower_band
    
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
        """Get market context data (VIX and market returns) with fallback"""
        try:
            # Try to get real VIX data
            vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
            if not vix.empty:
                vix_series = vix['Close'] / 100.0
            else:
                raise ValueError("Empty VIX data")
                
            # Try to get real market data
            spy = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
            if not spy.empty:
                market_returns = spy['Close'].pct_change()
            else:
                raise ValueError("Empty market data")
                
            return vix_series, market_returns
            
        except Exception as e:
            logging.warning(f"Failed to get market context: {e}, using fallback")
            
            # Generate fallback market context
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Simulate VIX (typically 12-40, average around 20)
            vix_values = np.random.normal(0.20, 0.05, len(dates))  # 20% +/- 5%
            vix_values = np.clip(vix_values, 0.10, 0.50)  # Reasonable bounds
            vix_series = pd.Series(vix_values, index=dates)
            
            # Simulate market returns (small daily changes)
            market_returns = pd.Series(np.random.normal(0.001, 0.015, len(dates)), index=dates)
            
            return vix_series, market_returns
    
    def get_random_stock_data(self) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray], Optional[bool]]:
        """Get random stock data with enhanced features for training"""
        max_attempts = 5
        
        for attempt in range(max_attempts):
            try:
                # Get random stock and date
                symbol = random.choice(self.sp100_symbols)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self.lookback_days)
                random_date = start_date + timedelta(days=random.randint(0, self.lookback_days - 30))
                
                logging.info(f"[Streamlit Cloud] Fetching data for {symbol} (attempt {attempt + 1}/{max_attempts})")
                logging.info(f"[Streamlit Cloud] Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                
                # Fetch stock data with fallback
                hist = self._fetch_with_fallback(
                    symbol,
                    random_date - timedelta(days=60),  # Get more data for indicators
                    random_date + timedelta(days=1)
                )
                
                if len(hist) < 50:  # Need enough data for indicators
                    logging.warning(f"[Streamlit Cloud] Insufficient data for {symbol}: {len(hist)} days")
                    continue
                
                logging.info(f"[Streamlit Cloud] Successfully fetched {len(hist)} days of data for {symbol}")
                
                # Calculate technical indicators
                hist = self._calculate_technical_indicators(hist)
                logging.info(f"[Streamlit Cloud] Calculated technical indicators for {symbol}")
                
                # Get market context
                vix, market_returns = self._get_market_context(
                    random_date - timedelta(days=60),
                    random_date + timedelta(days=1)
                )
                logging.info(f"[Streamlit Cloud] Retrieved market context for {symbol}")
                
                # Align market context with stock data
                hist = hist.reindex(hist.index.intersection(vix.index))
                vix = vix.reindex(hist.index)
                market_returns = market_returns.reindex(hist.index)
                
                # Add market context to dataframe
                hist['VIX'] = vix.values
                hist['Market_Return'] = market_returns.values
                
                # Placeholder for sentiment (to be implemented)
                hist['News_Sentiment'] = 0.0
                
                # Drop NaN values
                hist = hist.dropna()
                
                if len(hist) < 20:  # Need enough clean data
                    logging.warning(f"[Streamlit Cloud] Insufficient clean data for {symbol}: {len(hist)} days after processing")
                    continue
                
                logging.info(f"[Streamlit Cloud] Successfully processed data for {symbol} with {len(hist)} clean days")
                
                # Get the last row for features
                last_row = hist.iloc[-2]  # Use -2 to avoid lookahead bias
                
                # Create feature dictionary for display
                features = {
                    'Symbol': symbol,
                    'Date': random_date.strftime('%Y-%m-%d'),
                    'Current Price': float(last_row['Close']),
                    'Previous Close': float(hist.iloc[-3]['Close']),
                    'Volume': int(last_row['Volume']),
                    'SMA_20': float(last_row['SMA_20']),
                    'SMA_50': float(last_row['SMA_50']),
                    'RSI': float(last_row['RSI']),
                    'MACD': float(last_row['MACD']),
                    'MACD_Signal': float(last_row['MACD_Signal']),
                    'BB_Upper': float(last_row['BB_Upper']),
                    'BB_Lower': float(last_row['BB_Lower']),
                    'VIX': float(last_row['VIX']),
                    'Market_Return': float(last_row['Market_Return']),
                    'Volatility': float(last_row['Volatility']),
                    'Price Change %': float(((last_row['Close'] - hist.iloc[-3]['Close']) / hist.iloc[-3]['Close']) * 100),
                    'Volume Change %': float(((last_row['Volume'] - hist.iloc[-3]['Volume']) / hist.iloc[-3]['Volume']) * 100)
                }
                
                # Create feature vector for model
                feature_vector = np.array([
                    float(last_row['SMA_20'] / last_row['Close'] - 1),
                    float(last_row['SMA_50'] / last_row['Close'] - 1),
                    float(last_row['RSI'] / 100),
                    float(last_row['MACD']),
                    float(last_row['MACD_Signal']),
                    float(last_row['BB_Upper'] / last_row['Close'] - 1),
                    float(last_row['BB_Lower'] / last_row['Close'] - 1),
                    float(last_row['VIX']),
                    float(last_row['Market_Return']),
                    float(last_row['News_Sentiment']),
                    float(last_row['Volatility'])
                ])
                
                # Determine if price went up (for training target)
                price_went_up = hist.iloc[-1]['Close'] > last_row['Close']
                
                logging.info(f"Successfully generated data for {symbol}")
                return features, feature_vector, price_went_up
                
            except Exception as e:
                logging.error(f"Error fetching data for {symbol} (attempt {attempt + 1}): {str(e)}")
                continue
        
        logging.error("Failed to fetch data after all attempts")
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