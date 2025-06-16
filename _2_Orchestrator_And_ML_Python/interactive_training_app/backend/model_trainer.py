"""
Advanced Neural Network Model Trainer for Trading Algorithm
Combines technical analysis with sentiment analysis of human trading decisions
Uses TensorFlow neural networks and VADER sentiment analysis
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
import random
from typing import Dict, List, Tuple, Optional

# Import ML packages
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import yfinance as yf
    ML_IMPORTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML imports not available: {e}")
    ML_IMPORTS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingNeuralNetwork:
    """
    Neural Network that learns to mimic human trading decisions
    Combines technical indicators with sentiment analysis
    """
    
    def __init__(self, input_dim=20):
        self.input_dim = input_dim
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def build_model(self):
        """Build the neural network architecture"""
        model = keras.Sequential([
            # Input layer with technical indicators + sentiment features
            layers.Dense(128, activation='relu', input_shape=(self.input_dim,)),
            layers.Dropout(0.3),
            
            # Hidden layers for complex pattern recognition
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer for trading decisions (BUY, SELL, HOLD)
            layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, epochs=50, validation_split=0.2):
        """Train the neural network"""
        if self.model is None:
            self.build_model()
        
        # Fit the scaler and encode labels
        X_train_scaled = self.scaler.fit_transform(X_train)
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Early stopping and model checkpointing
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = self.model.fit(
            X_train_scaled, y_train_encoded,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        return history
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        predicted_classes = self.label_encoder.inverse_transform(np.argmax(predictions, axis=1))
        confidence_scores = np.max(predictions, axis=1)
        
        return predicted_classes, confidence_scores

class ModelTrainer:
    """
    Advanced Model Trainer that learns from human trading intuition
    Combines technical analysis with sentiment analysis of trading descriptions
    """
    
    def __init__(self):
        self.repo_root = Path(__file__).parent.parent.parent.parent
        self.model_state_file = self.repo_root / "model_state.json"
        self.training_data_file = self.repo_root / "training_data.json"
        
        # Load SP100 symbols
        symbols_file = Path(__file__).parent.parent / "sp100_symbols.json"
        if symbols_file.exists():
            with open(symbols_file, 'r') as f:
                self.symbols = json.load(f)
        else:
            logger.warning("SP100 symbols file not found, using fallback symbols")
            self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
        
        # Initialize components
        self.sentiment_analyzer = SentimentIntensityAnalyzer() if ML_IMPORTS_AVAILABLE else None
        self.neural_network = TradingNeuralNetwork() if ML_IMPORTS_AVAILABLE else None
        self.training_examples = []
        
        # Load existing training data
        self._load_training_data()
        
        logger.info(f"ModelTrainer initialized with {len(self.symbols)} symbols")
    
    def _load_training_data(self):
        """Load existing training data from file"""
        try:
            if self.training_data_file.exists():
                with open(self.training_data_file, 'r') as f:
                    self.training_examples = json.load(f)
                logger.info(f"Loaded {len(self.training_examples)} existing training examples")
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            self.training_examples = []
    
    def _save_training_data(self):
        """Save training data to file"""
        try:
            with open(self.training_data_file, 'w') as f:
                json.dump(self.training_examples, f, indent=2, default=str)
            logger.info(f"Saved {len(self.training_examples)} training examples")
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
    
    def get_historical_stock_data(self, symbol: str, years_back: int = 25) -> pd.DataFrame:
        """
        Fetch historical stock data going back up to 25 years
        Addresses the insufficient data problem from your logs
        """
        try:
            if not ML_IMPORTS_AVAILABLE:
                return self._generate_synthetic_historical_data(symbol, years_back)
            
            # Use current date minus buffer to ensure we get real data
            # Make timezone-aware to match yfinance data
            current_date = datetime.now()
            end_date = current_date - timedelta(days=5)  # 5 days ago to ensure data availability
            start_date = end_date - timedelta(days=years_back * 365)
            
            # Validate dates to prevent future date issues
            if start_date >= end_date:
                logger.error(f"Invalid date range: start_date {start_date} >= end_date {end_date}")
                return self._generate_synthetic_historical_data(symbol, years_back)
            
            logger.info(f"Fetching {years_back} years of data for {symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (using {current_date.year} reference date)")
            
            # Try different approaches to get maximum data
            ticker = yf.Ticker(symbol)
            
            # Try period first (more reliable) - but limit to reasonable periods
            for period in ["5y", "2y", "1y", "6mo"]:
                try:
                    data = ticker.history(period=period)
                    if not data.empty and len(data) > 30:  # At least 30 days of data
                        # Make end_date timezone-aware to match data.index timezone
                        if hasattr(data.index, 'tz') and data.index.tz is not None:
                            # Convert end_date to the same timezone as the data
                            end_date_tz = pd.Timestamp(end_date).tz_localize(data.index.tz)
                        else:
                            # If data has no timezone, localize end_date to UTC
                            end_date_tz = pd.Timestamp(end_date).tz_localize('UTC')
                        
                        # Filter data to end at our end_date
                        filtered_data = data[data.index <= end_date_tz]
                        if not filtered_data.empty and len(filtered_data) > 30:
                            logger.info(f"Successfully fetched {len(filtered_data)} days of {symbol} data using period={period}")
                            logger.info(f"Data range: {filtered_data.index.min().strftime('%Y-%m-%d')} to {filtered_data.index.max().strftime('%Y-%m-%d')}")
                            return self._clean_and_enhance_data(filtered_data)
                        else:
                            logger.warning(f"Data for {symbol} filtered to end_date is insufficient, trying next period")
                            continue
                except Exception as e:
                    logger.warning(f"Period {period} failed for {symbol}: {e}")
                    continue
            
            # Fallback to date range with strict validation
            try:
                # Use a more conservative date range (max 2 years back)
                safe_end_date = end_date
                safe_start_date = safe_end_date - timedelta(days=min(years_back * 365, 730))  # Max 2 years
                
                data = ticker.history(start=safe_start_date, end=safe_end_date)
                if not data.empty:
                    logger.info(f"Successfully fetched {len(data)} days of {symbol} data using date range")
                    logger.info(f"Data range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
                    return self._clean_and_enhance_data(data)
            except Exception as e:
                logger.warning(f"Date range fetch failed for {symbol}: {e}")
            
            # Final fallback to synthetic data
            logger.info(f"Using synthetic data for {symbol}")
            return self._generate_synthetic_historical_data(symbol, min(years_back, 2))  # Max 2 years for synthetic
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return self._generate_synthetic_historical_data(symbol, min(years_back, 2))  # Max 2 years for synthetic
    
    def _clean_and_enhance_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and enhance stock data with technical indicators"""
        df = data.copy()
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns. Available: {df.columns.tolist()}")
            return self._generate_synthetic_historical_data("UNKNOWN", 1)
        
        # Remove any invalid data
        df = df.dropna()
        df = df[df['Volume'] > 0]
        df = df[df['Close'] > 0]  # Ensure positive prices
        
        # Ensure index is properly formatted as datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex, attempting to convert")
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                logger.error(f"Could not convert index to datetime: {e}")
                return self._generate_synthetic_historical_data("UNKNOWN", 1)
        
        # Sort by date to ensure chronological order
        df = df.sort_index()
        
        # Calculate technical indicators
        df = self._add_comprehensive_technical_indicators(df)
        
        logger.info(f"Cleaned data shape: {df.shape}, date range: {df.index.min()} to {df.index.max()}")
        return df
    
    def _add_comprehensive_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators for neural network features"""
        df = data.copy()
        
        # Moving Averages
        for window in [5, 10, 20, 50, 200]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
        
        # Volatility measures
        df['Volatility_20'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()
        df['ATR'] = ((df['High'] - df['Low']).rolling(window=14).mean())
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price momentum
        for period in [1, 5, 10, 20]:
            df[f'Price_Change_{period}d'] = df['Close'].pct_change(periods=period)
            df[f'Price_Momentum_{period}d'] = df['Close'] / df['Close'].shift(period) - 1
        
        # Market structure
        df['Higher_High'] = (df['High'] > df['High'].shift(1)).astype(int)
        df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
        
        return df
    
    def _generate_synthetic_historical_data(self, symbol: str, years: int) -> pd.DataFrame:
        """Generate realistic synthetic historical data"""
        logger.info(f"Generating {years} years of synthetic data for {symbol}")
        
        # Create date range (business days only) - use recent dates
        end_date = datetime.now() - timedelta(days=5)  # 5 days ago
        start_date = end_date - timedelta(days=years * 365)
        date_range = pd.bdate_range(start=start_date, end=end_date)
        
        if len(date_range) < 30:
            logger.warning(f"Generated date range too short ({len(date_range)} days), using 30 days")
            end_date = datetime.now() - timedelta(days=5)
            start_date = end_date - timedelta(days=30)
            date_range = pd.bdate_range(start=start_date, end=end_date)
        
        # Generate realistic price movements with trends and cycles
        np.random.seed(hash(symbol) % 2**32)
        base_price = np.random.uniform(20, 500)  # Realistic stock price range
        
        # Add long-term trend and cyclical components
        trend = np.linspace(0, np.random.uniform(-0.3, 0.8), len(date_range))  # More conservative trend
        cycle = np.sin(np.linspace(0, years * 2 * np.pi, len(date_range))) * 0.05  # Smaller cycles
        
        # Generate daily returns with realistic volatility clustering
        returns = []
        volatility = 0.02  # 2% daily volatility
        
        for i in range(len(date_range)):
            # Volatility clustering
            volatility = volatility * 0.95 + np.random.exponential(0.001)
            volatility = np.clip(volatility, 0.005, 0.08)  # Between 0.5% and 8%
            
            # Daily return with trend and cycle
            daily_return = np.random.normal(0.0003 + trend[i]/252 + cycle[i]/252, volatility)
            returns.append(daily_return)
        
        # Calculate prices
        prices = [base_price]
        for ret in returns:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 1.0))  # Ensure minimum price of $1
        
        prices = prices[1:]  # Remove initial price
        
        # Generate OHLCV data with realistic relationships
        highs = []
        lows = []
        opens = []
        volumes = []
        
        for i, close_price in enumerate(prices):
            # Generate realistic OHLC relationships
            if i == 0:
                open_price = close_price * (1 + np.random.normal(0, 0.01))
            else:
                open_price = prices[i-1] * (1 + np.random.normal(0, 0.01))
            
            # High and low should bracket open and close
            price_range = close_price * np.random.uniform(0.005, 0.03)  # 0.5% to 3% range
            high_price = max(open_price, close_price) + price_range * np.random.uniform(0.3, 1.0)
            low_price = min(open_price, close_price) - price_range * np.random.uniform(0.3, 1.0)
            
            # Ensure low <= min(open, close) <= max(open, close) <= high
            low_price = min(low_price, open_price, close_price)
            high_price = max(high_price, open_price, close_price)
            
            # Generate realistic volume
            volume = np.random.randint(100000, 10000000)
            
            highs.append(high_price)
            lows.append(low_price)
            opens.append(open_price)
            volumes.append(volume)
        
        df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': volumes
        }, index=date_range)
        
        logger.info(f"Generated synthetic data: {len(df)} days, price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
        return self._add_comprehensive_technical_indicators(df)
    
    def get_random_stock_data(self) -> Tuple[Dict, pd.DataFrame, str]:
        """
        Get random stock data for training presentation
        Returns: (features_dict, historical_data, symbol)
        """
        try:
            # Select random symbol and date
            symbol = random.choice(self.symbols)
            
            # Get historical data
            historical_data = self.get_historical_stock_data(symbol, years_back=25)
            
            if historical_data.empty or len(historical_data) < 252:
                logger.error(f"Insufficient historical data for {symbol}")
                return None, None, None
            
            # Select random date (ensure enough history for indicators and future data)
            min_idx = 200  # Need history for indicators
            max_idx = len(historical_data) - 30  # Need future data to know actual outcome
            
            if min_idx >= max_idx:
                logger.error(f"Not enough data range for {symbol}")
                return None, None, None
            
            random_idx = random.randint(min_idx, max_idx)
            current_date = historical_data.index[random_idx]
            current_row = historical_data.iloc[random_idx]
            
            # Get historical context (last 252 days = 1 year)
            history_start_idx = max(0, random_idx - 252)
            historical_context = historical_data.iloc[history_start_idx:random_idx+1]
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(historical_context)
            
            # Create comprehensive feature dictionary
            features = {
                'symbol': symbol,
                'date': current_date.strftime('%Y-%m-%d'),
                'current_price': float(current_row['Close']),
                'volume': int(current_row['Volume']),
                
                # Technical indicators
                'rsi': float(current_row.get('RSI', 50)),
                'macd': float(current_row.get('MACD', 0)),
                'macd_signal': float(current_row.get('MACD_Signal', 0)),
                'bb_position': float(current_row.get('BB_Position', 0.5)),
                'volatility': float(current_row.get('Volatility_20', 0.02)),
                'atr': float(current_row.get('ATR', 0)),
                'volume_ratio': float(current_row.get('Volume_Ratio', 1.0)),
                
                # Moving averages
                'sma_10': float(current_row.get('SMA_10', current_row['Close'])),
                'sma_20': float(current_row.get('SMA_20', current_row['Close'])),
                'sma_50': float(current_row.get('SMA_50', current_row['Close'])),
                'ema_10': float(current_row.get('EMA_10', current_row['Close'])),
                'ema_20': float(current_row.get('EMA_20', current_row['Close'])),
                
                # Price momentum
                'price_change_1d': float(current_row.get('Price_Change_1d', 0)),
                'price_change_5d': float(current_row.get('Price_Change_5d', 0)),
                'price_change_20d': float(current_row.get('Price_Change_20d', 0)),
                
                # Performance metrics
                **performance_metrics,
                
                # Future outcome (for training validation)
                'actual_outcome': self._calculate_future_outcome(historical_data, random_idx)
            }
            
            logger.info(f"Generated training example: {symbol} on {current_date.strftime('%Y-%m-%d')}")
            return features, historical_context, symbol
            
        except Exception as e:
            logger.error(f"Error generating random stock data: {e}")
            return None, None, None
    
    def _calculate_performance_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        if len(data) < 20:
            return {}
        
        try:
            recent_data = data.tail(252) if len(data) >= 252 else data
            
            # Returns over different periods
            periods = [5, 20, 60, 252]
            metrics = {}
            
            for period in periods:
                if len(recent_data) >= period:
                    period_return = (recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[-period] - 1) * 100
                    metrics[f'return_{period}d'] = round(period_return, 2)
                    
                    # Volatility over period
                    period_vol = recent_data['Close'].tail(period).pct_change().std() * np.sqrt(252) * 100
                    metrics[f'volatility_{period}d'] = round(period_vol, 2)
            
            # Sharpe ratio (simplified)
            if len(recent_data) >= 60:
                returns = recent_data['Close'].pct_change().dropna()
                if returns.std() > 0:
                    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
                    metrics['sharpe_ratio'] = round(sharpe, 2)
            
            # Maximum drawdown
            rolling_max = recent_data['Close'].expanding().max()
            drawdown = (recent_data['Close'] - rolling_max) / rolling_max
            metrics['max_drawdown'] = round(drawdown.min() * 100, 2)
            
            # Current vs historical highs/lows
            metrics['distance_from_52w_high'] = round((recent_data['Close'].iloc[-1] / recent_data['Close'].max() - 1) * 100, 2)
            metrics['distance_from_52w_low'] = round((recent_data['Close'].iloc[-1] / recent_data['Close'].min() - 1) * 100, 2)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _calculate_future_outcome(self, data: pd.DataFrame, current_idx: int) -> str:
        """Calculate what actually happened after the presented date"""
        try:
            if current_idx + 20 >= len(data):
                return "HOLD"  # Not enough future data
            
            current_price = data.iloc[current_idx]['Close']
            future_price_5d = data.iloc[current_idx + 5]['Close']
            future_price_20d = data.iloc[current_idx + 20]['Close']
            
            # Calculate returns
            return_5d = (future_price_5d - current_price) / current_price
            return_20d = (future_price_20d - current_price) / current_price
            
            # Determine outcome based on thresholds
            if return_5d > 0.03 or return_20d > 0.05:  # Strong positive performance
                return "BUY"
            elif return_5d < -0.03 or return_20d < -0.05:  # Strong negative performance
                return "SELL"
            else:
                return "HOLD"
                
        except Exception as e:
            logger.error(f"Error calculating future outcome: {e}")
            return "HOLD"
    
    def analyze_sentiment_and_keywords(self, trading_description: str) -> Dict:
        """
        Analyze sentiment and extract keywords from trading description
        Uses VADER sentiment analysis
        """
        if not self.sentiment_analyzer:
            logger.warning("Sentiment analyzer not available")
            return {'sentiment_score': 0.0, 'keywords': [], 'confidence': 'medium'}
        
        try:
            # Get VADER sentiment scores
            sentiment_scores = self.sentiment_analyzer.polarity_scores(trading_description)
            
            # Extract keywords (simple approach - can be enhanced)
            keywords = self._extract_trading_keywords(trading_description)
            
            # Determine confidence level from description
            confidence = self._determine_confidence_level(trading_description)
            
            result = {
                'sentiment_score': sentiment_scores['compound'],
                'positive': sentiment_scores['pos'],
                'negative': sentiment_scores['neg'],
                'neutral': sentiment_scores['neu'],
                'keywords': keywords,
                'confidence': confidence,
                'raw_text': trading_description
            }
            
            logger.info(f"Sentiment analysis: score={result['sentiment_score']:.3f}, confidence={confidence}")
            return result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {'sentiment_score': 0.0, 'keywords': [], 'confidence': 'medium'}
    
    def _extract_trading_keywords(self, text: str) -> List[str]:
        """Extract trading-relevant keywords from text"""
        # Define trading-related keywords
        trading_keywords = {
            # Sentiment keywords
            'bullish', 'bearish', 'optimistic', 'pessimistic', 'confident', 'uncertain',
            'strong', 'weak', 'volatile', 'stable', 'risky', 'safe',
            
            # Action keywords  
            'buy', 'sell', 'hold', 'accumulate', 'distribute', 'short', 'long',
            'enter', 'exit', 'add', 'reduce', 'scale', 'position',
            
            # Technical keywords
            'breakout', 'support', 'resistance', 'trend', 'momentum', 'oversold', 'overbought',
            'reversal', 'continuation', 'pattern', 'indicator', 'signal',
            
            # Fundamental keywords
            'earnings', 'revenue', 'growth', 'valuation', 'expensive', 'cheap',
            'overvalued', 'undervalued', 'catalyst', 'news', 'announcement'
        }
        
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in trading_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _determine_confidence_level(self, text: str) -> str:
        """Determine confidence level from text description"""
        text_lower = text.lower()
        
        high_confidence_words = ['very', 'extremely', 'definitely', 'absolutely', 'certain', 'sure', 'confident', 'strong']
        low_confidence_words = ['maybe', 'perhaps', 'might', 'could', 'uncertain', 'unsure', 'weak', 'cautious']
        
        high_count = sum(1 for word in high_confidence_words if word in text_lower)
        low_count = sum(1 for word in low_confidence_words if word in text_lower)
        
        if high_count > low_count:
            return 'high'
        elif low_count > high_count:
            return 'low'
        else:
            return 'medium'
    
    def add_training_example(self, features: Dict, sentiment_analysis: Dict, user_decision: str) -> bool:
        """
        Add a new training example combining technical features with sentiment analysis
        """
        try:
            # Create comprehensive training example
            training_example = {
                'timestamp': datetime.now().isoformat(),
                'technical_features': features,
                'sentiment_analysis': sentiment_analysis,
                'user_decision': user_decision,
                'actual_outcome': features.get('actual_outcome', 'UNKNOWN')
            }
            
            # Add to training examples
            self.training_examples.append(training_example)
            
            # Save to file
            self._save_training_data()
            
            logger.info(f"Added training example: {features['symbol']} - {user_decision}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding training example: {e}")
            return False
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for the neural network
        Combines technical indicators with sentiment features
        """
        if len(self.training_examples) == 0:
            return np.array([]), np.array([])
        
        try:
            features_list = []
            labels_list = []
            
            for example in self.training_examples:
                # Extract technical features
                technical = example['technical_features']
                sentiment = example['sentiment_analysis']
                
                # Create feature vector (20 features)
                feature_vector = [
                    technical.get('current_price', 0) / 1000,  # Normalize price
                    technical.get('volume', 0) / 1e6,  # Normalize volume
                    technical.get('rsi', 50) / 100,
                    technical.get('macd', 0),
                    technical.get('bb_position', 0.5),
                    technical.get('volatility', 0.02),
                    technical.get('volume_ratio', 1.0),
                    technical.get('price_change_1d', 0),
                    technical.get('price_change_5d', 0),
                    technical.get('price_change_20d', 0),
                    technical.get('return_5d', 0) / 100,
                    technical.get('return_20d', 0) / 100,
                    technical.get('return_60d', 0) / 100,
                    technical.get('volatility_20d', 0) / 100,
                    technical.get('max_drawdown', 0) / 100,
                    
                    # Sentiment features
                    sentiment.get('sentiment_score', 0),
                    sentiment.get('positive', 0),
                    sentiment.get('negative', 0),
                    len(sentiment.get('keywords', [])) / 10,  # Normalize keyword count
                    {'high': 1.0, 'medium': 0.5, 'low': 0.0}.get(sentiment.get('confidence', 'medium'), 0.5)
                ]
                
                features_list.append(feature_vector)
                labels_list.append(example['user_decision'])
            
            X = np.array(features_list)
            y = np.array(labels_list)
            
            logger.info(f"Prepared training data: {X.shape[0]} examples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([])
    
    def train_neural_network(self) -> bool:
        """
        Train the neural network with collected examples
        """
        if not ML_IMPORTS_AVAILABLE:
            logger.error("ML packages not available for training")
            return False
        
        if len(self.training_examples) < 10:
            logger.warning(f"Insufficient training data: {len(self.training_examples)} examples (need at least 10)")
            return False
        
        try:
            logger.info("Starting neural network training...")
            
            # Prepare training data
            X, y = self.prepare_training_data()
            
            if X.size == 0:
                logger.error("No valid training data prepared")
                return False
            
            # Train the neural network
            history = self.neural_network.train(X, y, epochs=100)
            
            # Calculate final accuracy
            final_accuracy = max(history.history.get('val_accuracy', [0]))
            
            # Save model state
            self.save_model_state(
                is_trained=True,
                training_examples=len(self.training_examples),
                model_accuracy=final_accuracy
            )
            
            logger.info(f"Neural network training completed! Final accuracy: {final_accuracy:.2%}")
            return True
            
        except Exception as e:
            logger.error(f"Error during neural network training: {e}")
            return False
    
    def make_prediction(self, features: Dict) -> Dict:
        """
        Make a trading prediction using the trained neural network
        """
        if not self.neural_network or not self.neural_network.is_trained:
            return {'prediction': 'HOLD', 'confidence': 0.5, 'error': 'Model not trained'}
        
        try:
            # Prepare feature vector (same as training)
            feature_vector = np.array([[
                features.get('current_price', 0) / 1000,
                features.get('volume', 0) / 1e6,
                features.get('rsi', 50) / 100,
                features.get('macd', 0),
                features.get('bb_position', 0.5),
                features.get('volatility', 0.02),
                features.get('volume_ratio', 1.0),
                features.get('price_change_1d', 0),
                features.get('price_change_5d', 0),
                features.get('price_change_20d', 0),
                features.get('return_5d', 0) / 100,
                features.get('return_20d', 0) / 100,
                features.get('return_60d', 0) / 100,
                features.get('volatility_20d', 0) / 100,
                features.get('max_drawdown', 0) / 100,
                0, 0, 0, 0, 0.5  # Placeholder for sentiment features
            ]])
            
            # Make prediction
            prediction, confidence = self.neural_network.predict(feature_vector)
            
            return {
                'prediction': prediction[0],
                'confidence': float(confidence[0]),
                'model_trained': True
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {'prediction': 'HOLD', 'confidence': 0.5, 'error': str(e)}
    
    def get_model_state(self) -> Dict:
        """Get current model training state"""
        try:
            if self.model_state_file.exists():
                with open(self.model_state_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error reading model state: {e}")
        
        return {
            'is_trained': False,
            'training_examples': len(self.training_examples),
            'last_training_date': None,
            'model_accuracy': 0.0
        }
    
    def save_model_state(self, is_trained: bool = False, training_examples: int = 0, model_accuracy: float = 0.0):
        """Save model training state"""
        try:
            state = {
                'is_trained': is_trained,
                'training_examples': training_examples,
                'last_training_date': datetime.now().isoformat(),
                'model_accuracy': model_accuracy
            }
            with open(self.model_state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"Model state saved: {training_examples} examples, accuracy: {model_accuracy:.2%}")
        except Exception as e:
            logger.error(f"Error saving model state: {e}")
    
    def get_training_stats(self) -> Dict:
        """Get comprehensive training statistics"""
        return {
            'total_examples': len(self.training_examples),
            'symbols_trained': len(set(ex['technical_features']['symbol'] for ex in self.training_examples)),
            'decision_distribution': self._get_decision_distribution(),
            'sentiment_distribution': self._get_sentiment_distribution(),
            'model_state': self.get_model_state()
        }
    
    def _get_decision_distribution(self) -> Dict:
        """Get distribution of user decisions"""
        if not self.training_examples:
            return {}
        
        decisions = [ex['user_decision'] for ex in self.training_examples]
        unique, counts = np.unique(decisions, return_counts=True)
        return dict(zip(unique, counts.tolist()))
    
    def _get_sentiment_distribution(self) -> Dict:
        """Get distribution of sentiment scores"""
        if not self.training_examples:
            return {}
        
        sentiments = [ex['sentiment_analysis']['sentiment_score'] for ex in self.training_examples]
        return {
            'mean_sentiment': float(np.mean(sentiments)),
            'std_sentiment': float(np.std(sentiments)),
            'positive_count': sum(1 for s in sentiments if s > 0.1),
            'negative_count': sum(1 for s in sentiments if s < -0.1),
            'neutral_count': sum(1 for s in sentiments if -0.1 <= s <= 0.1)
        }