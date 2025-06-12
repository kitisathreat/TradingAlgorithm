"""
Neural Network Model for Market Pattern Analysis and Signal Generation
This module trains a neural network to analyze market patterns and generate insights
that can be used to enhance the C++ trading engine's decision-making process.
"""

# Standard library imports
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Third-party imports
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import yfinance as yf

# Local imports
from sentiment_analyzer import SentimentAnalyzer
from market_analyzer import MarketAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MarketPatternAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.sentiment_analyzer = SentimentAnalyzer()
        self.market_analyzer = MarketAnalyzer()
        
    def prepare_training_data(self, symbol: str, lookback_days: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data by combining price data, technical indicators, and sentiment.
        Returns features and targets for training.
        """
        try:
            # Fetch historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Get price data
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Calculate technical indicators
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['RSI'] = self._calculate_rsi(df['Close'])
            df['MACD'], df['MACD_Signal'] = self._calculate_macd(df['Close'])
            df['BB_Upper'], df['BB_Lower'] = self._calculate_bollinger_bands(df['Close'])
            
            # Get market context
            df['VIX'] = self._get_vix_data(start_date, end_date)
            df['Market_Return'] = self._get_market_returns(start_date, end_date)
            
            # Get sentiment data
            df['News_Sentiment'] = self._get_historical_sentiment(symbol, start_date, end_date)
            
            # Calculate target variables (what we want to predict)
            df['Price_Change'] = df['Close'].pct_change()
            df['Volatility'] = df['Price_Change'].rolling(window=20).std()
            df['Trend_Strength'] = self._calculate_trend_strength(df['Close'])
            
            # Drop rows with NaN values
            df = df.dropna()
            
            # Prepare features and targets
            feature_columns = [
                'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal',
                'BB_Upper', 'BB_Lower', 'VIX', 'Market_Return',
                'News_Sentiment', 'Volatility'
            ]
            
            target_columns = [
                'Price_Change', 'Trend_Strength'
            ]
            
            X = df[feature_columns].values
            y = df[target_columns].values
            
            return X, y
            
        except Exception as e:
            logging.error(f"Error preparing training data: {e}")
            raise
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal line"""
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
    
    def _get_vix_data(self, start_date: datetime, end_date: datetime) -> pd.Series:
        """Get VIX data for the given period"""
        vix = yf.download('^VIX', start=start_date, end=end_date)
        return vix['Close'] / 100.0  # Convert to decimal
    
    def _get_market_returns(self, start_date: datetime, end_date: datetime) -> pd.Series:
        """Get S&P 500 returns for the given period"""
        spy = yf.download('^GSPC', start=start_date, end=end_date)
        return spy['Close'].pct_change()
    
    def _get_historical_sentiment(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.Series:
        """Get historical sentiment data (placeholder - implement actual sentiment fetching)"""
        # This is a placeholder. In a real implementation, you would fetch historical news
        # and calculate sentiment scores for each day
        return pd.Series(0, index=pd.date_range(start=start_date, end=end_date))
    
    def _calculate_trend_strength(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate trend strength using ADX-like metric"""
        high = prices.rolling(window=period).max()
        low = prices.rolling(window=period).min()
        close = prices
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return (high - low) / atr
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """Build and compile the neural network model"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
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
            
            # Output layers
            layers.Dense(16, activation='relu'),
            layers.Dense(2)  # Price change and trend strength predictions
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        logging.info("Neural network model built successfully")
    
    def train(self, symbol: str, epochs: int = 100, batch_size: int = 32) -> None:
        """Train the model on historical data"""
        try:
            # Prepare data
            X, y = self.prepare_training_data(symbol)
            
            # Reshape input for LSTM [samples, time steps, features]
            X = X.reshape((X.shape[0], 1, X.shape[1]))
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
            X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
            X_test_scaled = self.scaler.transform(X_test_reshaped)
            X_train = X_train_scaled.reshape(X_train.shape)
            X_test = X_test_scaled.reshape(X_test.shape)
            
            # Build model if not already built
            if self.model is None:
                self.build_model(input_shape=(1, X.shape[2]))
            
            # Train model
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Save model and scaler
            self.model.save(f'market_pattern_model_{symbol}.keras')
            joblib.dump(self.scaler, f'market_pattern_scaler_{symbol}.joblib')
            
            logging.info(f"Model trained and saved for {symbol}")
            
        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise
    
    def predict_market_insights(self, symbol: str) -> Dict[str, float]:
        """
        Generate market insights for the C++ engine
        Returns a dictionary of insights including:
        - predicted_price_change: Expected price movement
        - trend_strength: Strength of the current trend
        - market_regime: Suggested market regime (BULLISH/BEARISH/NEUTRAL)
        - confidence: Model's confidence in the prediction
        """
        try:
            # Get latest data
            X, _ = self.prepare_training_data(symbol, lookback_days=30)
            X = X.reshape((1, 1, X.shape[1]))
            
            # Scale features
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.transform(X_reshaped)
            X = X_scaled.reshape(X.shape)
            
            # Make prediction
            prediction = self.model.predict(X, verbose=0)[0]
            predicted_price_change, trend_strength = prediction
            
            # Calculate confidence based on model's performance metrics
            confidence = 1.0 - min(abs(predicted_price_change), 1.0)
            
            # Determine market regime
            if trend_strength > 0.7 and predicted_price_change > 0:
                market_regime = "BULLISH"
            elif trend_strength > 0.7 and predicted_price_change < 0:
                market_regime = "BEARISH"
            else:
                market_regime = "NEUTRAL"
            
            return {
                "predicted_price_change": float(predicted_price_change),
                "trend_strength": float(trend_strength),
                "market_regime": market_regime,
                "confidence": float(confidence)
            }
            
        except Exception as e:
            logging.error(f"Error generating market insights: {e}")
            return {
                "predicted_price_change": 0.0,
                "trend_strength": 0.0,
                "market_regime": "NEUTRAL",
                "confidence": 0.0
            }

def main():
    """Main function to train models for all symbols"""
    try:
        # Get symbols from config
        from config import SYMBOLS_TO_TRADE
        
        analyzer = MarketPatternAnalyzer()
        
        for symbol in SYMBOLS_TO_TRADE:
            logging.info(f"Training model for {symbol}...")
            analyzer.train(symbol)
            
            # Test prediction
            insights = analyzer.predict_market_insights(symbol)
            logging.info(f"\nMarket insights for {symbol}:")
            logging.info(f"Predicted price change: {insights['predicted_price_change']:.2%}")
            logging.info(f"Trend strength: {insights['trend_strength']:.2f}")
            logging.info(f"Market regime: {insights['market_regime']}")
            logging.info(f"Confidence: {insights['confidence']:.2%}")
            
    except Exception as e:
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()