#!/usr/bin/env python3
"""
Neural Network Trading System - Flask Web Interface
Web-based version of the PyQt5 GUI that runs in a browser
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit
import yfinance as yf

# Add parent directory to path to import from orchestrator
REPO_ROOT = Path(__file__).parent.parent.parent
ORCHESTRATOR_PATH = REPO_ROOT / "_2_Orchestrator_And_ML_Python"
sys.path.append(str(ORCHESTRATOR_PATH))

try:
    from interactive_training_app.backend.model_trainer import ModelTrainer
    MODEL_TRAINER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ModelTrainer not available: {e}")
    MODEL_TRAINER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'trading_algorithm_secret_key_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
model_trainer = None
current_stock_data = None
current_stock_info = None
current_symbol = None
training_progress = 0
training_active = False

# Import stock selection utilities
try:
    from stock_selection_utils import StockSelectionManager
    stock_manager = StockSelectionManager()
    STOCK_MANAGER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"StockSelectionManager not available: {e}")
    stock_manager = None
    STOCK_MANAGER_AVAILABLE = False
    # Fallback stock list
    STOCK_SYMBOLS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "AMD", "INTC",
        "JPM", "BAC", "WMT", "JNJ", "PG", "UNH", "HD", "MA", "V", "DIS",
        "PYPL", "CRM", "ABT", "KO", "PEP", "TMO", "COST", "MRK", "PFE", "TXN"
    ]

def initialize_model_trainer():
    """Initialize the ModelTrainer instance"""
    global model_trainer
    if MODEL_TRAINER_AVAILABLE:
        try:
            model_trainer = ModelTrainer()
            logger.info("ModelTrainer initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ModelTrainer: {e}")
            return False
    return False

def get_stock_data(symbol, days=30):
    """Fetch stock data from Yahoo Finance"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            return None, None
        
        # Calculate basic metrics
        current_price = data['Close'].iloc[-1]
        volume = data['Volume'].iloc[-1]
        
        # Get market cap (approximate)
        try:
            info = stock.info
            market_cap = info.get('marketCap', 0)
        except:
            market_cap = 0
        
        stock_info = {
            'current_price': float(current_price),
            'volume': int(volume),
            'market_cap': int(market_cap) if market_cap else 0,
            'symbol': symbol
        }
        
        return stock_info, data
        
    except Exception as e:
        logger.error(f"Error fetching stock data for {symbol}: {e}")
        return None, None

def calculate_technical_indicators(data):
    """Calculate technical indicators for the stock data"""
    if data is None or data.empty:
        return {}
    
    try:
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # Moving Averages
        sma_20 = data['Close'].rolling(window=20).mean()
        ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
        
        # Bollinger Bands
        sma = data['Close'].rolling(window=20).mean()
        std = data['Close'].rolling(window=20).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        bollinger_width = (std.iloc[-1] / sma.iloc[-1]) * 100 if not pd.isna(sma.iloc[-1]) else 0
        
        # ATR (Average True Range)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(14).mean().iloc[-1] if len(data) >= 14 else true_range.mean()
        
        # Volume metrics
        volume_sma = data['Volume'].rolling(window=20).mean()
        volume_ratio = data['Volume'].iloc[-1] / volume_sma.iloc[-1] if not pd.isna(volume_sma.iloc[-1]) else 1.0
        
        # Price change
        price_change = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
        
        # Volatility metrics
        returns = data['Close'].pct_change().dropna()
        daily_vol = returns.std() * 100
        weekly_vol = returns.rolling(5).std().iloc[-1] * 100 if len(returns) >= 5 else daily_vol
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Price metrics
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        price_change_1d = current_price - prev_price
        price_change_pct_1d = (price_change_1d / prev_price) * 100 if prev_price != 0 else 0
        high_price = data['High'].max()
        low_price = data['Low'].min()
        avg_price = data['Close'].mean()
        
        # Volume metrics
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].mean()
        
        return {
            'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0,
            'macd': float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0.0,
            'macd_signal': float(signal.iloc[-1]) if not pd.isna(signal.iloc[-1]) else 0.0,
            'sma_20': float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else float(current_price),
            'ema_12': float(ema_12.iloc[-1]) if not pd.isna(ema_12.iloc[-1]) else float(current_price),
            'bb_upper': float(upper_band.iloc[-1]) if not pd.isna(upper_band.iloc[-1]) else float(current_price),
            'bb_lower': float(lower_band.iloc[-1]) if not pd.isna(lower_band.iloc[-1]) else float(current_price),
            'bb_middle': float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else float(current_price),
            'bb_width': float(bollinger_width),
            'atr': float(atr),
            'volume_ratio': float(volume_ratio),
            'price_change': float(price_change),
            'price_change_1d': float(price_change_1d),
            'price_change_pct_1d': float(price_change_pct_1d),
            'volatility': float(volatility),
            'daily_vol': float(daily_vol),
            'weekly_vol': float(weekly_vol),
            'current_price': float(current_price),
            'high_price': float(high_price),
            'low_price': float(low_price),
            'avg_price': float(avg_price),
            'current_volume': int(current_volume),
            'avg_volume': float(avg_volume),
            'trend': 'UP' if price_change > 0 else 'DOWN'
        }
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        return {}

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', 
                         model_trainer_available=MODEL_TRAINER_AVAILABLE,
                         stock_manager_available=STOCK_MANAGER_AVAILABLE)

@app.route('/api/get_stock_options')
def get_stock_options():
    """Get available stock options including special selections"""
    try:
        # Special options first
        options = [
            {'symbol': 'random', 'description': '🎲 Random Pick', 'type': 'special'},
            {'symbol': 'optimized', 'description': '🚀 Optimized Pick', 'type': 'special'},
            {'symbol': 'custom', 'description': '✏️ Custom Ticker', 'type': 'special'}
        ]
        
        # Add S&P100 stocks sorted by market cap
        try:
            sp100_file = Path(__file__).parent.parent.parent / "_2_Orchestrator_And_ML_Python" / "sp100_symbols_with_market_cap.json"
            with open(sp100_file, 'r') as f:
                sp100_data = json.load(f)
            
            # Sort by market cap in descending order
            sp100_data.sort(key=lambda x: x.get('market_cap', 0), reverse=True)
            
            for stock in sp100_data:
                options.append({
                    'symbol': stock['symbol'],
                    'description': f"{stock['symbol']} - {stock['name']}",
                    'type': 'stock',
                    'market_cap': stock.get('market_cap', 0)
                })
                
        except Exception as e:
            logger.warning(f"Could not load S&P100 data: {e}")
            # Fallback to basic stocks
            fallback_stocks = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "AMD", "INTC",
                "JPM", "BAC", "WMT", "JNJ", "PG", "UNH", "HD", "MA", "V", "DIS",
                "PYPL", "CRM", "ABT", "KO", "PEP", "TMO", "COST", "MRK", "PFE", "TXN"
            ]
            for symbol in fallback_stocks:
                options.append({
                    'symbol': symbol,
                    'description': symbol,
                    'type': 'stock'
                })
        
        return jsonify({
            'success': True,
            'options': options
        })
    except Exception as e:
        logger.error(f"Error getting stock options: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/load_stock', methods=['POST'])
def load_stock():
    """Load stock data"""
    global current_stock_data, current_stock_info, current_symbol
    
    data = request.get_json()
    symbol = data.get('symbol', 'AAPL')
    days = data.get('days', 30)
    
    # Handle special stock selections
    if symbol == 'random' and stock_manager:
        symbol, name = stock_manager.get_random_stock()
        logger.info(f"Random stock selected: {symbol} ({name})")
    elif symbol == 'optimized' and stock_manager:
        symbol, name = stock_manager.get_optimized_pick()
        logger.info(f"Optimized stock selected: {symbol} ({name})")
    elif symbol == 'custom':
        custom_symbol = data.get('custom_symbol', '').strip().upper()
        if not custom_symbol:
            return jsonify({
                'success': False,
                'error': 'Please provide a custom ticker symbol'
            })
        # Validate custom ticker
        if stock_manager:
            is_valid, message, company_name = stock_manager.validate_custom_ticker(custom_symbol)
            if not is_valid:
                return jsonify({
                    'success': False,
                    'error': message
                })
        symbol = custom_symbol
    
    try:
        stock_info, stock_data = get_stock_data(symbol, days)
        
        if stock_info and stock_data is not None:
            current_stock_info = stock_info
            current_stock_data = stock_data
            current_symbol = symbol
            
            # Calculate technical indicators
            indicators = calculate_technical_indicators(stock_data)
            
            # Prepare chart data
            chart_data = {
                'dates': stock_data.index.strftime('%Y-%m-%d').tolist(),
                'prices': stock_data['Close'].tolist(),
                'volumes': stock_data['Volume'].tolist(),
                'highs': stock_data['High'].tolist(),
                'lows': stock_data['Low'].tolist(),
                'opens': stock_data['Open'].tolist()
            }
            
            return jsonify({
                'success': True,
                'stock_info': stock_info,
                'indicators': indicators,
                'chart_data': chart_data
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Could not fetch data for {symbol}'
            })
            
    except Exception as e:
        logger.error(f"Error loading stock data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/submit_decision', methods=['POST'])
def submit_decision():
    """Submit trading decision for training"""
    global model_trainer, current_stock_data
    
    if not model_trainer:
        return jsonify({'success': False, 'error': 'Model trainer not available'})
    
    if current_stock_data is None:
        return jsonify({'success': False, 'error': 'No stock data loaded'})
    
    data = request.get_json()
    decision = data.get('decision', 'HOLD')
    reasoning = data.get('reasoning', '')
    
    try:
        # Calculate features
        features = model_trainer._calculate_performance_metrics(current_stock_data)
        
        # Analyze sentiment
        sentiment_analysis = model_trainer.analyze_sentiment_and_keywords(reasoning)
        
        # Add training example
        success = model_trainer.add_training_example(features, sentiment_analysis, decision)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Training example added: {decision} for {current_symbol}',
                'training_examples': len(model_trainer.training_examples)
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to add training example'
            })
            
    except Exception as e:
        logger.error(f"Error submitting decision: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/train_model', methods=['POST'])
def train_model():
    """Train the neural network model"""
    global model_trainer, training_progress, training_active
    
    if not model_trainer:
        return jsonify({'success': False, 'error': 'Model trainer not available'})
    
    if not model_trainer.training_examples:
        return jsonify({'success': False, 'error': 'No training examples available'})
    
    data = request.get_json()
    epochs = data.get('epochs', 100)
    
    training_active = True
    training_progress = 0
    
    def training_thread():
        global training_progress, training_active
        
        try:
            # Prepare training data
            X, y = model_trainer.prepare_training_data()
            
            # Train model
            success = model_trainer.train_neural_network(epochs=epochs)
            
            training_active = False
            training_progress = 100 if success else 0
            
            # Emit completion event
            socketio.emit('training_complete', {
                'success': success,
                'message': 'Training completed successfully' if success else 'Training failed'
            })
            
        except Exception as e:
            training_active = False
            training_progress = 0
            logger.error(f"Training error: {e}")
            socketio.emit('training_complete', {
                'success': False,
                'error': str(e)
            })
    
    # Start training in background thread
    thread = threading.Thread(target=training_thread)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Training started'
    })

@app.route('/api/get_training_progress')
def get_training_progress():
    """Get current training progress"""
    global training_progress, training_active
    
    return jsonify({
        'active': training_active,
        'progress': training_progress
    })

@app.route('/api/make_prediction', methods=['POST'])
def make_prediction():
    """Make a prediction using the trained model"""
    global model_trainer, current_stock_data
    
    if not model_trainer:
        return jsonify({'success': False, 'error': 'Model trainer not available'})
    
    if current_stock_data is None:
        return jsonify({'success': False, 'error': 'No stock data loaded'})
    
    try:
        # Check if model is trained
        model_state = model_trainer.get_model_state()
        if not model_state.get('is_trained', False):
            return jsonify({'success': False, 'error': 'Model needs to be trained first'})
        
        # Calculate features
        features = model_trainer._calculate_performance_metrics(current_stock_data)
        
        # Make prediction
        prediction = model_trainer.make_prediction(features)
        
        return jsonify({
            'success': True,
            'prediction': prediction
        })
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/get_model_status')
def get_model_status():
    """Get current model status"""
    global model_trainer
    
    if not model_trainer:
        return jsonify({
            'available': False,
            'message': 'Model trainer not available'
        })
    
    try:
        model_state = model_trainer.get_model_state()
        training_stats = model_trainer.get_training_stats()
        
        return jsonify({
            'available': True,
            'model_state': model_state,
            'training_stats': training_stats,
            'training_examples': len(model_trainer.training_examples)
        })
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return jsonify({
            'available': False,
            'error': str(e)
        })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'message': 'Connected to trading system'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

def main():
    """Main function"""
    print("=" * 80)
    print("                    NEURAL NETWORK TRADING SYSTEM - WEB INTERFACE")
    print("=" * 80)
    print()
    print(" This system trains an AI to mimic your trading decisions by combining:")
    print(" * Technical Analysis (RSI, MACD, Bollinger Bands)")
    print(" * Sentiment Analysis (VADER analysis of your reasoning)")
    print(" * Neural Network Learning (TensorFlow deep learning)")
    print()
    print(" Web interface will be available at: http://localhost:5000")
    print()
    
    # Initialize model trainer
    if initialize_model_trainer():
        print("[OK] Model trainer initialized successfully")
    else:
        print("[WARNING] Model trainer not available - some features will be limited")
    
    # Create templates directory if it doesn't exist
    templates_dir = Path(__file__).parent / 'templates'
    templates_dir.mkdir(exist_ok=True)
    
    # Run the Flask app
    print("Starting web server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main() 