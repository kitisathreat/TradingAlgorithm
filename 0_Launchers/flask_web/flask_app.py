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

# Eventlet monkey patching for async support (must be done before other imports)
try:
    import eventlet
    eventlet.monkey_patch()
    print("[INFO] Eventlet monkey patching applied successfully")
except ImportError as e:
    print(f"[WARNING] Eventlet not available: {e}")
    print("[INFO] Falling back to threading mode")

# Add parent directory to path to import from orchestrator
# Handle both local development and AWS deployment paths
REPO_ROOT = Path(__file__).parent.parent.parent
ORCHESTRATOR_PATH = REPO_ROOT / "_2_Orchestrator_And_ML_Python"

# For AWS deployment, the orchestrator might be in the current directory
if not ORCHESTRATOR_PATH.exists():
    ORCHESTRATOR_PATH = Path(__file__).parent / "_2_Orchestrator_And_ML_Python"

# Add both paths to sys.path to handle different deployment scenarios
sys.path.append(str(ORCHESTRATOR_PATH))
sys.path.append(str(REPO_ROOT / "_2_Orchestrator_And_ML_Python"))

print(f"[INFO] Looking for orchestrator at: {ORCHESTRATOR_PATH}")
print(f"[INFO] Also checking: {REPO_ROOT / '_2_Orchestrator_And_ML_Python'}")

try:
    from interactive_training_app.backend.model_trainer import ModelTrainer
    MODEL_TRAINER_AVAILABLE = True
    print("[INFO] ModelTrainer imported successfully")
except ImportError as e:
    logging.warning(f"ModelTrainer not available: {e}")
    MODEL_TRAINER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'trading_algorithm_secret_key_2024'

# Determine async mode based on eventlet availability
try:
    import eventlet
    async_mode = 'eventlet'
    print("[INFO] Using eventlet async mode")
except ImportError:
    async_mode = 'threading'
    print("[INFO] Using threading async mode")

# Enhanced SocketIO configuration for better WebSocket handling
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode=async_mode, 
    logger=True, 
    engineio_logger=True, 
    ping_timeout=60, 
    ping_interval=25,
    max_http_buffer_size=1e8,  # 100MB buffer for large messages
    allow_upgrades=True,  # Allow WebSocket upgrades
    transports=['polling', 'websocket'],  # Support both transports
    always_connect=True,  # Always try to connect
    reconnection=True,  # Enable reconnection
    reconnection_attempts=5,
    reconnection_delay=1000,
    reconnection_delay_max=5000
)

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
    print("[INFO] StockSelectionManager imported successfully")
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
    """Calculate comprehensive technical indicators matching the local GUI"""
    if data is None or data.empty:
        return {}
    
    try:
        # Price metrics
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
        
        high_price = data['High'].max()
        low_price = data['Low'].min()
        avg_price = data['Close'].mean()
        
        # Volume metrics
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # Volatility metrics
        daily_returns = data['Close'].pct_change().dropna()
        daily_vol = daily_returns.std() * 100
        weekly_vol = daily_returns.rolling(5).std().iloc[-1] * 100 if len(daily_returns) >= 5 else daily_vol
        
        # ATR calculation
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(14).mean().iloc[-1] if len(data) >= 14 else true_range.mean()
        
        # Bollinger Bands
        sma_20 = data['Close'].rolling(20).mean()
        std_20 = data['Close'].rolling(20).std()
        bollinger_width = (std_20.iloc[-1] / sma_20.iloc[-1]) * 100 if not pd.isna(sma_20.iloc[-1]) else 0
        bb_upper = sma_20 + (std_20 * 2)
        bb_lower = sma_20 - (std_20 * 2)
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        # MACD
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        current_macd = macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0
        current_macd_signal = macd_signal.iloc[-1] if not pd.isna(macd_signal.iloc[-1]) else 0
        macd_histogram = current_macd - current_macd_signal
        
        # Moving averages
        sma_20_current = sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else current_price
        ema_12_current = ema_12.iloc[-1] if not pd.isna(ema_12.iloc[-1]) else current_price
        
        # Additional moving averages
        sma_50 = data['Close'].rolling(50).mean()
        sma_200 = data['Close'].rolling(200).mean()
        ema_20 = data['Close'].ewm(span=20).mean()
        
        sma_50_current = sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else current_price
        sma_200_current = sma_200.iloc[-1] if not pd.isna(sma_200.iloc[-1]) else current_price
        ema_20_current = ema_20.iloc[-1] if not pd.isna(ema_20.iloc[-1]) else current_price
        
        # Bollinger Bands
        bb_upper_current = bb_upper.iloc[-1] if not pd.isna(bb_upper.iloc[-1]) else current_price
        bb_lower_current = bb_lower.iloc[-1] if not pd.isna(bb_lower.iloc[-1]) else current_price
        bb_middle_current = sma_20_current
        
        # Stochastic Oscillator
        low_14 = data['Low'].rolling(14).min()
        high_14 = data['High'].rolling(14).max()
        k_percent = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
        d_percent = k_percent.rolling(3).mean()
        
        stoch_k = k_percent.iloc[-1] if not pd.isna(k_percent.iloc[-1]) else 50
        stoch_d = d_percent.iloc[-1] if not pd.isna(d_percent.iloc[-1]) else 50
        
        # Williams %R
        williams_r = -100 * ((high_14 - data['Close']) / (high_14 - low_14))
        williams_r_current = williams_r.iloc[-1] if not pd.isna(williams_r.iloc[-1]) else -50
        
        # Price change over period
        price_change_period = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
        
        # Volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(252)
        
        return {
            # Price metrics
            'current_price': float(current_price),
            'price_change': float(price_change),
            'price_change_pct': float(price_change_pct),
            'price_change_period': float(price_change_period),
            'high_price': float(high_price),
            'low_price': float(low_price),
            'avg_price': float(avg_price),
            
            # Volume metrics
            'current_volume': int(current_volume),
            'avg_volume': int(avg_volume),
            'volume_ratio': float(volume_ratio),
            
            # Volatility metrics
            'daily_vol': float(daily_vol),
            'weekly_vol': float(weekly_vol),
            'volatility': float(volatility),
            'atr': float(atr),
            'bollinger_width': float(bollinger_width),
            
            # Technical indicators
            'rsi': float(current_rsi),
            'macd': float(current_macd),
            'macd_signal': float(current_macd_signal),
            'macd_histogram': float(macd_histogram),
            'sma_20': float(sma_20_current),
            'ema_12': float(ema_12_current),
            'sma_50': float(sma_50_current),
            'sma_200': float(sma_200_current),
            'ema_20': float(ema_20_current),
            
            # Bollinger Bands
            'bb_upper': float(bb_upper_current),
            'bb_lower': float(bb_lower_current),
            'bb_middle': float(bb_middle_current),
            
            # Stochastic
            'stoch_k': float(stoch_k),
            'stoch_d': float(stoch_d),
            'williams_r': float(williams_r_current)
        }
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
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
            # Try multiple possible paths for the S&P100 data file
            sp100_paths = [
                Path(__file__).parent.parent.parent / "_2_Orchestrator_And_ML_Python" / "sp100_symbols_with_market_cap.json",
                Path(__file__).parent / "_2_Orchestrator_And_ML_Python" / "sp100_symbols_with_market_cap.json",
                Path("/var/app/current/_2_Orchestrator_And_ML_Python/sp100_symbols_with_market_cap.json"),
                Path("/var/_2_Orchestrator_And_ML_Python/sp100_symbols_with_market_cap.json")
            ]
            
            sp100_file = None
            for path in sp100_paths:
                if path.exists():
                    sp100_file = path
                    break
            
            if sp100_file is None:
                raise FileNotFoundError("S&P100 data file not found in any expected location")
            
            logger.info(f"Loading S&P100 data from: {sp100_file}")
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
    
    # Emit initial training status
    emit_training_progress_update()
    
    def training_thread():
        global training_progress, training_active
        
        try:
            # Prepare training data
            X, y = model_trainer.prepare_training_data()
            
            # Train model with progress updates
            success = model_trainer.train_neural_network(epochs=epochs)
            
            training_active = False
            training_progress = 100 if success else 0
            
            # Emit final progress update
            emit_training_progress_update()
            
            # Emit completion event
            emit_training_complete(success)
            
        except Exception as e:
            training_active = False
            training_progress = 0
            logger.error(f"Training error: {e}")
            
            # Emit final progress update
            emit_training_progress_update()
            
            # Emit completion event with error
            emit_training_complete(False, str(e))
    
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
    try:
        client_id = request.sid
        logger.info(f"Client connected: {client_id}")
        
        # Send connection confirmation with detailed status
        emit('connected', {
            'message': 'Connected to trading system',
            'model_available': model_trainer is not None,
            'timestamp': datetime.now().isoformat(),
            'client_id': client_id,
            'system_status': {
                'model_trainer_available': model_trainer is not None,
                'training_active': training_active,
                'training_progress': training_progress,
                'current_stock': current_symbol
            }
        })
        
        # Log connection details
        logger.info(f"Connection established - Client ID: {client_id}, Model Available: {model_trainer is not None}")
        
    except Exception as e:
        logger.error(f"Error in connect handler: {e}")
        # Try to emit error to client
        try:
            emit('error', {
                'message': 'Connection error occurred',
                'error': str(e)
            })
        except:
            pass

@socketio.on('disconnect')
def handle_disconnect(data=None):
    """Handle client disconnection"""
    try:
        client_id = request.sid
        logger.info(f"Client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"Error in disconnect handler: {e}")

@socketio.on_error()
def error_handler(e):
    """Handle SocketIO errors"""
    try:
        client_id = request.sid if hasattr(request, 'sid') else 'unknown'
        logger.error(f"SocketIO error for client {client_id}: {e}")
        
        # Try to emit error to client
        try:
            emit('error', {
                'message': 'An error occurred in the connection',
                'error': str(e)
            })
        except:
            pass
            
    except Exception as error:
        logger.error(f"Error in error handler: {error}")

@socketio.on('ping')
def handle_ping():
    """Handle ping from client"""
    try:
        emit('pong', {
            'timestamp': datetime.now().isoformat(),
            'server_status': 'healthy',
            'model_available': model_trainer is not None
        })
    except Exception as e:
        logger.error(f"Error handling ping: {e}")

@socketio.on('get_status')
def handle_get_status():
    """Handle status request from client"""
    try:
        emit('status_update', {
            'model_available': model_trainer is not None,
            'training_active': training_active,
            'training_progress': training_progress,
            'current_stock': current_symbol,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error handling status request: {e}")
        emit('error', {'message': 'Failed to get status'})

@socketio.on('request_stock_data')
def handle_stock_data_request(data):
    """Handle real-time stock data request"""
    try:
        symbol = data.get('symbol')
        if not symbol:
            emit('error', {'message': 'Symbol is required'})
            return
        
        # Get stock data
        stock_info, stock_data = get_stock_data(symbol, days=30)
        
        if stock_info and stock_data is not None:
            # Calculate indicators
            indicators = calculate_technical_indicators(stock_data)
            
            # Prepare chart data
            chart_data = {
                'dates': stock_data.index.strftime('%Y-%m-%d').tolist(),
                'prices': stock_data['Close'].tolist(),
                'volumes': stock_data['Volume'].tolist(),
                'opens': stock_data['Open'].tolist(),
                'highs': stock_data['High'].tolist(),
                'lows': stock_data['Low'].tolist()
            }
            
            emit('stock_data_update', {
                'symbol': symbol,
                'stock_info': stock_info,
                'chart_data': chart_data,
                'indicators': indicators,
                'timestamp': datetime.now().isoformat()
            })
        else:
            emit('error', {'message': f'Failed to fetch data for {symbol}'})
            
    except Exception as e:
        logger.error(f"Error handling stock data request: {e}")
        emit('error', {'message': f'Error fetching stock data: {str(e)}'})

@socketio.on('request_prediction')
def handle_prediction_request(data):
    """Handle real-time prediction request"""
    try:
        if not model_trainer:
            emit('error', {'message': 'Model trainer not available'})
            return
        
        # Check if model is trained
        model_state = model_trainer.get_model_state()
        if not model_state.get('is_trained', False):
            emit('error', {'message': 'Model needs to be trained first'})
            return
        
        # Get stock data for prediction
        symbol = data.get('symbol')
        if not symbol:
            emit('error', {'message': 'Symbol is required'})
            return
        
        stock_info, stock_data = get_stock_data(symbol, days=30)
        if stock_data is None:
            emit('error', {'message': f'Failed to fetch data for {symbol}'})
            return
        
        # Calculate features
        features = model_trainer._calculate_performance_metrics(stock_data)
        
        # Make prediction
        prediction = model_trainer.make_prediction(features)
        
        emit('prediction_result', {
            'symbol': symbol,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error handling prediction request: {e}")
        emit('error', {'message': f'Error making prediction: {str(e)}'})

@socketio.on('training_progress_request')
def handle_training_progress_request():
    """Handle training progress request"""
    try:
        emit('training_progress_update', {
            'active': training_active,
            'progress': training_progress,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error handling training progress request: {e}")
        emit('error', {'message': 'Failed to get training progress'})

def emit_training_progress_update():
    """Emit training progress update to all connected clients"""
    try:
        socketio.emit('training_progress_update', {
            'active': training_active,
            'progress': training_progress,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error emitting training progress: {e}")

def emit_training_complete(success, error=None):
    """Emit training completion to all connected clients"""
    try:
        socketio.emit('training_complete', {
            'success': success,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error emitting training complete: {e}")

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
    print("Starting web server with SocketIO support...")
    print("SocketIO async mode:", socketio.async_mode)
    print("SocketIO engine:", socketio.server.eio.async_mode)
    
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

# Eventlet monkey patching for async support (must be done before other imports)
try:
    import eventlet
    eventlet.monkey_patch()
    print("[INFO] Eventlet monkey patching applied successfully")
except ImportError as e:
    print(f"[WARNING] Eventlet not available: {e}")
    print("[INFO] Falling back to threading mode")

# Add parent directory to path to import from orchestrator
# Handle both local development and AWS deployment paths
REPO_ROOT = Path(__file__).parent.parent.parent
ORCHESTRATOR_PATH = REPO_ROOT / "_2_Orchestrator_And_ML_Python"

# For AWS deployment, the orchestrator might be in the current directory
if not ORCHESTRATOR_PATH.exists():
    ORCHESTRATOR_PATH = Path(__file__).parent / "_2_Orchestrator_And_ML_Python"

# Add both paths to sys.path to handle different deployment scenarios
sys.path.append(str(ORCHESTRATOR_PATH))
sys.path.append(str(REPO_ROOT / "_2_Orchestrator_And_ML_Python"))

print(f"[INFO] Looking for orchestrator at: {ORCHESTRATOR_PATH}")
print(f"[INFO] Also checking: {REPO_ROOT / '_2_Orchestrator_And_ML_Python'}")

try:
    from interactive_training_app.backend.model_trainer import ModelTrainer
    MODEL_TRAINER_AVAILABLE = True
    print("[INFO] ModelTrainer imported successfully")
except ImportError as e:
    logging.warning(f"ModelTrainer not available: {e}")
    MODEL_TRAINER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'trading_algorithm_secret_key_2024'

# Determine async mode based on eventlet availability
try:
    import eventlet
    async_mode = 'eventlet'
    print("[INFO] Using eventlet async mode")
except ImportError:
    async_mode = 'threading'
    print("[INFO] Using threading async mode")

# Enhanced SocketIO configuration for better WebSocket handling
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode=async_mode, 
    logger=True, 
    engineio_logger=True, 
    ping_timeout=60, 
    ping_interval=25,
    max_http_buffer_size=1e8,  # 100MB buffer for large messages
    allow_upgrades=True,  # Allow WebSocket upgrades
    transports=['polling', 'websocket'],  # Support both transports
    always_connect=True,  # Always try to connect
    reconnection=True,  # Enable reconnection
    reconnection_attempts=5,
    reconnection_delay=1000,
    reconnection_delay_max=5000
)

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
    print("[INFO] StockSelectionManager imported successfully")
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
    """Calculate comprehensive technical indicators matching the local GUI"""
    if data is None or data.empty:
        return {}
    
    try:
        # Price metrics
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
        
        high_price = data['High'].max()
        low_price = data['Low'].min()
        avg_price = data['Close'].mean()
        
        # Volume metrics
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # Volatility metrics
        daily_returns = data['Close'].pct_change().dropna()
        daily_vol = daily_returns.std() * 100
        weekly_vol = daily_returns.rolling(5).std().iloc[-1] * 100 if len(daily_returns) >= 5 else daily_vol
        
        # ATR calculation
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(14).mean().iloc[-1] if len(data) >= 14 else true_range.mean()
        
        # Bollinger Bands
        sma_20 = data['Close'].rolling(20).mean()
        std_20 = data['Close'].rolling(20).std()
        bollinger_width = (std_20.iloc[-1] / sma_20.iloc[-1]) * 100 if not pd.isna(sma_20.iloc[-1]) else 0
        bb_upper = sma_20 + (std_20 * 2)
        bb_lower = sma_20 - (std_20 * 2)
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        # MACD
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        current_macd = macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0
        current_macd_signal = macd_signal.iloc[-1] if not pd.isna(macd_signal.iloc[-1]) else 0
        macd_histogram = current_macd - current_macd_signal
        
        # Moving averages
        sma_20_current = sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else current_price
        ema_12_current = ema_12.iloc[-1] if not pd.isna(ema_12.iloc[-1]) else current_price
        
        # Additional moving averages
        sma_50 = data['Close'].rolling(50).mean()
        sma_200 = data['Close'].rolling(200).mean()
        ema_20 = data['Close'].ewm(span=20).mean()
        
        sma_50_current = sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else current_price
        sma_200_current = sma_200.iloc[-1] if not pd.isna(sma_200.iloc[-1]) else current_price
        ema_20_current = ema_20.iloc[-1] if not pd.isna(ema_20.iloc[-1]) else current_price
        
        # Bollinger Bands
        bb_upper_current = bb_upper.iloc[-1] if not pd.isna(bb_upper.iloc[-1]) else current_price
        bb_lower_current = bb_lower.iloc[-1] if not pd.isna(bb_lower.iloc[-1]) else current_price
        bb_middle_current = sma_20_current
        
        # Stochastic Oscillator
        low_14 = data['Low'].rolling(14).min()
        high_14 = data['High'].rolling(14).max()
        k_percent = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
        d_percent = k_percent.rolling(3).mean()
        
        stoch_k = k_percent.iloc[-1] if not pd.isna(k_percent.iloc[-1]) else 50
        stoch_d = d_percent.iloc[-1] if not pd.isna(d_percent.iloc[-1]) else 50
        
        # Williams %R
        williams_r = -100 * ((high_14 - data['Close']) / (high_14 - low_14))
        williams_r_current = williams_r.iloc[-1] if not pd.isna(williams_r.iloc[-1]) else -50
        
        # Price change over period
        price_change_period = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
        
        # Volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(252)
        
        return {
            # Price metrics
            'current_price': float(current_price),
            'price_change': float(price_change),
            'price_change_pct': float(price_change_pct),
            'price_change_period': float(price_change_period),
            'high_price': float(high_price),
            'low_price': float(low_price),
            'avg_price': float(avg_price),
            
            # Volume metrics
            'current_volume': int(current_volume),
            'avg_volume': int(avg_volume),
            'volume_ratio': float(volume_ratio),
            
            # Volatility metrics
            'daily_vol': float(daily_vol),
            'weekly_vol': float(weekly_vol),
            'volatility': float(volatility),
            'atr': float(atr),
            'bollinger_width': float(bollinger_width),
            
            # Technical indicators
            'rsi': float(current_rsi),
            'macd': float(current_macd),
            'macd_signal': float(current_macd_signal),
            'macd_histogram': float(macd_histogram),
            'sma_20': float(sma_20_current),
            'ema_12': float(ema_12_current),
            'sma_50': float(sma_50_current),
            'sma_200': float(sma_200_current),
            'ema_20': float(ema_20_current),
            
            # Bollinger Bands
            'bb_upper': float(bb_upper_current),
            'bb_lower': float(bb_lower_current),
            'bb_middle': float(bb_middle_current),
            
            # Stochastic
            'stoch_k': float(stoch_k),
            'stoch_d': float(stoch_d),
            'williams_r': float(williams_r_current)
        }
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
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
            # Try multiple possible paths for the S&P100 data file
            sp100_paths = [
                Path(__file__).parent.parent.parent / "_2_Orchestrator_And_ML_Python" / "sp100_symbols_with_market_cap.json",
                Path(__file__).parent / "_2_Orchestrator_And_ML_Python" / "sp100_symbols_with_market_cap.json",
                Path("/var/app/current/_2_Orchestrator_And_ML_Python/sp100_symbols_with_market_cap.json"),
                Path("/var/_2_Orchestrator_And_ML_Python/sp100_symbols_with_market_cap.json")
            ]
            
            sp100_file = None
            for path in sp100_paths:
                if path.exists():
                    sp100_file = path
                    break
            
            if sp100_file is None:
                raise FileNotFoundError("S&P100 data file not found in any expected location")
            
            logger.info(f"Loading S&P100 data from: {sp100_file}")
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
    
    # Emit initial training status
    emit_training_progress_update()
    
    def training_thread():
        global training_progress, training_active
        
        try:
            # Prepare training data
            X, y = model_trainer.prepare_training_data()
            
            # Train model with progress updates
            success = model_trainer.train_neural_network(epochs=epochs)
            
            training_active = False
            training_progress = 100 if success else 0
            
            # Emit final progress update
            emit_training_progress_update()
            
            # Emit completion event
            emit_training_complete(success)
            
        except Exception as e:
            training_active = False
            training_progress = 0
            logger.error(f"Training error: {e}")
            
            # Emit final progress update
            emit_training_progress_update()
            
            # Emit completion event with error
            emit_training_complete(False, str(e))
    
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
    try:
        client_id = request.sid
        logger.info(f"Client connected: {client_id}")
        
        # Send connection confirmation with detailed status
        emit('connected', {
            'message': 'Connected to trading system',
            'model_available': model_trainer is not None,
            'timestamp': datetime.now().isoformat(),
            'client_id': client_id,
            'system_status': {
                'model_trainer_available': model_trainer is not None,
                'training_active': training_active,
                'training_progress': training_progress,
                'current_stock': current_symbol
            }
        })
        
        # Log connection details
        logger.info(f"Connection established - Client ID: {client_id}, Model Available: {model_trainer is not None}")
        
    except Exception as e:
        logger.error(f"Error in connect handler: {e}")
        # Try to emit error to client
        try:
            emit('error', {
                'message': 'Connection error occurred',
                'error': str(e)
            })
        except:
            pass

@socketio.on('disconnect')
def handle_disconnect(data=None):
    """Handle client disconnection"""
    try:
        client_id = request.sid
        logger.info(f"Client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"Error in disconnect handler: {e}")

@socketio.on_error()
def error_handler(e):
    """Handle SocketIO errors"""
    try:
        client_id = request.sid if hasattr(request, 'sid') else 'unknown'
        logger.error(f"SocketIO error for client {client_id}: {e}")
        
        # Try to emit error to client
        try:
            emit('error', {
                'message': 'An error occurred in the connection',
                'error': str(e)
            })
        except:
            pass
            
    except Exception as error:
        logger.error(f"Error in error handler: {error}")

@socketio.on('ping')
def handle_ping():
    """Handle ping from client"""
    try:
        emit('pong', {
            'timestamp': datetime.now().isoformat(),
            'server_status': 'healthy',
            'model_available': model_trainer is not None
        })
    except Exception as e:
        logger.error(f"Error handling ping: {e}")

@socketio.on('get_status')
def handle_get_status():
    """Handle status request from client"""
    try:
        emit('status_update', {
            'model_available': model_trainer is not None,
            'training_active': training_active,
            'training_progress': training_progress,
            'current_stock': current_symbol,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error handling status request: {e}")
        emit('error', {'message': 'Failed to get status'})

@socketio.on('request_stock_data')
def handle_stock_data_request(data):
    """Handle real-time stock data request"""
    try:
        symbol = data.get('symbol')
        if not symbol:
            emit('error', {'message': 'Symbol is required'})
            return
        
        # Get stock data
        stock_info, stock_data = get_stock_data(symbol, days=30)
        
        if stock_info and stock_data is not None:
            # Calculate indicators
            indicators = calculate_technical_indicators(stock_data)
            
            # Prepare chart data
            chart_data = {
                'dates': stock_data.index.strftime('%Y-%m-%d').tolist(),
                'prices': stock_data['Close'].tolist(),
                'volumes': stock_data['Volume'].tolist(),
                'opens': stock_data['Open'].tolist(),
                'highs': stock_data['High'].tolist(),
                'lows': stock_data['Low'].tolist()
            }
            
            emit('stock_data_update', {
                'symbol': symbol,
                'stock_info': stock_info,
                'chart_data': chart_data,
                'indicators': indicators,
                'timestamp': datetime.now().isoformat()
            })
        else:
            emit('error', {'message': f'Failed to fetch data for {symbol}'})
            
    except Exception as e:
        logger.error(f"Error handling stock data request: {e}")
        emit('error', {'message': f'Error fetching stock data: {str(e)}'})

@socketio.on('request_prediction')
def handle_prediction_request(data):
    """Handle real-time prediction request"""
    try:
        if not model_trainer:
            emit('error', {'message': 'Model trainer not available'})
            return
        
        # Check if model is trained
        model_state = model_trainer.get_model_state()
        if not model_state.get('is_trained', False):
            emit('error', {'message': 'Model needs to be trained first'})
            return
        
        # Get stock data for prediction
        symbol = data.get('symbol')
        if not symbol:
            emit('error', {'message': 'Symbol is required'})
            return
        
        stock_info, stock_data = get_stock_data(symbol, days=30)
        if stock_data is None:
            emit('error', {'message': f'Failed to fetch data for {symbol}'})
            return
        
        # Calculate features
        features = model_trainer._calculate_performance_metrics(stock_data)
        
        # Make prediction
        prediction = model_trainer.make_prediction(features)
        
        emit('prediction_result', {
            'symbol': symbol,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error handling prediction request: {e}")
        emit('error', {'message': f'Error making prediction: {str(e)}'})

@socketio.on('training_progress_request')
def handle_training_progress_request():
    """Handle training progress request"""
    try:
        emit('training_progress_update', {
            'active': training_active,
            'progress': training_progress,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error handling training progress request: {e}")
        emit('error', {'message': 'Failed to get training progress'})

def emit_training_progress_update():
    """Emit training progress update to all connected clients"""
    try:
        socketio.emit('training_progress_update', {
            'active': training_active,
            'progress': training_progress,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error emitting training progress: {e}")

def emit_training_complete(success, error=None):
    """Emit training completion to all connected clients"""
    try:
        socketio.emit('training_complete', {
            'success': success,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error emitting training complete: {e}")

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
    print("Starting web server with SocketIO support...")
    print("SocketIO async mode:", socketio.async_mode)
    print("SocketIO engine:", socketio.server.eio.async_mode)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    main() 