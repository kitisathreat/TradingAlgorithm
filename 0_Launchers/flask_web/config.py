"""
Configuration settings for Flask Trading Algorithm Web Interface
Supports both development and production environments
"""

import os
from pathlib import Path

class Config:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'trading_algorithm_secret_key_2024_production'
    DEBUG = False
    TESTING = False
    
    # Flask-SocketIO settings
    SOCKETIO_ASYNC_MODE = 'eventlet'
    
    # Security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # CORS settings for production
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'flask_app.log')
    
    # Model settings
    MODEL_SAVE_PATH = os.environ.get('MODEL_SAVE_PATH', '/tmp/trading_model')
    
    # Data settings
    DATA_CACHE_DIR = os.environ.get('DATA_CACHE_DIR', '/tmp/stock_data')
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        # Create necessary directories
        Path(Config.MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)
        Path(Config.DATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        import logging
        from logging.handlers import RotatingFileHandler
        
        if not app.debug and not app.testing:
            if not os.path.exists('logs'):
                os.mkdir('logs')
            file_handler = RotatingFileHandler(
                Config.LOG_FILE, 
                maxBytes=10240000, 
                backupCount=10
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)
            app.logger.setLevel(logging.INFO)
            app.logger.info('Trading Algorithm startup')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SESSION_COOKIE_SECURE = False
    
    # Development-specific settings
    SOCKETIO_ASYNC_MODE = 'threading'
    
    # Local development paths
    MODEL_SAVE_PATH = './models'
    DATA_CACHE_DIR = './cache'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    
    # Production security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    
    # Production logging
    LOG_LEVEL = 'WARNING'
    
    # Use environment variables for sensitive settings
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY environment variable is required for production")

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    WTF_CSRF_ENABLED = False
    
    # Test-specific settings
    MODEL_SAVE_PATH = './test_models'
    DATA_CACHE_DIR = './test_cache'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 