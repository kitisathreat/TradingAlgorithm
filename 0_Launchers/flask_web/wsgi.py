#!/usr/bin/env python3
"""
WSGI application entry point for Trading Algorithm Flask Web Interface
This file is used by production servers like Gunicorn, uWSGI, or AWS Elastic Beanstalk
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
REPO_ROOT = Path(__file__).parent.parent.parent
ORCHESTRATOR_PATH = REPO_ROOT / "_2_Orchestrator_And_ML_Python"
sys.path.append(str(ORCHESTRATOR_PATH))

# Set environment variables for production
os.environ.setdefault('FLASK_ENV', 'production')

# Import the Flask app
from flask_app import app, socketio

# Create application factory for production
def create_app(config_name=None):
    """Application factory pattern for production deployment"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'production')
    
    # Import configuration
    from config import config
    
    # Configure the app
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    
    return app

# WSGI application object
application = create_app()

# For direct execution
if __name__ == '__main__':
    # Get configuration from environment
    config_name = os.environ.get('FLASK_ENV', 'production')
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    # Create and run the app
    app = create_app(config_name)
    
    print(f"Starting Trading Algorithm Web Interface on {host}:{port}")
    print(f"Environment: {config_name}")
    
    # Run with SocketIO for WebSocket support
    socketio.run(app, host=host, port=port, debug=False) 