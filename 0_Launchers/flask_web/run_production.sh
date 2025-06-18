#!/bin/bash

# Neural Network Trading System - Production Server Launcher
# This script runs the Flask web interface in production mode

set -e

echo "================================================================================"
echo "                    NEURAL NETWORK TRADING SYSTEM - PRODUCTION SERVER"
echo "================================================================================"
echo ""
echo " This system trains an AI to mimic your trading decisions by combining:"
echo " * Technical Analysis (RSI, MACD, Bollinger Bands)"
echo " * Sentiment Analysis (VADER analysis of your reasoning)"
echo " * Neural Network Learning (TensorFlow deep learning)"
echo ""
echo " Production server will be available at: http://0.0.0.0:5000"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Set production environment variables
export FLASK_ENV=production
export SECRET_KEY=trading_algorithm_production_secret_2024
export LOG_LEVEL=INFO
export CORS_ORIGINS="*"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
print_status "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found in PATH"
    echo "Please install Python 3.9 or later"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_status "Found Python $PYTHON_VERSION"

# Check if virtual environment exists
print_status "Setting up virtual environment..."
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    print_status "Found existing virtual environment"
    source venv/bin/activate
else
    print_status "Creating new virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    print_status "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Check if Gunicorn is available
print_status "Checking production dependencies..."
if ! python -c "import gunicorn" 2>/dev/null; then
    print_status "Installing production dependencies..."
    pip install gunicorn eventlet
fi

# Create necessary directories
print_status "Setting up directories..."
mkdir -p logs models cache

# Check if port 5000 is available
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_warning "Port 5000 is already in use"
    print_status "Attempting to use port 5001 instead..."
    export PORT=5001
else
    export PORT=5000
fi

# Launch production server
echo ""
echo "================================================================================"
echo "                    STARTING PRODUCTION SERVER"
echo "================================================================================"
echo ""
echo " Server will be available at: http://0.0.0.0:$PORT"
echo " Environment: Production"
echo " Press Ctrl+C to stop the server"
echo ""
echo "================================================================================"
echo ""

# Run with Gunicorn for production
exec gunicorn --config gunicorn.conf.py wsgi:application 