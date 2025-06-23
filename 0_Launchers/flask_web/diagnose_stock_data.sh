#!/bin/bash
# STOCK DATA LOADING DIAGNOSTIC - EC2 INSTANCE

set -e

echo "================================================================================"
echo "              STOCK DATA LOADING DIAGNOSTIC"
echo "================================================================================"
echo "Started at: $(date)"
echo "================================================================================"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Find application directory
APP_DIR=""
POSSIBLE_DIRS=(
    "/home/ec2-user/trading-algorithm"
    "/home/ubuntu/trading-algorithm"
    "$HOME/trading-algorithm"
    "$HOME/flask_web"
)

for dir in "${POSSIBLE_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        APP_DIR="$dir"
        print_success "Found application directory: $dir"
        break
    fi
done

if [ -z "$APP_DIR" ]; then
    print_error "No application directory found!"
    exit 1
fi

# Check if virtual environment exists and activate it
if [ -d "$APP_DIR/flask_web/venv" ]; then
    print_success "Virtual environment found"
    source "$APP_DIR/flask_web/venv/bin/activate"
else
    print_error "Virtual environment not found at $APP_DIR/flask_web/venv"
    exit 1
fi

echo
print_status "Testing Python imports and yfinance functionality..."

# Test basic Python imports
python -c "import yfinance; print('yfinance version:', yfinance.__version__)" 2>/dev/null
if [ $? -eq 0 ]; then
    print_success "yfinance import successful"
else
    print_error "yfinance import failed"
fi

python -c "import pandas; print('pandas version:', pandas.__version__)" 2>/dev/null
if [ $? -eq 0 ]; then
    print_success "pandas import successful"
else
    print_error "pandas import failed"
fi

python -c "import numpy; print('numpy version:', numpy.__version__)" 2>/dev/null
if [ $? -eq 0 ]; then
    print_success "numpy import successful"
else
    print_error "numpy import failed"
fi

echo
print_status "Testing yfinance data fetching..."

# Test basic yfinance functionality
python3 - <<EOF
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

print('Testing yfinance data fetch...')
try:
    # Test with a simple stock
    ticker = yf.Ticker('AAPL')
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    
    print(f'Fetching AAPL data from {start_date.date()} to {end_date.date()}...')
    data = ticker.history(start=start_date, end=end_date)
    
    if data.empty:
        print('ERROR: No data returned')
    else:
        print(f'SUCCESS: Retrieved {len(data)} rows of data')
        print(f'Columns: {list(data.columns)}')
        print(f'Date range: {data.index[0].date()} to {data.index[-1].date()}')
        print(f'Latest close: ${data["Close"].iloc[-1]:.2f}')
        
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
EOF

echo
print_status "Testing network connectivity to Yahoo Finance..."

# Test network connectivity
curl -s --connect-timeout 10 --max-time 30 "https://query1.finance.yahoo.com/v8/finance/chart/AAPL?period1=1640995200&period2=1641081600&interval=1d" > /dev/null
if [ $? -eq 0 ]; then
    print_success "Network connectivity to Yahoo Finance successful"
else
    print_error "Network connectivity to Yahoo Finance failed"
fi

echo
print_status "Testing rate limiter functionality..."

# Test rate limiter if available
cd "$APP_DIR"
if [ -f "flask_web/rate_limiter.py" ]; then
    print_success "Rate limiter module found"
    
    python3 - <<EOF
import sys
sys.path.append('flask_web')
try:
    from rate_limiter import EnhancedStockDataFetcher
    print('Rate limiter import successful')
    
    fetcher = EnhancedStockDataFetcher()
    print('EnhancedStockDataFetcher created successfully')
    
    # Test stock data fetch
    stock_info, data = fetcher.get_stock_data('AAPL', 5)
    if stock_info and data is not None and not data.empty:
        print(f'SUCCESS: Rate-limited fetch returned {len(data)} rows')
        print(f'Stock info: {stock_info}')
    else:
        print('ERROR: Rate-limited fetch returned no data')
        
except Exception as e:
    print(f'ERROR testing rate limiter: {e}')
    import traceback
    traceback.print_exc()
EOF
else
    print_warning "Rate limiter module not found"
fi

echo
print_status "Testing Flask app stock data loading..."

# Test Flask app stock loading functionality
if [ -f "flask_web/flask_app.py" ]; then
    print_success "Flask app found"
    
    python3 - <<EOF
import sys
import os
sys.path.append('flask_web')
sys.path.append('_2_Orchestrator_And_ML_Python')

try:
    from flask_app import get_stock_data
    print('Flask app import successful')
    
    # Test stock data function
    stock_info, data = get_stock_data('AAPL', 5)
    if stock_info and data is not None and not data.empty:
        print(f'SUCCESS: Flask get_stock_data returned {len(data)} rows')
        print(f'Stock info: {stock_info}')
    else:
        print('ERROR: Flask get_stock_data returned no data')
        
except Exception as e:
    print(f'ERROR testing Flask app: {e}')
    import traceback
    traceback.print_exc()
EOF
else
    print_error "Flask app not found"
fi

echo
print_status "Checking application logs..."

# Check recent application logs
if [ -d "flask_web/logs" ]; then
    print_success "Logs directory found"
    echo "Recent log files:"
    ls -la flask_web/logs/ 2>/dev/null | head -5
else
    print_warning "Logs directory not found"
fi

# Check system logs
echo
echo "Recent system logs for trading-algorithm service:"
sudo journalctl -u trading-algorithm --no-pager -n 10 2>/dev/null | grep -E "(ERROR|WARNING|stock|data|yfinance)" || echo "No relevant logs found"

echo
print_status "Testing application endpoints..."

# Test if the application is responding
if curl -s --connect-timeout 5 "http://localhost:8000/api/get_stock_options" > /dev/null 2>&1; then
    print_success "Application is responding on port 8000"
    
    # Test stock options endpoint
    echo "Testing stock options endpoint..."
    curl -s "http://localhost:8000/api/get_stock_options" | python3 -m json.tool 2>/dev/null || echo "Response not valid JSON"
else
    print_error "Application is not responding on port 8000"
fi

echo
print_status "Checking system resources..."

# Check system resources
echo "Memory usage:"
free -h

echo
echo "Disk space:"
df -h /

echo
echo "CPU usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1

echo
print_status "DIAGNOSTIC SUMMARY"
echo "================================================================================"

echo "If stock data is not loading, check the following:"
echo "1. Network connectivity to Yahoo Finance"
echo "2. yfinance package installation and version"
echo "3. Rate limiting configuration"
echo "4. Application logs for specific errors"
echo "5. System resources (memory, disk space)"
echo "6. Firewall settings blocking outbound connections"
echo
echo "Common solutions:"
echo "- Restart the trading-algorithm service: sudo systemctl restart trading-algorithm"
echo "- Clear yfinance cache: curl -X POST http://localhost:8000/api/clear_yfinance_cache"
echo "- Reset rate limits: curl -X POST http://localhost:8000/api/reset_rate_limits"
echo "- Check service logs: sudo journalctl -u trading-algorithm -f"
echo
echo "================================================================================" 