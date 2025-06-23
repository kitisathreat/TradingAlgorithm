#!/bin/bash
# FLASK APP STATUS CHECK - EC2 INSTANCE

echo "=== SERVICE STATUS ==="
sudo systemctl status trading-algorithm --no-pager

echo
echo "=== RECENT LOGS (last 20 lines) ==="
sudo journalctl -u trading-algorithm --no-pager -n 20

echo
echo "=== APPLICATION LOGS ==="
if [ -d "/home/ec2-user/trading-algorithm/flask_web/logs" ]; then
    echo "Log files found:"
    ls -la /home/ec2-user/trading-algorithm/flask_web/logs/
    echo
    echo "Recent log content:"
    find /home/ec2-user/trading-algorithm/flask_web/logs/ -name "*.log" -exec tail -10 {} \;
else
    echo "No application logs directory found"
fi

echo
echo "=== TESTING APP ENDPOINTS ==="
echo "Testing if app responds on port 8000..."
if curl -s --connect-timeout 5 "http://localhost:8000/api/get_stock_options" > /dev/null 2>&1; then
    echo "✓ App is responding on port 8000"
    echo "Testing stock options endpoint..."
    curl -s "http://localhost:8000/api/get_stock_options" | head -5
else
    echo "✗ App is not responding on port 8000"
fi

echo
echo "=== RESTARTING SERVICE ==="
echo "Restarting trading-algorithm service..."
sudo systemctl restart trading-algorithm
sleep 3

echo
echo "=== SERVICE STATUS AFTER RESTART ==="
sudo systemctl status trading-algorithm --no-pager

echo
echo "=== TESTING APP AFTER RESTART ==="
sleep 2
if curl -s --connect-timeout 5 "http://localhost:8000/api/get_stock_options" > /dev/null 2>&1; then
    echo "✓ App is responding after restart"
else
    echo "✗ App is still not responding after restart"
fi

echo
echo "=== TESTING STOCK DATA ENDPOINT ==="
echo "Testing stock data loading endpoint..."
curl -X POST -H "Content-Type: application/json" -d '{"symbol":"AAPL","days":5}' http://localhost:8000/api/load_stock 