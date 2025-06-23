#!/bin/bash

# Trading Algorithm Health Check Script
# Run this on your EC2 instance to verify everything is working

echo "================================================================================"
echo "                    TRADING ALGORITHM - HEALTH CHECK"
echo "================================================================================"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Determine the correct user and paths
if [ -d "/home/ec2-user" ]; then
    CURRENT_USER="ec2-user"
    USER_HOME="/home/ec2-user"
    APP_DIR="$USER_HOME/trading-algorithm"
elif [ -d "/home/ubuntu" ]; then
    CURRENT_USER="ubuntu"
    USER_HOME="/home/ubuntu"
    APP_DIR="$USER_HOME/trading-algorithm"
else
    print_error "Could not determine user (ec2-user or ubuntu)"
    exit 1
fi

print_status "Detected user: $CURRENT_USER"
print_status "App directory: $APP_DIR"
echo

# 1. Check Trading Algorithm Service
echo "1. Checking Trading Algorithm Service..."
if sudo systemctl is-active --quiet trading-algorithm; then
    print_success "Trading Algorithm service is running"
else
    print_error "Trading Algorithm service is not running"
    print_status "Recent service logs:"
    sudo journalctl -u trading-algorithm --no-pager -n 10
fi

if sudo systemctl is-enabled trading-algorithm; then
    print_success "Trading Algorithm service is enabled for boot"
else
    print_warning "Trading Algorithm service is not enabled for boot"
fi

echo

# 2. Check Nginx Service
echo "2. Checking Nginx Service..."
if sudo systemctl is-active --quiet nginx; then
    print_success "Nginx service is running"
else
    print_error "Nginx service is not running"
    print_status "Recent nginx logs:"
    sudo journalctl -u nginx --no-pager -n 5
fi

if sudo systemctl is-enabled nginx; then
    print_success "Nginx service is enabled for boot"
else
    print_warning "Nginx service is not enabled for boot"
fi

echo

# 3. Check Port Listening
echo "3. Checking Port Status..."
if sudo netstat -tlnp | grep -q ":80 "; then
    print_success "Port 80 is listening (nginx)"
else
    print_error "Port 80 is not listening"
fi

if sudo netstat -tlnp | grep -q ":8000 "; then
    print_success "Port 8000 is listening (gunicorn)"
else
    print_error "Port 8000 is not listening"
fi

echo

# 4. Check Application Health
echo "4. Testing Application Health..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000 | grep -q "200\|302"; then
    print_success "Application is responding on port 8000"
else
    print_error "Application is not responding on port 8000"
fi

if curl -s -o /dev/null -w "%{http_code}" http://localhost | grep -q "200\|302"; then
    print_success "Nginx is serving the application"
else
    print_error "Nginx is not serving the application"
fi

echo

# 5. Check Process Status
echo "5. Checking Process Status..."
GUNICORN_COUNT=$(ps aux | grep gunicorn | grep -v grep | wc -l)
if [ $GUNICORN_COUNT -gt 0 ]; then
    print_success "Gunicorn processes running: $GUNICORN_COUNT"
    ps aux | grep gunicorn | grep -v grep
else
    print_error "No gunicorn processes found"
fi

NGINX_COUNT=$(ps aux | grep nginx | grep -v grep | wc -l)
if [ $NGINX_COUNT -gt 0 ]; then
    print_success "Nginx processes running: $NGINX_COUNT"
else
    print_error "No nginx processes found"
fi

echo

# 6. Check File Structure
echo "6. Checking File Structure..."
if [ -d "$APP_DIR/flask_web/venv" ]; then
    print_success "Virtual environment exists"
    if [ -f "$APP_DIR/flask_web/venv/bin/gunicorn" ]; then
        print_success "Gunicorn executable found"
    else
        print_error "Gunicorn executable not found in venv"
    fi
else
    print_error "Virtual environment not found at $APP_DIR/flask_web/venv"
fi

if [ -f "$APP_DIR/flask_web/flask_app.py" ]; then
    print_success "Flask app found"
else
    print_error "Flask app not found"
fi

if [ -f "$APP_DIR/flask_web/wsgi.py" ]; then
    print_success "WSGI file found"
else
    print_error "WSGI file not found"
fi

echo

# 7. Check System Resources
echo "7. Checking System Resources..."
DISK_USAGE=$(df -h / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -lt 80 ]; then
    print_success "Disk usage: ${DISK_USAGE}% (healthy)"
else
    print_error "Disk usage: ${DISK_USAGE}% (high usage)"
fi

MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
print_status "Memory usage: ${MEMORY_USAGE}%"

echo

# 8. Check Recent Logs
echo "8. Checking Recent Logs..."
print_status "Recent application logs (last 5 lines):"
sudo journalctl -u trading-algorithm --no-pager -n 5

echo
print_status "Recent nginx error logs (last 5 lines):"
sudo tail -n 5 /var/log/nginx/error.log 2>/dev/null || echo "No nginx error logs found"

echo

# 9. Get Public IP
echo "9. Network Information..."
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "unknown")
print_status "Public IP: $PUBLIC_IP"
print_status "Application URL: http://$PUBLIC_IP"

echo
echo "================================================================================"
echo "Health check completed!"
echo "================================================================================"
echo
echo "If you see any errors above, here are some troubleshooting commands:"
echo "  Fix service paths: bash fix_service_paths.sh"
echo "  Restart trading algorithm: sudo systemctl restart trading-algorithm"
echo "  Restart nginx: sudo systemctl restart nginx"
echo "  View app logs: sudo journalctl -u trading-algorithm -f"
echo "  View nginx logs: sudo tail -f /var/log/nginx/error.log"
echo "  Check nginx config: sudo nginx -t"
echo "  Check service config: sudo systemctl cat trading-algorithm"
echo