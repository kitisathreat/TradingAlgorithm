#!/bin/bash

# Fix Trading Algorithm Service Paths
# Run this on your EC2 instance to fix the service configuration

echo "================================================================================"
echo "                    FIXING TRADING ALGORITHM SERVICE"
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

# Stop the current service
print_status "Stopping current service..."
sudo systemctl stop trading-algorithm

# Create corrected systemd service file
print_status "Creating corrected systemd service file..."
sudo tee /etc/systemd/system/trading-algorithm.service > /dev/null << EOF
[Unit]
Description=Trading Algorithm Flask App
After=network.target
Wants=network.target

[Service]
Type=simple
User=$CURRENT_USER
Group=$CURRENT_USER
WorkingDirectory=$APP_DIR/flask_web
Environment=PATH=$APP_DIR/flask_web/venv/bin
Environment=FLASK_ENV=production
Environment=FLASK_APP=flask_app.py
ExecStart=$APP_DIR/flask_web/venv/bin/gunicorn --config gunicorn.conf.py wsgi:app
ExecReload=/bin/kill -s HUP \$MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=trading-algorithm

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$APP_DIR

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and restart service
print_status "Reloading systemd daemon..."
sudo systemctl daemon-reload

print_status "Starting trading algorithm service..."
sudo systemctl start trading-algorithm

# Wait a moment for the service to start
sleep 5

# Check if service is running
if sudo systemctl is-active --quiet trading-algorithm; then
    print_success "Trading Algorithm service started successfully"
else
    print_error "Failed to start Trading Algorithm service"
    print_status "Checking service logs..."
    sudo journalctl -u trading-algorithm --no-pager -n 10
    exit 1
fi

# Enable service for boot
print_status "Enabling service for boot..."
sudo systemctl enable trading-algorithm

print_success "Service configuration fixed and started!"
echo
echo "================================================================================"
echo "Service Status:"
echo "================================================================================"
sudo systemctl status trading-algorithm --no-pager -n 15
echo
echo "================================================================================" 