#!/bin/bash
# Trading Algorithm EC2 Setup Script
# This script configures an EC2 instance for running the trading algorithm

set -e

echo "================================================================================"
echo "                    TRADING ALGORITHM - EC2 SETUP SCRIPT"
echo "================================================================================"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Update system
print_status "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and dependencies
print_status "Installing Python and system dependencies..."
sudo apt-get install -y python3.9 python3.9-venv python3.9-dev python3-pip
sudo apt-get install -y nginx build-essential libssl-dev libffi-dev
sudo apt-get install -y git curl wget unzip htop

# Install AWS CLI
print_status "Installing AWS CLI..."
if ! command -v aws &> /dev/null; then
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    rm -rf aws awscliv2.zip
    print_success "AWS CLI installed successfully"
else
    print_warning "AWS CLI already installed"
fi

# Create application directory
print_status "Setting up application directory..."
mkdir -p /home/ubuntu/trading-algorithm
cd /home/ubuntu/trading-algorithm

# Copy application files (assuming they're in the current directory)
print_status "Copying application files..."
if [ -f "../flask_app.py" ]; then
    cp ../flask_app.py .
    cp ../requirements.txt .
    cp ../wsgi.py .
    cp ../gunicorn.conf.py .
    cp ../config.py .
    
    if [ -d "../templates" ]; then
        cp -r ../templates .
    fi
    
    if [ -d "../_2_Orchestrator_And_ML_Python" ]; then
        cp -r ../_2_Orchestrator_And_ML_Python .
    fi
    
    print_success "Application files copied successfully"
else
    print_warning "Application files not found in parent directory"
    print_status "Please ensure flask_app.py, requirements.txt, and other files are available"
fi

# Create virtual environment in the correct location
python3.9 -m venv flask_web/venv
source flask_web/venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
print_status "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_success "Python dependencies installed successfully"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Create nginx configuration
print_status "Creating nginx configuration..."
cat > nginx-trading-algorithm << 'EOF'
server {
    listen 80;
    server_name _;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied expired no-cache no-store private must-revalidate auth;
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml+rss;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /socket.io {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Static files (if any)
    location /static {
        alias /home/ubuntu/trading-algorithm/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

# Create systemd service file
print_status "Creating systemd service file..."
cat > trading-algorithm.service << 'EOF'
[Unit]
Description=Trading Algorithm Flask App
After=network.target
Wants=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/trading-algorithm
Environment=PATH=/home/ubuntu/trading-algorithm/flask_web/venv/bin
Environment=FLASK_ENV=production
Environment=FLASK_APP=flask_app.py
ExecStart=/home/ubuntu/trading-algorithm/flask_web/venv/bin/gunicorn --config gunicorn.conf.py wsgi:app
ExecReload=/bin/kill -s HUP $MAINPID
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
ReadWritePaths=/home/ubuntu/trading-algorithm

[Install]
WantedBy=multi-user.target
EOF

# Setup nginx
print_status "Configuring nginx..."
sudo cp nginx-trading-algorithm /etc/nginx/sites-available/trading-algorithm
sudo ln -sf /etc/nginx/sites-available/trading-algorithm /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx configuration
if sudo nginx -t; then
    sudo systemctl restart nginx
    sudo systemctl enable nginx
    print_success "Nginx configured and started successfully"
else
    print_error "Nginx configuration test failed"
    exit 1
fi

# Setup systemd service
print_status "Setting up systemd service..."
sudo cp trading-algorithm.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable trading-algorithm

# Set permissions
print_status "Setting file permissions..."
sudo chown -R ubuntu:ubuntu /home/ubuntu/trading-algorithm
chmod +x /home/ubuntu/trading-algorithm/flask_web/venv/bin/*

# Create log directory
mkdir -p /home/ubuntu/trading-algorithm/logs
sudo chown -R ubuntu:ubuntu /home/ubuntu/trading-algorithm/logs

# Start the application
print_status "Starting trading algorithm application..."
sudo systemctl start trading-algorithm

# Wait a moment for the service to start
sleep 5

# Check if service is running
if sudo systemctl is-active --quiet trading-algorithm; then
    print_success "Trading Algorithm service started successfully"
else
    print_error "Failed to start Trading Algorithm service"
    sudo systemctl status trading-algorithm
    exit 1
fi

# Get public IP
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "unknown")

echo
echo "================================================================================"
print_success "Trading Algorithm setup completed!"
echo "================================================================================"
echo
echo "Application URL: http://$PUBLIC_IP"
echo "SSH Access: ssh -i your-key.pem ubuntu@$PUBLIC_IP"
echo
echo "Useful commands:"
echo "  Check app status: sudo systemctl status trading-algorithm"
echo "  View app logs: sudo journalctl -u trading-algorithm -f"
echo "  Restart app: sudo systemctl restart trading-algorithm"
echo "  Check nginx: sudo systemctl status nginx"
echo "  View nginx logs: sudo tail -f /var/log/nginx/access.log"
echo "  Monitor system: htop"
echo "  Check disk space: df -h"
echo
echo "================================================================================" 