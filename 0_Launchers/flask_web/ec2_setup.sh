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
cat > nginx-trading-algorithm << EOF
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
    gzip_proxied expired no-cache no-store private auth;
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml+rss;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /socket.io {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Static files (if any)
    location /static {
        alias $USER_HOME/trading-algorithm/static;
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
sudo cp nginx-trading-algorithm /etc/nginx/conf.d/trading-algorithm.conf
sudo rm -f /etc/nginx/conf.d/default.conf

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

# Create health check script
print_status "Creating health check script..."
cat > health_check.sh << 'EOF'
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
echo "  Restart trading algorithm: sudo systemctl restart trading-algorithm"
echo "  Restart nginx: sudo systemctl restart nginx"
echo "  View app logs: sudo journalctl -u trading-algorithm -f"
echo "  View nginx logs: sudo tail -f /var/log/nginx/error.log"
echo "  Check nginx config: sudo nginx -t"
echo "  Check service config: sudo systemctl cat trading-algorithm"
echo
EOF

chmod +x health_check.sh
print_success "Health check script created successfully"

# Set permissions
print_status "Setting file permissions..."
sudo chown -R ubuntu:ubuntu /home/ubuntu/trading-algorithm
chmod +x /home/ubuntu/trading-algorithm/flask_web/venv/bin/*

# Create log directory
mkdir -p /home/ec2-user/trading-algorithm/flask_web/logs
sudo chown -R ubuntu:ubuntu /home/ec2-user/trading-algorithm/flask_web/logs

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
echo "  Health check: bash health_check.sh"
echo "  Check app status: sudo systemctl status trading-algorithm"
echo "  View app logs: sudo journalctl -u trading-algorithm -f"
echo "  Restart app: sudo systemctl restart trading-algorithm"
echo "  Check nginx: sudo systemctl status nginx"
echo "  View nginx logs: sudo tail -f /var/log/nginx/access.log"
echo "  Monitor system: htop"
echo "  Check disk space: df -h"
echo
echo "================================================================================" 