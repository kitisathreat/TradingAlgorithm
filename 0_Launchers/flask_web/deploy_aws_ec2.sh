#!/bin/bash

# AWS EC2 Deployment Script for Trading Algorithm Web Interface
# This script sets up and deploys the Flask application on an EC2 instance

set -e

echo "=========================================="
echo "  Trading Algorithm Web Interface"
echo "  AWS EC2 Deployment Script"
echo "=========================================="

# Configuration
APP_NAME="trading-algorithm-web"
APP_DIR="/opt/$APP_NAME"
SERVICE_NAME="$APP_NAME"
USER_NAME="trading-app"

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

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Update system packages
print_status "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install required system packages
print_status "Installing system dependencies..."
sudo apt-get install -y \
    python3.9 \
    python3.9-pip \
    python3.9-venv \
    nginx \
    supervisor \
    curl \
    git \
    build-essential \
    python3-dev \
    libxml2-dev \
    libxslt-dev \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgthread-2.0-0

# Create application user
print_status "Creating application user..."
sudo useradd -r -s /bin/false $USER_NAME || true

# Create application directory
print_status "Setting up application directory..."
sudo mkdir -p $APP_DIR
sudo chown $USER:$USER $APP_DIR

# Clone or copy application files
if [ -d ".git" ]; then
    print_status "Copying application files..."
    sudo cp -r . $APP_DIR/
else
    print_status "Cloning application from repository..."
    sudo git clone https://github.com/your-repo/trading-algorithm.git $APP_DIR
fi

# Set proper permissions
sudo chown -R $USER_NAME:$USER_NAME $APP_DIR

# Create virtual environment
print_status "Setting up Python virtual environment..."
cd $APP_DIR
python3.9 -m venv venv
source venv/bin/activate

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
print_status "Creating application directories..."
mkdir -p logs models cache

# Set environment variables
print_status "Setting up environment variables..."
cat > .env << EOF
FLASK_ENV=production
SECRET_KEY=$(openssl rand -hex 32)
LOG_LEVEL=INFO
CORS_ORIGINS=*
MODEL_SAVE_PATH=$APP_DIR/models
DATA_CACHE_DIR=$APP_DIR/cache
EOF

# Setup Supervisor configuration
print_status "Setting up Supervisor configuration..."
sudo tee /etc/supervisor/conf.d/$SERVICE_NAME.conf > /dev/null << EOF
[program:$SERVICE_NAME]
command=$APP_DIR/venv/bin/gunicorn --config $APP_DIR/gunicorn.conf.py wsgi:application
directory=$APP_DIR
user=$USER_NAME
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=$APP_DIR/logs/gunicorn.log
environment=FLASK_ENV=production
EOF

# Setup Nginx configuration
print_status "Setting up Nginx configuration..."
sudo tee /etc/nginx/sites-available/$SERVICE_NAME > /dev/null << EOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /static {
        alias $APP_DIR/static;
        expires 30d;
    }
}
EOF

# Enable Nginx site
sudo ln -sf /etc/nginx/sites-available/$SERVICE_NAME /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
sudo nginx -t

# Start services
print_status "Starting services..."
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start $SERVICE_NAME
sudo systemctl restart nginx
sudo systemctl enable nginx

# Setup firewall (if ufw is available)
if command -v ufw &> /dev/null; then
    print_status "Setting up firewall..."
    sudo ufw allow 22/tcp
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp
    sudo ufw --force enable
fi

# Health check
print_status "Performing health check..."
sleep 5
if curl -f http://localhost/api/get_model_status > /dev/null 2>&1; then
    print_status "Application is running successfully!"
    print_status "Access your application at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
else
    print_error "Application failed to start. Check logs at $APP_DIR/logs/"
    exit 1
fi

print_status "Deployment completed successfully!"
print_status "Application logs: $APP_DIR/logs/"
print_status "Supervisor logs: sudo supervisorctl status $SERVICE_NAME"
print_status "Nginx logs: sudo tail -f /var/log/nginx/access.log" 