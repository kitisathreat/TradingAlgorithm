#!/bin/bash
# Trading Algorithm EC2 Clean Install Script
# This script does a complete clean install with proper file management

set -e

# Create log file
LOG_FILE="/tmp/trading_algorithm_install_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================================"
echo "              TRADING ALGORITHM - EC2 CLEAN INSTALL SCRIPT"
echo "================================================================================"
echo "Installation log: $LOG_FILE"
echo "Started at: $(date)"
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

# Function to log installation details
log_installation() {
    local component="$1"
    local location="$2"
    local version="$3"
    echo "[INSTALL] $component: $location (Version: $version)" >> "$LOG_FILE"
}

# Detect the correct user and home directory
CURRENT_USER=$(whoami)
USER_HOME=$(eval echo ~$CURRENT_USER)
APP_DIR="$USER_HOME/trading-algorithm"

print_status "System Information:"
echo "  Current user: $CURRENT_USER"
echo "  User home: $USER_HOME"
echo "  App directory: $APP_DIR"
echo "  Operating system: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "  Python version: $(python3.9 --version 2>/dev/null || echo 'Python 3.9 not found')"
echo "  Script location: $(pwd)"
echo "  Script directory contents:"
ls -la
echo

# COMPLETE CLEANUP - Remove ALL previous installations
print_status "Performing complete cleanup..."
echo "  Stopping all services..."
sudo systemctl stop trading-algorithm 2>/dev/null || true
sudo systemctl disable trading-algorithm 2>/dev/null || true
sudo systemctl stop nginx 2>/dev/null || true

echo "  Removing all service files..."
sudo rm -f /etc/systemd/system/trading-algorithm.service
sudo rm -f /etc/nginx/sites-enabled/trading-algorithm
sudo rm -f /etc/nginx/sites-available/trading-algorithm

echo "  Removing all application directories..."
rm -rf "$APP_DIR" 2>/dev/null || true
rm -rf "$USER_HOME/flask_web" 2>/dev/null || true
rm -rf "$USER_HOME/trading-algorithm" 2>/dev/null || true
rm -rf "/home/ec2-user/trading-algorithm" 2>/dev/null || true
rm -rf "/home/ubuntu/trading-algorithm" 2>/dev/null || true

echo "  Cleaning pip cache..."
pip cache purge 2>/dev/null || true
rm -rf ~/.cache/pip 2>/dev/null || true

echo "  Cleaning temporary files..."
sudo rm -rf /tmp/trading_algorithm_* 2>/dev/null || true
sudo rm -rf /tmp/pip_* 2>/dev/null || true

print_success "Complete cleanup finished"
echo

# Create fresh application directory
print_status "Creating fresh application directory..."
mkdir -p "$APP_DIR"
cd "$APP_DIR"

# The script is run from the flask_web directory where all files are located
print_status "Verifying required application files in current directory (flask_web)..."
REQUIRED_FILES=("flask_app.py" "requirements.txt" "wsgi.py" "gunicorn.conf.py" "config.py")
MISSING=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        print_error "Missing required file: $file in $(pwd)"
        MISSING=1
    fi
    
    # Optionally, print found files
    if [ -f "$file" ]; then
        print_status "Found: $file"
    fi
done

if [ $MISSING -ne 0 ]; then
    print_error "One or more required files are missing in $(pwd)."
    print_error "Please ensure all files are uploaded to this directory before running the install script."
    print_status "Current directory: $(pwd)"
    print_status "Files in current directory:"
    ls -la
    exit 1
fi

# All files are already in the correct location (flask_web directory)
print_success "All required files are present in $(pwd)"

# Create the proper directory structure for the ML orchestrator
mkdir -p "$APP_DIR/_2_Orchestrator_And_ML_Python"

# Copy ML orchestrator files if they exist in the parent directory
if [ -d "../_2_Orchestrator_And_ML_Python" ]; then
    print_status "Copying ML orchestrator files..."
    cp -r ../_2_Orchestrator_And_ML_Python "$APP_DIR/"
    print_success "ML orchestrator files copied"
    log_installation "ML Orchestrator" "$APP_DIR/_2_Orchestrator_And_ML_Python" "copied"
fi

# We're already in the flask_web directory, so no need to change directories
print_status "Working directory: $(pwd)"

# Create the proper directory structure
mkdir -p "$APP_DIR/flask_web"

# Copy files from current directory (where script is run from)
if [ -f "flask_app.py" ]; then
    print_status "Copying files from current directory..."
    cp flask_app.py "$APP_DIR/flask_web/"
    cp requirements.txt "$APP_DIR/flask_web/"
    cp wsgi.py "$APP_DIR/flask_web/"
    cp gunicorn.conf.py "$APP_DIR/flask_web/"
    cp config.py "$APP_DIR/flask_web/"
    
    if [ -d "templates" ]; then
        cp -r templates "$APP_DIR/flask_web/"
    fi
    
    print_success "Files copied with relative pathing maintained"
    log_installation "Application Files" "$APP_DIR" "copied with relative paths"
elif [ -f "../flask_app.py" ]; then
    print_status "Copying files from parent directory..."
    cp ../flask_app.py "$APP_DIR/flask_web/"
    cp ../requirements.txt "$APP_DIR/flask_web/"
    cp ../wsgi.py "$APP_DIR/flask_web/"
    cp ../gunicorn.conf.py "$APP_DIR/flask_web/"
    cp ../config.py "$APP_DIR/flask_web/"
    
    if [ -d "../templates" ]; then
        cp -r ../templates "$APP_DIR/flask_web/"
    fi
    
    print_success "Files copied from parent directory"
    log_installation "Application Files" "$APP_DIR" "copied from parent directory"
else
    print_error "Application files not found in current directory or parent directory"
    print_status "Please ensure this script is run from the flask_web directory"
    print_status "Expected files: flask_app.py, requirements.txt, wsgi.py, gunicorn.conf.py, config.py"
    print_status "Current directory: $(pwd)"
    print_status "Files in current directory:"
    ls -la
    if [ -d ".." ]; then
        print_status "Files in parent directory:"
        ls -la ..
    fi
    exit 1
fi

# Change to the flask_web directory for installation
cd "$APP_DIR/flask_web"

# Create virtual environment in the current directory (flask_web)
print_status "Creating Python virtual environment..."
python3.9 -m venv venv
source venv/bin/activate

# Log virtual environment details
log_installation "Virtual Environment" "$APP_DIR/flask_web/venv" "created"
echo "  Virtual environment: $APP_DIR/flask_web/venv"
echo "  Python in venv: $(which python)"
echo "  Python version in venv: $(python --version)"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip
PIP_VERSION=$(pip --version | cut -d' ' -f2)
log_installation "pip" "$(which pip)" "$PIP_VERSION"
echo "  pip version: $PIP_VERSION"

# Install Python dependencies with detailed logging
print_status "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    echo "  Installing from requirements.txt..."
    pip install -r requirements.txt
    
    # Verify key packages
    print_status "Verifying key package installations..."
    
    # Core packages
    GUNICORN_VERSION=$(./venv/bin/gunicorn --version 2>/dev/null | cut -d' ' -f1 || echo "NOT FOUND")
    FLASK_VERSION=$(python -c "import flask; print(flask.__version__)" 2>/dev/null || echo "NOT FOUND")
    TENSORFLOW_VERSION=$(python -c "import tensorflow; print(tensorflow.__version__)" 2>/dev/null || echo "NOT FOUND")
    NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "NOT FOUND")
    PANDAS_VERSION=$(python -c "import pandas; print(pandas.__version__)" 2>/dev/null || echo "NOT FOUND")
    YFINANCE_VERSION=$(python -c "import yfinance; print(yfinance.__version__)" 2>/dev/null || echo "NOT FOUND")
    
    # Log all installations
    log_installation "gunicorn" "$APP_DIR/flask_web/venv/bin/gunicorn" "$GUNICORN_VERSION"
    log_installation "Flask" "$(python -c 'import flask; print(flask.__file__)' 2>/dev/null || echo 'NOT FOUND')" "$FLASK_VERSION"
    log_installation "TensorFlow" "$(python -c 'import tensorflow; print(tensorflow.__file__)' 2>/dev/null || echo 'NOT FOUND')" "$TENSORFLOW_VERSION"
    log_installation "NumPy" "$(python -c 'import numpy; print(numpy.__file__)' 2>/dev/null || echo 'NOT FOUND')" "$NUMPY_VERSION"
    log_installation "Pandas" "$(python -c 'import pandas; print(pandas.__file__)' 2>/dev/null || echo 'NOT FOUND')" "$PANDAS_VERSION"
    log_installation "yfinance" "$(python -c 'import yfinance; print(yfinance.__file__)' 2>/dev/null || echo 'NOT FOUND')" "$YFINANCE_VERSION"
    
    echo "  Key package versions:"
    echo "    gunicorn: $GUNICORN_VERSION"
    echo "    Flask: $FLASK_VERSION"
    echo "    TensorFlow: $TENSORFLOW_VERSION"
    echo "    NumPy: $NUMPY_VERSION"
    echo "    Pandas: $PANDAS_VERSION"
    echo "    yfinance: $YFINANCE_VERSION"
    
    print_success "Python dependencies installed successfully"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Verify gunicorn installation
print_status "Verifying gunicorn installation..."
if [ -f "venv/bin/gunicorn" ]; then
    print_success "Gunicorn found at: $(pwd)/venv/bin/gunicorn"
    ./venv/bin/gunicorn --version
    log_installation "gunicorn" "$(pwd)/venv/bin/gunicorn" "$(./venv/bin/gunicorn --version | cut -d' ' -f1)"
else
    print_error "Gunicorn not found in virtual environment"
    print_status "Attempting to install gunicorn manually..."
    pip install gunicorn==21.2.0
    if [ -f "venv/bin/gunicorn" ]; then
        print_success "Gunicorn installed successfully"
        log_installation "gunicorn" "$(pwd)/venv/bin/gunicorn" "$(./venv/bin/gunicorn --version | cut -d' ' -f1)"
    else
        print_error "Failed to install gunicorn"
        exit 1
    fi
fi

# Create systemd service file with correct paths
print_status "Creating systemd service file..."
cat > trading-algorithm.service << EOF
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

log_installation "systemd service" "/etc/systemd/system/trading-algorithm.service" "created"
echo "  Service file created: $APP_DIR/flask_web/trading-algorithm.service"

# Setup nginx configuration
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
        alias $APP_DIR/flask_web/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

log_installation "nginx config" "/etc/nginx/sites-available/trading-algorithm" "created"

# Setup nginx
print_status "Configuring nginx..."
sudo cp nginx-trading-algorithm /etc/nginx/sites-available/trading-algorithm
sudo ln -sf /etc/nginx/sites-available/trading-algorithm /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx configuration
if sudo nginx -t; then
    sudo systemctl restart nginx
    sudo systemctl enable nginx
    NGINX_VERSION=$(nginx -v 2>&1 | cut -d'/' -f2)
    log_installation "nginx" "$(which nginx)" "$NGINX_VERSION"
    print_success "Nginx configured and started successfully"
    echo "  nginx version: $NGINX_VERSION"
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
sudo chown -R $CURRENT_USER:$CURRENT_USER $APP_DIR
chmod +x $APP_DIR/flask_web/venv/bin/*

# Create log directory
mkdir -p $APP_DIR/logs
sudo chown -R $CURRENT_USER:$CURRENT_USER $APP_DIR/logs

# Test the application manually first
print_status "Testing application manually..."
cd $APP_DIR/flask_web
source venv/bin/activate
timeout 10s ./venv/bin/gunicorn --config gunicorn.conf.py wsgi:app &
GUNICORN_PID=$!
sleep 5

if kill -0 $GUNICORN_PID 2>/dev/null; then
    print_success "Application test successful"
    kill $GUNICORN_PID
    wait $GUNICORN_PID 2>/dev/null || true
else
    print_error "Application test failed"
    exit 1
fi

# Start the application
print_status "Starting trading algorithm application..."
sudo systemctl start trading-algorithm

# Wait a moment for the service to start
sleep 5

# Check if service is running
if sudo systemctl is-active --quiet trading-algorithm; then
    print_success "Trading Algorithm service started successfully"
    log_installation "trading-algorithm service" "systemd" "running"
else
    print_error "Failed to start Trading Algorithm service"
    sudo systemctl status trading-algorithm
    print_status "Checking service logs..."
    sudo journalctl -u trading-algorithm --no-pager -n 20
    exit 1
fi

# Get public IP
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "unknown")

# Create installation summary
INSTALL_SUMMARY="$APP_DIR/installation_summary.txt"
cat > "$INSTALL_SUMMARY" << EOF
===============================================================================
                    TRADING ALGORITHM - INSTALLATION SUMMARY
===============================================================================
Installation Date: $(date)
Installation Log: $LOG_FILE

SYSTEM INFORMATION:
- User: $CURRENT_USER
- Home Directory: $USER_HOME
- Application Directory: $APP_DIR
- Operating System: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)
- Public IP: $PUBLIC_IP

PYTHON ENVIRONMENT:
- Virtual Environment: $APP_DIR/flask_web/venv
- Python Version: $(python --version)
- pip Version: $(pip --version | cut -d' ' -f2)

KEY PACKAGES INSTALLED:
- gunicorn: $GUNICORN_VERSION
- Flask: $FLASK_VERSION
- TensorFlow: $TENSORFLOW_VERSION
- NumPy: $NUMPY_VERSION
- Pandas: $PANDAS_VERSION
- yfinance: $YFINANCE_VERSION

SERVICES:
- nginx: $NGINX_VERSION
- trading-algorithm: running (systemd)

CONFIGURATION FILES:
- Service File: /etc/systemd/system/trading-algorithm.service
- Nginx Config: /etc/nginx/sites-available/trading-algorithm
- Application Config: $APP_DIR/flask_web/config.py
- Gunicorn Config: $APP_DIR/flask_web/gunicorn.conf.py

DIRECTORY STRUCTURE:
$APP_DIR/
├── flask_web/
│   ├── flask_app.py
│   ├── requirements.txt
│   ├── wsgi.py
│   ├── gunicorn.conf.py
│   ├── config.py
│   ├── templates/
│   └── venv/
├── _2_Orchestrator_And_ML_Python/
└── logs/

ACCESS INFORMATION:
- Application URL: http://$PUBLIC_IP
- SSH Access: ssh -i your-key.pem $CURRENT_USER@$PUBLIC_IP

USEFUL COMMANDS:
- Check app status: sudo systemctl status trading-algorithm
- View app logs: sudo journalctl -u trading-algorithm -f
- Restart app: sudo systemctl restart trading-algorithm
- Check nginx: sudo systemctl status nginx
- View nginx logs: sudo tail -f /var/log/nginx/access.log
- Monitor system: htop
- Check disk space: df -h

===============================================================================
EOF

echo
echo "================================================================================"
print_success "Trading Algorithm clean install completed!"
echo "================================================================================"
echo
echo "Installation Summary: $INSTALL_SUMMARY"
echo "Detailed Log: $LOG_FILE"
echo
echo "Application URL: http://$PUBLIC_IP"
echo "SSH Access: ssh -i your-key.pem $CURRENT_USER@$PUBLIC_IP"
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