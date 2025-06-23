#!/bin/bash
# Trading Algorithm EC2 Clean Install Script
# This script does a complete clean install with proper file management

set -e

# Create log file
mkdir -p /home/ec2-user/trading-algorithm/flask_web/logs
LOG_FILE="/home/ec2-user/trading-algorithm/flask_web/logs/trading_algorithm_install_$(date +%Y%m%d_%H%M%S).log"
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

# Skip cleanup since upload script already cleaned up and uploaded files
print_status "Skipping cleanup - upload script already handled this..."
print_status "Proceeding directly to file verification and installation..."

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
    if [ -d "$APP_DIR/_2_Orchestrator_And_ML_Python" ]; then
        print_status "ML orchestrator directory already exists, skipping copy"
    else
        cp -r ../_2_Orchestrator_And_ML_Python "$APP_DIR/"
        print_success "ML orchestrator files copied"
    fi
    log_installation "ML Orchestrator" "$APP_DIR/_2_Orchestrator_And_ML_Python" "present"
fi

# We're already in the flask_web directory, so no need to change directories
print_status "Working directory: $(pwd)"

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

# Create systemd service file
print_status "Disabling and removing systemd service..."
# Stop and disable the service if it exists from a previous installation
if sudo systemctl list-units --type=service | grep -q 'trading-algorithm.service'; then
    sudo systemctl stop trading-algorithm || true
    sudo systemctl disable trading-algorithm || true
    print_success "Stopped and disabled existing systemd service."
fi
# Remove the service file to prevent conflicts
sudo rm -f /etc/systemd/system/trading-algorithm.service
sudo systemctl daemon-reload
print_success "Removed old systemd service file."

# Setup nginx configuration
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
        alias $USER_HOME/trading-algorithm/flask_web/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

log_installation "nginx config" "/etc/nginx/conf.d/trading-algorithm.conf" "created"

# Setup nginx
print_status "Configuring nginx..."
sudo cp nginx-trading-algorithm /etc/nginx/conf.d/trading-algorithm.conf
sudo rm -f /etc/nginx/conf.d/default.conf

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

# Test the application manually first
print_status "Skipping manual application test as we will launch with nohup."

# Start the application using nohup
print_status "Starting trading algorithm application using nohup..."
cd $APP_DIR/flask_web

# Kill any existing gunicorn processes to ensure a clean start
# Use fuser to kill any process using port 8000, which is more reliable than pkill
print_status "Ensuring port 8000 is free..."
sudo fuser -k 8000/tcp || true
sleep 2

# Activate virtual environment and launch in the background
source venv/bin/activate
nohup gunicorn --config gunicorn.conf.py wsgi:app > app.log 2>&1 &
sleep 5 # Wait a few seconds for the process to initialize

# Check if the process is running
if pgrep -f "gunicorn.*wsgi:app" > /dev/null; then
    print_success "Trading Algorithm application started successfully with nohup."
    log_installation "trading-algorithm process" "nohup" "running"
else
    print_error "Failed to start Trading Algorithm application with nohup."
    print_status "Checking logs in app.log..."
    cat app.log
    exit 1
fi

# Deactivate for safety
deactivate

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
- trading-algorithm: running (nohup)

CONFIGURATION FILES:
- Nginx Config: /etc/nginx/conf.d/trading-algorithm.conf
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
- Check app status: ps aux | grep gunicorn
- View app logs: tail -f $APP_DIR/flask_web/app.log
- Restart app: cd $APP_DIR/flask_web && pkill -f gunicorn && source venv/bin/activate && nohup gunicorn --config gunicorn.conf.py wsgi:app > app.log 2>&1 &
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
echo "  Health check: bash health_check.sh"
echo "  Check app status: ps aux | grep gunicorn"
echo "  View app logs: tail -f $APP_DIR/flask_web/app.log"
echo "  Restart app: cd $APP_DIR/flask_web && pkill -f gunicorn && source venv/bin/activate && nohup gunicorn --config gunicorn.conf.py wsgi:app > app.log 2>&1 &"
echo "  Check nginx: sudo systemctl status nginx"
echo "  View nginx logs: sudo tail -f /var/log/nginx/access.log"
echo "  Monitor system: htop"
echo "  Check disk space: df -h"
echo
echo "================================================================================"

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
echo "1. Checking Trading Algorithm Process..."
if pgrep -f "gunicorn.*wsgi:app" > /dev/null; then
    print_success "Trading Algorithm process is running (via gunicorn)"
    ps aux | grep gunicorn | grep -v grep
else
    print_error "Trading Algorithm process is not running"
    print_status "Checking recent logs from app.log:"
    tail -n 10 "$APP_DIR/flask_web/app.log"
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

# 5. Check File Structure
echo "5. Checking File Structure..."
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

# 6. Check System Resources
echo "6. Checking System Resources..."
DISK_USAGE=$(df -h / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -lt 80 ]; then
    print_success "Disk usage: ${DISK_USAGE}% (healthy)"
else
    print_error "Disk usage: ${DISK_USAGE}% (high usage)"
fi

MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
print_status "Memory usage: ${MEMORY_USAGE}%"

echo

# 7. Check Recent Logs
echo "8. Checking Recent Logs..."
print_status "Recent application logs (last 10 lines from app.log):"
tail -n 10 "$APP_DIR/flask_web/app.log" 2>/dev/null || echo "No app.log found"

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
echo "  Restart trading algorithm: cd $APP_DIR/flask_web && pkill -f gunicorn && source venv/bin/activate && nohup gunicorn --config gunicorn.conf.py wsgi:app > app.log 2>&1 &"
echo "  Restart nginx: sudo systemctl restart nginx"
echo "  View app logs: tail -f $APP_DIR/flask_web/app.log"
echo "  View nginx logs: sudo tail -f /var/log/nginx/error.log"
echo "  Check nginx config: sudo nginx -t"
echo
EOF

chmod +x health_check.sh
log_installation "health check script" "$APP_DIR/health_check.sh" "created"
print_success "Health check script created successfully" 