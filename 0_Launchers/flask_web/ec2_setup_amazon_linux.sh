#!/bin/bash
# Trading Algorithm EC2 Setup Script for Amazon Linux
# This script configures an Amazon Linux EC2 instance for running the trading algorithm
# Updated with all fixes: curl conflicts, TensorFlow disk space, file copying, error handling

set -e

echo "================================================================================"
echo "                    TRADING ALGORITHM - AMAZON LINUX SETUP SCRIPT"
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
sudo yum update -y

# Fix curl package conflict by removing curl-minimal and installing full curl
print_status "Resolving curl package conflict..."
if rpm -q curl-minimal &> /dev/null; then
    print_status "Attempting to remove curl-minimal package..."
    # Try to remove curl-minimal, but skip if it's protected
    sudo yum remove -y curl-minimal --skip-broken || true
    print_warning "curl-minimal may still be present (protected package), continuing with installation..."
fi

# Install Python and dependencies with curl conflict handling
print_status "Installing Python and system dependencies..."
sudo yum install -y python3 python3-pip python3-devel
sudo yum install -y nginx gcc gcc-c++ make openssl-devel libffi-devel
sudo yum install -y git wget unzip htop

# Install curl separately to avoid conflicts
print_status "Installing curl (full version)..."
sudo yum install -y curl --skip-broken || print_warning "curl installation may have conflicts, but continuing..."

# Install Python 3.9 if not available
if ! command -v python3.9 &> /dev/null; then
    print_status "Installing Python 3.9..."
    sudo yum install -y python39 python39-pip python39-devel
    # Create symlink for python3.9
    sudo ln -sf /usr/bin/python3.9 /usr/bin/python3.9
fi

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

# Copy application files from current directory (flask_web)
print_status "Copying application files..."
if [ -f "flask_app.py" ]; then
    # Create the application directory
    mkdir -p /home/ec2-user/trading-algorithm
    cd /home/ec2-user/trading-algorithm
    
    # Copy files from the flask_web directory
    cp /home/ec2-user/flask_web/flask_app.py .
    cp /home/ec2-user/flask_web/requirements.txt .
    cp /home/ec2-user/flask_web/wsgi.py .
    cp /home/ec2-user/flask_web/gunicorn.conf.py .
    cp /home/ec2-user/flask_web/config.py .
    
    if [ -d "/home/ec2-user/flask_web/templates" ]; then
        cp -r /home/ec2-user/flask_web/templates .
    fi
    
    # Copy orchestrator files if they exist in the uploaded directory
    if [ -d "/home/ec2-user/flask_web/_2_Orchestrator_And_ML_Python" ]; then
        cp -r /home/ec2-user/flask_web/_2_Orchestrator_And_ML_Python .
    fi
    
    print_success "Application files copied successfully"
else
    print_error "flask_app.py not found in flask_web directory"
    print_status "Current directory contents:"
    ls -la
    exit 1
fi

# Create virtual environment
print_status "Creating Python virtual environment..."
python3.9 -m venv venv
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Clean up disk space before installing large packages
print_status "Cleaning up disk space for large package installation..."
sudo yum clean all
sudo rm -rf /tmp/*
sudo rm -rf /var/tmp/*
sudo rm -rf /var/cache/yum/*
sudo rm -rf /var/cache/dnf/*

# Clean up deprecated and old pip packages
print_status "Cleaning up deprecated pip packages and caches..."
pip cache purge
pip uninstall -y tensorflow tensorflow-cpu tensorflow-gpu || true
pip uninstall -y $(pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1) || true

# Remove old package versions and unused dependencies
print_status "Removing old package versions and unused dependencies..."
pip uninstall -y $(pip check 2>&1 | grep -o "'.*'" | tr -d "'") || true

# Clean up system deprecated files
print_status "Cleaning up system deprecated files..."
sudo rm -rf /var/log/*.old
sudo rm -rf /var/log/*.gz
sudo rm -rf /var/cache/man/*
sudo rm -rf /var/cache/fontconfig/*
sudo rm -rf /var/cache/ldconfig/*

# Clean up any remaining temporary files
print_status "Cleaning up remaining temporary files..."
find /tmp -type f -atime +1 -delete 2>/dev/null || true
find /var/tmp -type f -atime +1 -delete 2>/dev/null || true
find /home/ec2-user -name "*.pyc" -delete 2>/dev/null || true
find /home/ec2-user -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Check available disk space
print_status "Checking available disk space after cleanup..."
df -h /

# Force pip to use disk storage instead of RAM
print_status "Configuring pip to use disk storage instead of RAM..."
export TMPDIR=/home/ec2-user/trading-algorithm/tmp
export TEMP=/home/ec2-user/trading-algorithm/tmp
export TMP=/home/ec2-user/trading-algorithm/tmp
export PIP_TARGET=/home/ec2-user/trading-algorithm/tmp
export PIP_CACHE_DIR=/home/ec2-user/trading-algorithm/pip_cache
export PIP_DOWNLOAD_CACHE=/home/ec2-user/trading-algorithm/pip_cache

# Create disk-based directories for pip operations
mkdir -p $TMPDIR
mkdir -p $PIP_CACHE_DIR
mkdir -p $PIP_DOWNLOAD_CACHE

# Configure pip to use disk-based cache and avoid RAM
pip config set global.cache-dir /home/ec2-user/trading-algorithm/pip_cache
pip config set global.temp /home/ec2-user/trading-algorithm/tmp
pip config set global.download-cache /home/ec2-user/trading-algorithm/pip_cache

print_status "Pip configured to use disk storage. Cache directory: $PIP_CACHE_DIR"
print_status "Temporary directory: $TMPDIR"

# Install Python dependencies with proper disk space management
print_status "Installing Python dependencies using disk storage..."
if [ -f "requirements.txt" ]; then
    # Install core data science packages first (smaller packages)
    print_status "Installing core data science packages..."
    pip install pandas==2.0.3 numpy==1.24.3 scikit-learn==1.3.0 --cache-dir $PIP_CACHE_DIR --build $TMPDIR
    
    # Install visualization packages
    print_status "Installing visualization packages..."
    pip install matplotlib==3.7.0 plotly==5.18.0 seaborn==0.12.0 --cache-dir $PIP_CACHE_DIR --build $TMPDIR
    
    # Check if TensorFlow is already installed
    if python -c "import tensorflow; print('TensorFlow version:', tensorflow.__version__)" 2>/dev/null; then
        print_success "TensorFlow is already installed, skipping installation"
    else
        # Install full TensorFlow from requirements.txt
        print_status "Installing full TensorFlow (from requirements.txt)..."
        if pip install tensorflow==2.13.0 --cache-dir $PIP_CACHE_DIR --build $TMPDIR; then
            print_success "Full TensorFlow installed successfully"
        else
            print_warning "Full TensorFlow installation failed, trying CPU version as fallback..."
            # Fallback to CPU version if full version fails
            pip install tensorflow-cpu==2.13.0 --cache-dir $PIP_CACHE_DIR --build $TMPDIR || print_warning "TensorFlow installation failed, continuing without it"
        fi
    fi
    
    # Install remaining dependencies (excluding TensorFlow if it was already installed)
    print_status "Installing remaining Python dependencies..."
    pip install -r requirements.txt --cache-dir $PIP_CACHE_DIR --build $TMPDIR --ignore-installed tensorflow
    
    print_success "Python dependencies installed successfully"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Clean up temporary files after installation
print_status "Cleaning up temporary files after installation..."
rm -rf $TMPDIR/*
rm -rf $PIP_CACHE_DIR/*

# Clean up deprecated files and build artifacts
print_status "Cleaning up deprecated files and build artifacts..."
pip cache purge
find /home/ec2-user/trading-algorithm -name "*.pyc" -delete 2>/dev/null || true
find /home/ec2-user/trading-algorithm -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find /home/ec2-user/trading-algorithm -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
find /home/ec2-user/trading-algorithm -name "build" -type d -exec rm -rf {} + 2>/dev/null || true
find /home/ec2-user/trading-algorithm -name "dist" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove unused dependencies
print_status "Removing unused dependencies..."
pip uninstall -y $(pip check 2>&1 | grep -o "'.*'" | tr -d "'") || true

# Final disk space check
print_status "Final disk space check..."
df -h /

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
    gzip_proxied expired no-cache no-store private auth;
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
        alias /home/ec2-user/trading-algorithm/static;
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
User=ec2-user
Group=ec2-user
WorkingDirectory=/home/ec2-user/trading-algorithm
Environment=PATH=/home/ec2-user/trading-algorithm/venv/bin
Environment=FLASK_ENV=production
Environment=FLASK_APP=flask_app.py
Environment=TMPDIR=/home/ec2-user/trading-algorithm/tmp
ExecStart=/home/ec2-user/trading-algorithm/venv/bin/gunicorn --config gunicorn.conf.py wsgi:app
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
ReadWritePaths=/home/ec2-user/trading-algorithm

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

# Set permissions
print_status "Setting file permissions..."
sudo chown -R ec2-user:ec2-user /home/ec2-user/trading-algorithm
chmod +x /home/ec2-user/trading-algorithm/venv/bin/*

# Create log directory
mkdir -p /home/ec2-user/trading-algorithm/logs
sudo chown -R ec2-user:ec2-user /home/ec2-user/trading-algorithm/logs

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
echo "SSH Access: ssh -i your-key.pem ec2-user@$PUBLIC_IP"
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