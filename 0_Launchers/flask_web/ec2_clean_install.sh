#!/bin/bash
# Trading Algorithm EC2 Clean Installation Script
# This script completely removes all previous installations and starts fresh

set -e

echo "================================================================================"
echo "                    TRADING ALGORITHM - CLEAN INSTALLATION"
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

print_warning "This will completely remove all previous installations!"
print_warning "All data, packages, and configurations will be deleted!"
echo
read -p "Are you sure you want to continue? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy](es)?$ ]]; then
    print_error "Installation cancelled"
    exit 1
fi

# Stop any running services
print_status "Stopping existing services..."
sudo systemctl stop trading-algorithm 2>/dev/null || true
sudo systemctl disable trading-algorithm 2>/dev/null || true
sudo systemctl stop nginx 2>/dev/null || true

# Remove systemd service
print_status "Removing systemd service..."
sudo rm -f /etc/systemd/system/trading-algorithm.service
sudo systemctl daemon-reload

# Remove nginx configuration
print_status "Removing nginx configuration..."
sudo rm -f /etc/nginx/conf.d/trading-algorithm.conf

# Complete cleanup of all previous installations
print_status "Performing complete cleanup..."
sudo rm -rf /home/ec2-user/trading-algorithm
sudo rm -rf /home/ec2-user/flask_web
sudo rm -rf /tmp/*
sudo rm -rf /var/tmp/*
sudo rm -rf /var/cache/yum/*
sudo rm -rf /var/cache/dnf/*
sudo rm -rf /var/log/*.old
sudo rm -rf /var/log/*.gz
sudo rm -rf /var/cache/man/*
sudo rm -rf /var/cache/fontconfig/*
sudo rm -rf /var/cache/ldconfig/*

# Clean up any Python environments
print_status "Cleaning up Python environments..."
find /home/ec2-user -name "venv" -type d -exec rm -rf {} + 2>/dev/null || true
find /home/ec2-user -name "*.pyc" -delete 2>/dev/null || true
find /home/ec2-user -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find /home/ec2-user -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
find /home/ec2-user -name "build" -type d -exec rm -rf {} + 2>/dev/null || true
find /home/ec2-user -name "dist" -type d -exec rm -rf {} + 2>/dev/null || true

# Clean pip cache globally
print_status "Cleaning pip cache..."
pip cache purge 2>/dev/null || true
rm -rf ~/.cache/pip 2>/dev/null || true

# Check available disk space
print_status "Checking available disk space after cleanup..."
df -h /

# Now run the fresh installation
print_status "Starting fresh installation..."
cd /home/ec2-user
chmod +x flask_web/ec2_setup_amazon_linux.sh
./flask_web/ec2_setup_amazon_linux.sh

print_success "Clean installation completed!" 