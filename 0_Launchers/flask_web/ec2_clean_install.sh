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

# Function to count files/directories to be deleted
count_items_to_delete() {
    local count=0
    
    # Count existing services and configurations
    if systemctl is-active --quiet trading-algorithm 2>/dev/null; then count=$((count + 1)); fi
    if systemctl is-enabled --quiet trading-algorithm 2>/dev/null; then count=$((count + 1)); fi
    if [ -f /etc/systemd/system/trading-algorithm.service ]; then count=$((count + 1)); fi
    if [ -f /etc/nginx/conf.d/trading-algorithm.conf ]; then count=$((count + 1)); fi
    
    # Count directories to be removed
    if [ -d /home/ec2-user/trading-algorithm ]; then count=$((count + 1)); fi
    if [ -d /home/ec2-user/flask_web ]; then count=$((count + 1)); fi
    
    # Count cache directories
    if [ -d /tmp ] && [ "$(ls -A /tmp 2>/dev/null)" ]; then count=$((count + 1)); fi
    if [ -d /var/tmp ] && [ "$(ls -A /var/tmp 2>/dev/null)" ]; then count=$((count + 1)); fi
    if [ -d /var/cache/yum ] && [ "$(ls -A /var/cache/yum 2>/dev/null)" ]; then count=$((count + 1)); fi
    if [ -d /var/cache/dnf ] && [ "$(ls -A /var/cache/dnf 2>/dev/null)" ]; then count=$((count + 1)); fi
    
    # Count Python environments
    local venv_count=$(find /home/ec2-user -name "venv" -type d 2>/dev/null | wc -l)
    count=$((count + venv_count))
    
    local pyc_count=$(find /home/ec2-user -name "*.pyc" 2>/dev/null | wc -l)
    count=$((count + pyc_count))
    
    local pycache_count=$(find /home/ec2-user -name "__pycache__" -type d 2>/dev/null | wc -l)
    count=$((count + pycache_count))
    
    echo $count
}

# Function to show progress
show_progress() {
    local current=$1
    local total=$2
    local item_name=$3
    
    local percentage=$((current * 100 / total))
    local filled=$((percentage / 2))
    local empty=$((50 - filled))
    
    printf "\r[INFO] Progress: ["
    printf "%${filled}s" | tr ' ' '#'
    printf "%${empty}s" | tr ' ' '-'
    printf "] %d%% (%d/%d) - %s" $percentage $current $total "$item_name"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

print_warning "This will completely remove all previous installations!"
print_warning "All data, packages, and configurations will be deleted!"
echo

# Count items to be deleted
print_status "Counting items to be removed..."
total_items=$(count_items_to_delete)
print_status "Found $total_items items to remove"

read -p "Are you sure you want to continue? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy](es)?$ ]]; then
    print_error "Installation cancelled"
    exit 1
fi

echo
print_status "Starting cleanup process..."
current_item=0

# Stop any running services
current_item=$((current_item + 1))
show_progress $current_item $total_items "Stopping existing services"
sudo systemctl stop trading-algorithm 2>/dev/null || true
sudo systemctl disable trading-algorithm 2>/dev/null || true
sudo systemctl stop nginx 2>/dev/null || true
echo

# Remove systemd service
current_item=$((current_item + 1))
show_progress $current_item $total_items "Removing systemd service"
sudo rm -f /etc/systemd/system/trading-algorithm.service
sudo systemctl daemon-reload
echo

# Remove nginx configuration
current_item=$((current_item + 1))
show_progress $current_item $total_items "Removing nginx configuration"
sudo rm -f /etc/nginx/conf.d/trading-algorithm.conf
echo

# Complete cleanup of all previous installations
current_item=$((current_item + 1))
show_progress $current_item $total_items "Removing application directories"
sudo rm -rf /home/ec2-user/trading-algorithm
sudo rm -rf /home/ec2-user/flask_web
echo

# Clean cache directories
current_item=$((current_item + 1))
show_progress $current_item $total_items "Cleaning cache directories"
sudo rm -rf /tmp/*
sudo rm -rf /var/tmp/*
sudo rm -rf /var/cache/yum/*
sudo rm -rf /var/cache/dnf/*
sudo rm -rf /var/log/*.old
sudo rm -rf /var/log/*.gz
sudo rm -rf /var/cache/man/*
sudo rm -rf /var/cache/fontconfig/*
sudo rm -rf /var/cache/ldconfig/*
echo

# Clean up any Python environments
current_item=$((current_item + 1))
show_progress $current_item $total_items "Cleaning Python environments"
find /home/ec2-user -name "venv" -type d -exec rm -rf {} + 2>/dev/null || true
find /home/ec2-user -name "*.pyc" -delete 2>/dev/null || true
find /home/ec2-user -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find /home/ec2-user -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
find /home/ec2-user -name "build" -type d -exec rm -rf {} + 2>/dev/null || true
find /home/ec2-user -name "dist" -type d -exec rm -rf {} + 2>/dev/null || true
echo

# Clean pip cache globally
current_item=$((current_item + 1))
show_progress $current_item $total_items "Cleaning pip cache"
pip cache purge 2>/dev/null || true
rm -rf ~/.cache/pip 2>/dev/null || true
echo

# Check available disk space
print_status "Checking available disk space after cleanup..."
df -h /

# Now run the fresh installation
print_status "Starting fresh installation..."
cd /home/ec2-user
chmod +x flask_web/ec2_setup_amazon_linux.sh
./flask_web/ec2_setup_amazon_linux.sh

print_success "Clean installation completed!" 