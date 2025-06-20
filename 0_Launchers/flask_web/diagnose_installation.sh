#!/bin/bash
# Trading Algorithm Installation Diagnostic Script
# This script checks the current state of the installation and identifies issues

set -e

echo "================================================================================"
echo "              TRADING ALGORITHM - INSTALLATION DIAGNOSTIC"
echo "================================================================================"
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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if file exists
file_exists() {
    [ -f "$1" ]
}

# Function to check if directory exists
dir_exists() {
    [ -d "$1" ]
}

# Function to check service status
check_service() {
    local service="$1"
    if sudo systemctl is-active --quiet "$service" 2>/dev/null; then
        print_success "$service is running"
        return 0
    elif sudo systemctl is-enabled --quiet "$service" 2>/dev/null; then
        print_warning "$service is enabled but not running"
        return 1
    else
        print_error "$service is not installed or not enabled"
        return 2
    fi
}

# Detect the correct user and home directory
CURRENT_USER=$(whoami)
USER_HOME=$(eval echo ~$CURRENT_USER)

print_status "System Information:"
echo "  Current user: $CURRENT_USER"
echo "  User home: $USER_HOME"
echo "  Operating system: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "  Kernel: $(uname -r)"
echo "  Architecture: $(uname -m)"
echo

# Check Python installations
print_status "Checking Python installations..."
PYTHON_VERSIONS=("python3.9" "python3.8" "python3" "python")
PYTHON_FOUND=false

for py in "${PYTHON_VERSIONS[@]}"; do
    if command_exists "$py"; then
        VERSION=$($py --version 2>&1 | cut -d' ' -f2)
        print_success "$py found: $VERSION at $(which $py)"
        PYTHON_FOUND=true
        break
    fi
done

if [ "$PYTHON_FOUND" = false ]; then
    print_error "No Python installation found"
fi

# Check pip
if command_exists pip; then
    PIP_VERSION=$(pip --version | cut -d' ' -f2)
    print_success "pip found: $PIP_VERSION at $(which pip)"
else
    print_error "pip not found"
fi

echo

# Check for application directories
print_status "Checking application directories..."
POSSIBLE_DIRS=(
    "$USER_HOME/trading-algorithm"
    "/home/ec2-user/trading-algorithm"
    "/home/ubuntu/trading-algorithm"
    "$USER_HOME/flask_web"
    "/home/ec2-user/flask_web"
    "/home/ubuntu/flask_web"
)

APP_DIR=""
for dir in "${POSSIBLE_DIRS[@]}"; do
    if dir_exists "$dir"; then
        print_success "Found application directory: $dir"
        APP_DIR="$dir"
        break
    fi
done

if [ -z "$APP_DIR" ]; then
    print_error "No application directory found!"
    echo "  Searched in:"
    for dir in "${POSSIBLE_DIRS[@]}"; do
        echo "    $dir"
    done
else
    echo "  Application directory: $APP_DIR"
    
    # Check application files
    print_status "Checking application files..."
    REQUIRED_FILES=("flask_app.py" "requirements.txt" "wsgi.py" "gunicorn.conf.py")
    for file in "${REQUIRED_FILES[@]}"; do
        if file_exists "$APP_DIR/$file"; then
            print_success "Found: $file"
        else
            print_error "Missing: $file"
        fi
    done
    
    # Check virtual environment
    print_status "Checking virtual environment..."
    if dir_exists "$APP_DIR/flask_web/venv"; then
        print_success "Virtual environment found at: $APP_DIR/flask_web/venv"
        
        # Check if virtual environment is activated
        if [ "$VIRTUAL_ENV" = "$APP_DIR/flask_web/venv" ]; then
            print_success "Virtual environment is currently activated"
        else
            print_warning "Virtual environment is not activated"
        fi
        
        # Check gunicorn in virtual environment
        if file_exists "$APP_DIR/flask_web/venv/bin/gunicorn"; then
            GUNICORN_VERSION=$("$APP_DIR/flask_web/venv/bin/gunicorn" --version 2>/dev/null | cut -d' ' -f1 || echo "unknown")
            print_success "Gunicorn found at: $APP_DIR/flask_web/venv/bin/gunicorn (Version: $GUNICORN_VERSION)"
        else
            print_error "Gunicorn not found in virtual environment!"
        fi
        
        # Check other key packages
        print_status "Checking key packages in virtual environment..."
        cd "$APP_DIR"
        source flask_web/venv/bin/activate 2>/dev/null || true
        
        KEY_PACKAGES=("flask" "tensorflow" "numpy" "pandas" "yfinance" "scikit-learn")
        for package in "${KEY_PACKAGES[@]}"; do
            if python -c "import $package" 2>/dev/null; then
                VERSION=$(python -c "import $package; print($package.__version__)" 2>/dev/null || echo "unknown")
                print_success "$package found (Version: $VERSION)"
            else
                print_error "$package not found"
            fi
        done
        
        # Check total packages installed
        PACKAGE_COUNT=$(pip list | wc -l)
        print_status "Total packages installed: $((PACKAGE_COUNT - 2))"  # Subtract header lines
        
    else
        print_error "Virtual environment not found at: $APP_DIR/flask_web/venv"
    fi
fi

echo

# Check system services
print_status "Checking system services..."
check_service "nginx"
check_service "trading-algorithm"

# Check nginx configuration
if command_exists nginx; then
    NGINX_VERSION=$(nginx -v 2>&1 | cut -d'/' -f2)
    print_success "nginx found: $NGINX_VERSION at $(which nginx)"
    
    if sudo nginx -t 2>/dev/null; then
        print_success "nginx configuration is valid"
    else
        print_error "nginx configuration has errors"
        sudo nginx -t
    fi
    
    # Check nginx sites
    if file_exists "/etc/nginx/sites-enabled/trading-algorithm"; then
        print_success "nginx trading-algorithm site is enabled"
    else
        print_warning "nginx trading-algorithm site is not enabled"
    fi
else
    print_error "nginx not found"
fi

echo

# Check systemd service file
print_status "Checking systemd service configuration..."
if file_exists "/etc/systemd/system/trading-algorithm.service"; then
    print_success "systemd service file found"
    
    # Check service configuration
    SERVICE_USER=$(grep "^User=" /etc/systemd/system/trading-algorithm.service | cut -d'=' -f2)
    SERVICE_WORKDIR=$(grep "^WorkingDirectory=" /etc/systemd/system/trading-algorithm.service | cut -d'=' -f2)
    SERVICE_EXEC=$(grep "^ExecStart=" /etc/systemd/system/trading-algorithm.service | cut -d'=' -f2)
    
    echo "  Service user: $SERVICE_USER"
    echo "  Working directory: $SERVICE_WORKDIR"
    echo "  ExecStart: $SERVICE_EXEC"
    
    # Check if paths match
    if [ "$SERVICE_USER" != "$CURRENT_USER" ]; then
        print_warning "Service user ($SERVICE_USER) doesn't match current user ($CURRENT_USER)"
    fi
    
    if [ "$SERVICE_WORKDIR" != "$APP_DIR" ] && [ -n "$APP_DIR" ]; then
        print_warning "Service working directory ($SERVICE_WORKDIR) doesn't match app directory ($APP_DIR)"
    fi
    
    # Check if gunicorn path exists
    GUNICORN_PATH=$(echo "$SERVICE_EXEC" | awk '{print $1}')
    if [ -n "$GUNICORN_PATH" ] && [ ! -f "$GUNICORN_PATH" ]; then
        print_error "Gunicorn executable not found at: $GUNICORN_PATH"
    elif [ -n "$GUNICORN_PATH" ]; then
        print_success "Gunicorn executable found at: $GUNICORN_PATH"
    fi
else
    print_error "systemd service file not found"
fi

echo

# Check network and ports
print_status "Checking network configuration..."
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "unknown")
echo "  Public IP: $PUBLIC_IP"

# Check if ports are listening
if command_exists netstat; then
    if netstat -tlnp 2>/dev/null | grep -q ":80 "; then
        print_success "Port 80 is listening (nginx)"
    else
        print_warning "Port 80 is not listening"
    fi
    
    if netstat -tlnp 2>/dev/null | grep -q ":8000 "; then
        print_success "Port 8000 is listening (gunicorn)"
    else
        print_warning "Port 8000 is not listening"
    fi
fi

echo

# Check disk space
print_status "Checking disk space..."
df -h | grep -E "(Filesystem|/$)" | while read line; do
    echo "  $line"
done

echo

# Check recent logs
print_status "Checking recent service logs..."
if sudo journalctl -u trading-algorithm --no-pager -n 5 2>/dev/null | grep -q .; then
    echo "  Recent trading-algorithm logs:"
    sudo journalctl -u trading-algorithm --no-pager -n 3 2>/dev/null | sed 's/^/    /'
else
    print_warning "No recent trading-algorithm logs found"
fi

echo

# Test application manually
print_status "Testing application manually..."
if [ -n "$APP_DIR" ] && file_exists "$APP_DIR/flask_web/venv/bin/gunicorn"; then
    cd "$APP_DIR"
    source flask_web/venv/bin/activate 2>/dev/null || true
    
    # Test if gunicorn can start
    timeout 5s ./flask_web/venv/bin/gunicorn --config gunicorn.conf.py wsgi:app &
    GUNICORN_PID=$!
    sleep 2
    
    if kill -0 $GUNICORN_PID 2>/dev/null; then
        print_success "Application test successful"
        kill $GUNICORN_PID
        wait $GUNICORN_PID 2>/dev/null || true
    else
        print_error "Application test failed"
    fi
else
    print_warning "Cannot test application - missing gunicorn or app directory"
fi

echo

# Summary and recommendations
echo "================================================================================"
print_status "DIAGNOSTIC SUMMARY"
echo "================================================================================"

if [ -n "$APP_DIR" ]; then
    echo "Application directory: $APP_DIR"
    
    if dir_exists "$APP_DIR/flask_web/venv" && file_exists "$APP_DIR/flask_web/venv/bin/gunicorn"; then
        print_success "✓ Virtual environment and gunicorn are properly installed"
    else
        print_error "✗ Virtual environment or gunicorn is missing"
        echo "  Recommendation: Run the clean install script"
    fi
    
    if check_service "trading-algorithm" >/dev/null; then
        print_success "✓ Trading algorithm service is running"
    else
        print_error "✗ Trading algorithm service is not running properly"
        echo "  Recommendation: Check service logs and restart"
    fi
    
    if check_service "nginx" >/dev/null; then
        print_success "✓ Nginx is running"
    else
        print_error "✗ Nginx is not running properly"
        echo "  Recommendation: Check nginx configuration and restart"
    fi
    
    echo
    echo "Application should be accessible at: http://$PUBLIC_IP"
    echo
    echo "Useful commands:"
    echo "  Check app status: sudo systemctl status trading-algorithm"
    echo "  View app logs: sudo journalctl -u trading-algorithm -f"
    echo "  Restart app: sudo systemctl restart trading-algorithm"
    echo "  Check nginx: sudo systemctl status nginx"
    echo "  View nginx logs: sudo tail -f /var/log/nginx/access.log"
    echo "  SSH access: ssh -i your-key.pem $CURRENT_USER@$PUBLIC_IP"
else
    print_error "✗ No application directory found"
    echo "  Recommendation: Run the deployment script to install the application"
fi

echo "================================================================================" 