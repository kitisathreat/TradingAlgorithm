# EC2 Deployment Guide for Trading Algorithm

## Prerequisites
- EC2 instance running (Ubuntu 20.04+ recommended)
- SSH access to your instance
- Your EC2 key pair (.pem file)

## Step 1: Connect to Your EC2 Instance

```bash
ssh -i "your-key.pem" ubuntu@your-instance-public-ip
```

## Step 2: Upload Your Application Files

### Option A: Using SCP (from your local machine)
```bash
# From your local machine, navigate to the flask_web directory
cd path/to/TradingAlgorithm/0_Launchers/flask_web

# Upload files to EC2
scp -i "your-key.pem" -r . ubuntu@your-instance-public-ip:/home/ubuntu/
```

### Option B: Using Git (if your code is in a repository)
```bash
# On your EC2 instance
cd /home/ubuntu
git clone your-repository-url
cd your-repository/0_Launchers/flask_web
```

## Step 3: Run the Setup Script

```bash
# Make the script executable
chmod +x ec2_setup.sh

# Run the setup script
./ec2_setup.sh
```

The setup script will:
- Update system packages
- Install Python 3.9 and dependencies
- Install nginx and other required software
- Create a Python virtual environment
- Install your Python dependencies (including TensorFlow)
- Configure nginx as a reverse proxy
- Set up systemd service for automatic startup
- Start your application

## Step 4: Verify Deployment

### Check Application Status
```bash
sudo systemctl status trading-algorithm
```

### View Application Logs
```bash
sudo journalctl -u trading-algorithm -f
```

### Check Nginx Status
```bash
sudo systemctl status nginx
```

### Access Your Application
Open your web browser and go to:
```
http://your-instance-public-ip
```

## Step 5: Useful Commands

### Application Management
```bash
# Restart the application
sudo systemctl restart trading-algorithm

# Stop the application
sudo systemctl stop trading-algorithm

# Start the application
sudo systemctl start trading-algorithm

# Enable auto-start on boot
sudo systemctl enable trading-algorithm
```

### Logs and Monitoring
```bash
# View application logs
sudo journalctl -u trading-algorithm -f

# View nginx access logs
sudo tail -f /var/log/nginx/access.log

# View nginx error logs
sudo tail -f /var/log/nginx/error.log

# Monitor system resources
htop

# Check disk space
df -h

# Check memory usage
free -h
```

### File Management
```bash
# Navigate to application directory
cd /home/ubuntu/trading-algorithm

# Activate virtual environment
source venv/bin/activate

# Install new dependencies
pip install new-package

# Update application files
# (upload new files and restart the service)
```

## Troubleshooting

### Application Won't Start
```bash
# Check detailed status
sudo systemctl status trading-algorithm

# View recent logs
sudo journalctl -u trading-algorithm --since "5 minutes ago"

# Check if port 8000 is in use
sudo netstat -tlnp | grep :8000
```

### Nginx Issues
```bash
# Test nginx configuration
sudo nginx -t

# Restart nginx
sudo systemctl restart nginx

# Check nginx error logs
sudo tail -f /var/log/nginx/error.log
```

### TensorFlow Installation Issues
```bash
# Check disk space
df -h

# Check Python version
python3.9 --version

# Reinstall TensorFlow
source venv/bin/activate
pip uninstall tensorflow
pip install tensorflow==2.13.0
```

### Permission Issues
```bash
# Fix ownership
sudo chown -R ubuntu:ubuntu /home/ubuntu/trading-algorithm

# Fix permissions
chmod +x /home/ubuntu/trading-algorithm/venv/bin/*
```

## Security Considerations

### Update Security Group
Ensure your EC2 security group allows:
- Port 22 (SSH) - from your IP only
- Port 80 (HTTP) - from anywhere
- Port 443 (HTTPS) - from anywhere (if using SSL)

### SSL Certificate (Optional)
For production use, consider adding SSL:
```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

## Backup and Updates

### Backup Your Application
```bash
# Create backup
tar -czf trading-algorithm-backup-$(date +%Y%m%d).tar.gz /home/ubuntu/trading-algorithm

# Download backup to local machine
scp -i "your-key.pem" ubuntu@your-instance-public-ip:/home/ubuntu/trading-algorithm-backup-*.tar.gz .
```

### Update Application
```bash
# Stop the application
sudo systemctl stop trading-algorithm

# Upload new files
# (use scp or git pull)

# Restart the application
sudo systemctl start trading-algorithm
```

## Performance Optimization

### Increase Swap Space (if needed)
```bash
# Check current swap
free -h

# Create swap file
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Monitor Resource Usage
```bash
# Install monitoring tools
sudo apt-get install htop iotop

# Monitor in real-time
htop
```

## Cost Optimization

### Stop Instance When Not in Use
```bash
# From AWS Console or CLI
aws ec2 stop-instances --instance-ids your-instance-id
```

### Use Spot Instances (for development)
Consider using spot instances for cost savings during development and testing.

## Support

If you encounter issues:
1. Check the logs first
2. Verify all prerequisites are met
3. Ensure sufficient disk space (40GB+ recommended)
4. Check network connectivity
5. Verify security group settings 