@echo off
setlocal enabledelayedexpansion

echo ================================================================================
echo                    TRADING ALGORITHM - EC2 DEPLOYMENT SCRIPT
echo ================================================================================
echo.
echo This script will deploy your trading algorithm to AWS EC2
echo.
echo Prerequisites:
echo - AWS CLI installed and configured
echo - EC2 instance running with Python 3.9+
echo - Security group with ports 22 (SSH) and 80/443 (HTTP/HTTPS) open
echo.

:: Configuration
set "INSTANCE_ID="
set "KEY_NAME="
set "REGION=us-west-2"
set "APP_NAME=trading-algorithm"
set "DEPLOY_USER=ubuntu"

:: Get user input
echo Please provide the following information:
echo.
set /p "INSTANCE_ID=EC2 Instance ID: "
set /p "KEY_NAME=EC2 Key Pair Name: "
set /p "REGION=AWS Region (default: us-west-2): "
if "!REGION!"=="" set "REGION=us-west-2"

echo.
echo ================================================================================
echo                           DEPLOYMENT CONFIGURATION
echo ================================================================================
echo Instance ID: %INSTANCE_ID%
echo Key Pair: %KEY_NAME%
echo Region: %REGION%
echo App Name: %APP_NAME%
echo Deploy User: %DEPLOY_USER%
echo.

:: Confirm deployment
set /p "CONFIRM=Proceed with deployment? (y/N): "
if /i not "!CONFIRM!"=="y" (
    echo Deployment cancelled.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo                           STEP 1: PREPARING DEPLOYMENT
echo ================================================================================

:: Create deployment directory
if not exist "ec2_deployment" mkdir "ec2_deployment"
cd "ec2_deployment"

:: Copy application files
echo [INFO] Copying application files...
xcopy "..\flask_app.py" "." /Y >nul
xcopy "..\requirements.txt" "." /Y >nul
xcopy "..\wsgi.py" "." /Y >nul
xcopy "..\gunicorn.conf.py" "." /Y >nul
xcopy "..\config.py" "." /Y >nul
xcopy "..\Dockerfile" "." /Y >nul

:: Copy templates directory
if not exist "templates" mkdir "templates"
xcopy "..\templates\*" "templates\" /E /Y >nul

:: Copy orchestrator files
if not exist "_2_Orchestrator_And_ML_Python" mkdir "_2_Orchestrator_And_ML_Python"
xcopy "..\..\..\_2_Orchestrator_And_ML_Python\*" "_2_Orchestrator_And_ML_Python\" /E /Y >nul

:: Create systemd service file
echo [INFO] Creating systemd service file...
(
echo [Unit]
echo Description=Trading Algorithm Flask App
echo After=network.target
echo.
echo [Service]
echo Type=simple
echo User=%DEPLOY_USER%
echo WorkingDirectory=/home/%DEPLOY_USER%/trading-algorithm
echo Environment=PATH=/home/%DEPLOY_USER%/trading-algorithm/venv/bin
echo ExecStart=/home/%DEPLOY_USER%/trading-algorithm/venv/bin/gunicorn --config gunicorn.conf.py wsgi:app
echo Restart=always
echo RestartSec=10
echo.
echo [Install]
echo WantedBy=multi-user.target
) > trading-algorithm.service

:: Create nginx configuration
echo [INFO] Creating nginx configuration...
(
echo server {
echo     listen 80;
echo     server_name _;
echo.
echo     location / {
echo         proxy_pass http://127.0.0.1:8000;
echo         proxy_set_header Host $host;
echo         proxy_set_header X-Real-IP $remote_addr;
echo         proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
echo         proxy_set_header X-Forwarded-Proto $scheme;
echo     }
echo.
echo     location /socket.io {
echo         proxy_pass http://127.0.0.1:8000;
echo         proxy_http_version 1.1;
echo         proxy_set_header Upgrade $http_upgrade;
echo         proxy_set_header Connection "upgrade";
echo         proxy_set_header Host $host;
echo         proxy_set_header X-Real-IP $remote_addr;
echo         proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
echo         proxy_set_header X-Forwarded-Proto $scheme;
echo     }
echo }
) > nginx-trading-algorithm

:: Create deployment script
echo [INFO] Creating deployment script...
(
echo #!/bin/bash
echo set -e
echo.
echo echo "================================================================================"
echo echo "                    TRADING ALGORITHM - EC2 SETUP SCRIPT"
echo echo "================================================================================"
echo echo.
echo.
echo # Update system
echo echo "[INFO] Updating system packages..."
echo sudo apt-get update
echo sudo apt-get upgrade -y
echo.
echo # Install Python and dependencies
echo echo "[INFO] Installing Python and system dependencies..."
echo sudo apt-get install -y python3.9 python3.9-venv python3.9-dev python3-pip nginx
echo sudo apt-get install -y build-essential libssl-dev libffi-dev
echo.
echo # Create application directory
echo echo "[INFO] Setting up application directory..."
echo mkdir -p /home/%DEPLOY_USER%/trading-algorithm
echo cd /home/%DEPLOY_USER%/trading-algorithm
echo.
echo # Create virtual environment
echo echo "[INFO] Creating Python virtual environment..."
echo python3.9 -m venv venv
echo source venv/bin/activate
echo.
echo # Upgrade pip
echo echo "[INFO] Upgrading pip..."
echo pip install --upgrade pip
echo.
echo # Install Python dependencies
echo echo "[INFO] Installing Python dependencies..."
echo pip install -r requirements.txt
echo.
echo # Setup nginx
echo echo "[INFO] Configuring nginx..."
echo sudo cp nginx-trading-algorithm /etc/nginx/sites-available/trading-algorithm
echo sudo ln -sf /etc/nginx/sites-available/trading-algorithm /etc/nginx/sites-enabled/
echo sudo rm -f /etc/nginx/sites-enabled/default
echo sudo nginx -t
echo sudo systemctl restart nginx
echo sudo systemctl enable nginx
echo.
echo # Setup systemd service
echo echo "[INFO] Setting up systemd service..."
echo sudo cp trading-algorithm.service /etc/systemd/system/
echo sudo systemctl daemon-reload
echo sudo systemctl enable trading-algorithm
echo sudo systemctl start trading-algorithm
echo.
echo # Set permissions
echo echo "[INFO] Setting file permissions..."
echo sudo chown -R %DEPLOY_USER%:%DEPLOY_USER% /home/%DEPLOY_USER%/trading-algorithm
echo chmod +x /home/%DEPLOY_USER%/trading-algorithm/venv/bin/*
echo.
echo echo "[SUCCESS] Trading Algorithm deployment completed!"
echo echo "Application should be available at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
echo echo.
echo echo "Useful commands:"
echo echo "  Check status: sudo systemctl status trading-algorithm"
echo echo "  View logs: sudo journalctl -u trading-algorithm -f"
echo echo "  Restart app: sudo systemctl restart trading-algorithm"
echo echo "  Check nginx: sudo systemctl status nginx"
) > deploy_to_ec2.sh

:: Create deployment package
echo [INFO] Creating deployment package...
powershell -Command "Compress-Archive -Path * -DestinationPath trading-algorithm-ec2.zip -Force"

echo.
echo ================================================================================
echo                           STEP 2: UPLOADING TO EC2
echo ================================================================================

:: Upload files to EC2
echo [INFO] Uploading deployment package to EC2...
aws s3 cp trading-algorithm-ec2.zip s3://trading-algorithm-deployments/ --region %REGION%

:: Execute deployment on EC2
echo [INFO] Executing deployment on EC2 instance...
aws ssm send-command ^
    --instance-ids %INSTANCE_ID% ^
    --document-name "AWS-RunShellScript" ^
    --parameters commands="cd /home/%DEPLOY_USER% && aws s3 cp s3://trading-algorithm-deployments/trading-algorithm-ec2.zip . && unzip -o trading-algorithm-ec2.zip -d trading-algorithm && cd trading-algorithm && chmod +x deploy_to_ec2.sh && ./deploy_to_ec2.sh" ^
    --region %REGION%

echo.
echo ================================================================================
echo                           STEP 3: VERIFICATION
echo ================================================================================

echo [INFO] Waiting for deployment to complete...
timeout /t 30 /nobreak >nul

:: Get instance public IP
echo [INFO] Getting instance public IP...
for /f "tokens=*" %%i in ('aws ec2 describe-instances --instance-ids %INSTANCE_ID% --query "Reservations[0].Instances[0].PublicIpAddress" --output text --region %REGION%') do set "PUBLIC_IP=%%i"

echo.
echo ================================================================================
echo                           DEPLOYMENT COMPLETE
echo ================================================================================
echo.
echo Your Trading Algorithm is now deployed on EC2!
echo.
echo Application URL: http://%PUBLIC_IP%
echo.
echo SSH Access: ssh -i "%KEY_NAME%.pem" %DEPLOY_USER%@%PUBLIC_IP%
echo.
echo Useful commands:
echo   Check app status: ssh -i "%KEY_NAME%.pem" %DEPLOY_USER%@%PUBLIC_IP% "sudo systemctl status trading-algorithm"
echo   View app logs: ssh -i "%KEY_NAME%.pem" %DEPLOY_USER%@%PUBLIC_IP% "sudo journalctl -u trading-algorithm -f"
echo   Restart app: ssh -i "%KEY_NAME%.pem" %DEPLOY_USER%@%PUBLIC_IP% "sudo systemctl restart trading-algorithm"
echo.
echo ================================================================================

:: Cleanup
cd ..
echo [INFO] Cleaning up temporary files...
rmdir /s /q "ec2_deployment"

pause 