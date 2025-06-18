@echo off
echo ========================================
echo Trading Algorithm - AWS CLI Deployment
echo ========================================
echo.

echo [1/5] Checking AWS CLI installation...
aws --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: AWS CLI not found. Please install it from:
    echo https://aws.amazon.com/cli/
    pause
    exit /b 1
)
echo ✓ AWS CLI found

echo.
echo [2/5] Checking AWS credentials...
aws sts get-caller-identity >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: AWS credentials not configured.
    echo Please run: aws configure
    echo And enter your AWS Access Key ID, Secret Access Key, and region.
    pause
    exit /b 1
)
echo ✓ AWS credentials configured

echo.
echo [3/5] Creating deployment package...
if exist "deployment.zip" del "deployment.zip"
powershell -command "Compress-Archive -Path 'flask_app.py', 'wsgi.py', 'requirements.txt', 'Procfile', 'gunicorn.conf.py', 'config.py', '.ebextensions', 'templates' -DestinationPath 'deployment.zip' -Force"

if not exist "deployment.zip" (
    echo ✗ Failed to create deployment package
    pause
    exit /b 1
)
echo ✓ Deployment package created

echo.
echo [4/5] Creating S3 bucket for deployment...
set BUCKET_NAME=trading-algorithm-deployment-%RANDOM%
echo Creating bucket: %BUCKET_NAME%

aws s3 mb s3://%BUCKET_NAME% --region us-east-1
if %errorlevel% neq 0 (
    echo ✗ Failed to create S3 bucket
    pause
    exit /b 1
)
echo ✓ S3 bucket created

echo.
echo [5/5] Uploading deployment package...
aws s3 cp deployment.zip s3://%BUCKET_NAME%/deployment.zip
if %errorlevel% neq 0 (
    echo ✗ Failed to upload deployment package
    pause
    exit /b 1
)
echo ✓ Deployment package uploaded

echo.
echo ========================================
echo DEPLOYMENT READY
echo ========================================
echo.
echo Your deployment package is now available at:
echo s3://%BUCKET_NAME%/deployment.zip
echo.
echo Next steps:
echo 1. Go to AWS Elastic Beanstalk Console
echo 2. Create a new application: trading-algorithm-web
echo 3. Create environment with Python 3.9 platform
echo 4. Use the S3 URL above as your source
echo.
echo Or use the manual deployment method with deployment.zip
echo.
pause 