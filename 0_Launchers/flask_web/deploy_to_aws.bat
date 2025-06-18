@echo off
echo ========================================
echo Trading Algorithm - AWS Deployment
echo ========================================
echo.

echo [1/5] Checking AWS CLI installation...
aws --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: AWS CLI is not installed or not in PATH
    echo Please install AWS CLI from: https://aws.amazon.com/cli/
    pause
    exit /b 1
)
echo [OK] AWS CLI found

echo [2/5] Checking AWS credentials...
aws sts get-caller-identity >nul 2>&1
if errorlevel 1 (
    echo ERROR: AWS credentials not configured
    echo Please run: aws configure
    pause
    exit /b 1
)
echo [OK] AWS credentials configured

echo [3/5] Creating deployment package...
if exist deployment_fixed.zip del deployment_fixed.zip

echo [4/5] Adding files to deployment package...
powershell -Command "Compress-Archive -Path 'flask_app.py', 'wsgi.py', 'requirements.txt', 'Procfile', 'gunicorn.conf.py', 'templates', 'config.py' -DestinationPath 'deployment_fixed.zip' -Force"

if not exist deployment_fixed.zip (
    echo ERROR: Failed to create deployment package
    pause
    exit /b 1
)
echo [OK] Deployment package created: deployment_fixed.zip

echo [5/5] Deploying to AWS Elastic Beanstalk...
echo.
echo Please enter your Elastic Beanstalk environment details:
echo.

set /p EB_APP_NAME="Enter Application Name (e.g., trading-algorithm): "
set /p EB_ENV_NAME="Enter Environment Name (e.g., tradingalgorithm-env): "
set /p EB_REGION="Enter AWS Region (e.g., us-west-2): "

echo.
echo Deploying to:
echo Application: %EB_APP_NAME%
echo Environment: %EB_ENV_NAME%
echo Region: %EB_REGION%
echo.

echo Starting deployment...
aws elasticbeanstalk create-application-version ^
    --application-name "%EB_APP_NAME%" ^
    --version-label "v%date:~-4,4%%date:~-10,2%%date:~-7,2%-%time:~0,2%%time:~3,2%%time:~6,2%" ^
    --source-bundle S3Bucket="elasticbeanstalk-%EB_REGION%-%EB_APP_NAME%",S3Key="deployment_fixed.zip" ^
    --region %EB_REGION%

if errorlevel 1 (
    echo ERROR: Failed to create application version
    echo Trying alternative deployment method...
    echo.
    echo Please manually upload deployment_fixed.zip to your EB environment
    echo.
    pause
    exit /b 1
)

echo [OK] Application version created successfully
echo.
echo Deployment completed! Your application should be updating now.
echo Check your Elastic Beanstalk console for deployment status.
echo.
echo Recent updates included:
echo - Enhanced SocketIO configuration for production deployment
echo - Improved error handling and logging
echo - Optimized Gunicorn settings for web applications
echo - Updated dependencies for better compatibility
echo.

pause 