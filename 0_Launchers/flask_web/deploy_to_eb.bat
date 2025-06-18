@echo off
echo ========================================
echo Trading Algorithm - EB Deployment Prep
echo ========================================
echo.

echo [1/4] Checking current directory...
if not exist "flask_app.py" (
    echo ERROR: flask_app.py not found. Please run this script from the flask_web directory.
    pause
    exit /b 1
)
echo ✓ Found Flask application

echo.
echo [2/4] Creating deployment package...
if exist "deployment.zip" del "deployment.zip"
echo Creating ZIP file with all necessary files...

REM Create ZIP with all required files for EB deployment
powershell -command "Compress-Archive -Path 'flask_app.py', 'wsgi.py', 'requirements.txt', 'Procfile', 'gunicorn.conf.py', 'config.py', '.ebextensions', 'templates' -DestinationPath 'deployment.zip' -Force"

if exist "deployment.zip" (
    echo ✓ Deployment package created: deployment.zip
) else (
    echo ✗ Failed to create deployment package
    pause
    exit /b 1
)

echo.
echo [3/4] Validating configuration files...
if exist ".ebextensions\01_flask.config" (
    echo ✓ EB configuration found
) else (
    echo ✗ Missing .ebextensions\01_flask.config
)

if exist "Procfile" (
    echo ✓ Procfile found
) else (
    echo ✗ Missing Procfile
)

if exist "requirements.txt" (
    echo ✓ Requirements file found
) else (
    echo ✗ Missing requirements.txt
)

echo.
echo [4/4] Deployment Instructions
echo ========================================
echo.
echo MANUAL DEPLOYMENT STEPS:
echo.
echo 1. Go to AWS Console: https://console.aws.amazon.com/elasticbeanstalk/
echo.
echo 2. Click "Create Application"
echo    - Application name: trading-algorithm-web
echo    - Platform: Python
echo    - Platform branch: Python 3.9
echo    - Platform version: 3.9.16 (latest)
echo.
echo 3. Click "Configure more options" and set:
echo    - Environment type: Single instance (free tier)
echo    - Instance type: t3.micro (free tier) or t3.medium
echo.
echo 4. Click "Create environment"
echo.
echo 5. Once environment is created, go to "Upload and deploy"
echo.
echo 6. Upload the deployment.zip file created in this directory
echo.
echo 7. Click "Deploy"
echo.
echo 8. Wait for deployment to complete (5-10 minutes)
echo.
echo 9. Your app will be available at the provided URL
echo.
echo ========================================
echo.
echo Files included in deployment.zip:
powershell -command "Get-ChildItem -Path 'deployment.zip' | Select-Object Name, Length | Format-Table -AutoSize"

echo.
echo Ready for deployment! Follow the manual steps above.
echo.
pause 