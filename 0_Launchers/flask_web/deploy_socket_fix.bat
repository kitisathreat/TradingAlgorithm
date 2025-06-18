@echo off
echo ========================================
echo   Trading Algorithm - WebSocket Fix
echo   Deploying to AWS Elastic Beanstalk
echo ========================================
echo.

REM Check if AWS CLI is installed
aws --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: AWS CLI is not installed or not in PATH
    echo Please install AWS CLI from: https://aws.amazon.com/cli/
    pause
    exit /b 1
)

REM Check if EB CLI is installed
eb --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: EB CLI is not installed or not in PATH
    echo Please install EB CLI with: pip install awsebcli
    pause
    exit /b 1
)

echo [INFO] Creating deployment package...
echo.

REM Create deployment directory
if exist deployment_temp rmdir /s /q deployment_temp
mkdir deployment_temp

REM Copy Flask application files
echo [INFO] Copying Flask application files...
xcopy /s /y flask_app.py deployment_temp\
xcopy /s /y wsgi.py deployment_temp\
xcopy /s /y requirements.txt deployment_temp\
xcopy /s /y gunicorn.conf.py deployment_temp\
xcopy /s /y Procfile deployment_temp\
xcopy /s /y config.py deployment_temp\

REM Copy configuration files
echo [INFO] Copying configuration files...
xcopy /s /y .ebextensions deployment_temp\.ebextensions\

REM Copy templates
echo [INFO] Copying templates...
xcopy /s /y templates deployment_temp\templates\

REM Copy orchestrator files
echo [INFO] Copying orchestrator files...
if exist "..\..\_2_Orchestrator_And_ML_Python" (
    xcopy /s /y "..\..\_2_Orchestrator_And_ML_Python" deployment_temp\_2_Orchestrator_And_ML_Python\
) else (
    echo [WARNING] Orchestrator directory not found, skipping...
)

REM Create deployment ZIP
echo [INFO] Creating deployment ZIP...
cd deployment_temp
powershell -command "Compress-Archive -Path * -DestinationPath ..\deployment_websocket_fix.zip -Force"
cd ..

echo [INFO] Deployment package created: deployment_websocket_fix.zip
echo.

REM Deploy to AWS
echo [INFO] Deploying to AWS Elastic Beanstalk...
echo [INFO] This may take several minutes...
echo.

REM Get current environment name
for /f "tokens=*" %%i in ('eb list --output table ^| findstr "tradingalgorithm"') do set ENV_NAME=%%i
set ENV_NAME=%ENV_NAME: =%

if "%ENV_NAME%"=="" (
    echo [ERROR] Could not find trading algorithm environment
    echo Please run 'eb list' to see available environments
    pause
    exit /b 1
)

echo [INFO] Deploying to environment: %ENV_NAME%

REM Deploy using EB CLI
eb deploy %ENV_NAME% --staged

if errorlevel 1 (
    echo.
    echo [ERROR] Deployment failed!
    echo Please check the logs above for details.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Deployment completed successfully!
echo.
echo [INFO] The application should be available at:
echo [INFO] http://tradingalgorithm-env.eba-dppmvzbf.us-west-2.elasticbeanstalk.com/
echo.
echo [INFO] WebSocket connections should now work properly.
echo [INFO] Check the logs if you experience any issues.
echo.

REM Clean up
echo [INFO] Cleaning up temporary files...
rmdir /s /q deployment_temp

echo.
echo [DONE] Deployment process completed.
pause 