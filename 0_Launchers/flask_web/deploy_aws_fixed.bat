@echo off
echo ========================================
echo    AWS DEPLOYMENT - FIXED VERSION
echo ========================================
echo.
echo This deployment includes the orchestrator modules
echo to fix the model access issues.
echo.

REM Create the deployment package with orchestrator modules
echo [1/4] Creating deployment package with orchestrator modules...
python create_eb_zip.py
if %errorlevel% neq 0 (
    echo ERROR: Failed to create deployment package
    pause
    exit /b 1
)

REM Deploy to AWS Elastic Beanstalk
echo [2/4] Deploying to AWS Elastic Beanstalk...
aws elasticbeanstalk create-application-version ^
    --application-name "tradingalgorithm" ^
    --version-label "v1.0-fixed-$(date /t)" ^
    --source-bundle S3Bucket="tradingalgorithm-deployment",S3Key="deployment.zip" ^
    --auto-create-application

if %errorlevel% neq 0 (
    echo ERROR: Failed to create application version
    pause
    exit /b 1
)

echo [3/4] Updating environment...
aws elasticbeanstalk update-environment ^
    --environment-name "tradingalgorithm-env" ^
    --version-label "v1.0-fixed-$(date /t)"

if %errorlevel% neq 0 (
    echo ERROR: Failed to update environment
    pause
    exit /b 1
)

echo [4/4] Deployment completed successfully!
echo.
echo The application should be available at:
echo http://tradingalgorithm-env.eba-dppmvzbf.us-west-2.elasticbeanstalk.com/
echo.
echo NOTE: This deployment includes:
echo - Flask web application
echo - Orchestrator modules (_2_Orchestrator_And_ML_Python)
echo - Model trainer and stock selection utilities
echo - All required data files
echo.
pause 