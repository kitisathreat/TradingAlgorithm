@echo off
REM Trading Algorithm EC2 Fix Script
REM This script helps fix deployment issues on EC2

echo ================================================================================
echo              TRADING ALGORITHM - EC2 FIX SCRIPT
echo ================================================================================
echo.

REM Try to read configuration from a file first
set "CONFIG_FILE=ec2_config.txt"
set "INSTANCE_ID="
set "KEY_NAME="
set "KEY_FILE="
set "PUBLIC_IP="
set "REGION=us-west-2"
set "DEPLOY_USER=ubuntu"

REM Check if config file exists
if exist "%CONFIG_FILE%" (
    echo [INFO] Reading configuration from %CONFIG_FILE%...
    for /f "tokens=1,2 delims==" %%a in (%CONFIG_FILE%) do (
        if "%%a"=="INSTANCE_ID" set "INSTANCE_ID=%%b"
        if "%%a"=="KEY_NAME" set "KEY_NAME=%%b"
        if "%%a"=="KEY_FILE" set "KEY_FILE=%%b"
        if "%%a"=="PUBLIC_IP" set "PUBLIC_IP=%%b"
        if "%%a"=="REGION" set "REGION=%%b"
        if "%%a"=="DEPLOY_USER" set "DEPLOY_USER=%%b"
    )
)

REM If not found in config file, try to get from AWS CLI
if "%INSTANCE_ID%"=="" (
    echo [INFO] Trying to get instance ID from AWS CLI...
    for /f "tokens=*" %%i in ('aws ec2 describe-instances --filters "Name=instance-state-name,Values=running" "Name=tag:Name,Values=trading-algorithm*" --query "Reservations[0].Instances[0].InstanceId" --output text --region %REGION% 2^>nul') do set "INSTANCE_ID=%%i"
)

REM If still not found, ask user
if "%INSTANCE_ID%"=="" (
    echo [WARNING] Could not automatically detect EC2 instance.
    set /p INSTANCE_ID="Enter EC2 instance ID: "
)

REM Try to find key file automatically if not in config
if "%KEY_FILE%"=="" (
    echo [INFO] Looking for key files...
    for %%f in (*.pem) do (
        set "KEY_FILE=%%f"
        goto :found_key
    )
    for %%f in (..\*.pem) do (
        set "KEY_FILE=%%f"
        goto :found_key
    )
    for %%f in (..\..\*.pem) do (
        set "KEY_FILE=%%f"
        goto :found_key
    )
)

:found_key
if "%KEY_FILE%"=="" (
    echo [WARNING] Could not automatically find key file.
    set /p KEY_FILE="Enter path to your .pem key file: "
)

if not exist "%KEY_FILE%" (
    echo ERROR: Key file not found at %KEY_FILE%
    pause
    exit /b 1
)

REM Get public IP from config or AWS CLI
if "%PUBLIC_IP%"=="" (
    echo [INFO] Getting instance public IP from AWS CLI...
    for /f "tokens=*" %%i in ('aws ec2 describe-instances --instance-ids %INSTANCE_ID% --query "Reservations[0].Instances[0].PublicIpAddress" --output text --region %REGION% 2^>nul') do set "PUBLIC_IP=%%i"
)

if "%PUBLIC_IP%"=="" (
    echo [WARNING] Could not get public IP automatically.
    set /p PUBLIC_IP="Enter EC2 public IP address: "
)

if "%PUBLIC_IP%"=="" (
    echo ERROR: No public IP provided
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo                           FIX CONFIGURATION
echo ================================================================================
echo Instance ID: %INSTANCE_ID%
echo Public IP: %PUBLIC_IP%
echo Key File: %KEY_FILE%
echo Region: %REGION%
echo Deploy User: %DEPLOY_USER%
echo.

echo Creating deployment package with relative pathing...
REM Create a temporary directory for the deployment
if exist "temp_deploy" rmdir /s /q "temp_deploy"
mkdir "temp_deploy"

REM Copy files maintaining relative structure
echo [INFO] Copying files with relative pathing...
xcopy "flask_app.py" "temp_deploy\flask_web\" /Y >nul
xcopy "requirements.txt" "temp_deploy\flask_web\" /Y >nul
xcopy "wsgi.py" "temp_deploy\flask_web\" /Y >nul
xcopy "gunicorn.conf.py" "temp_deploy\flask_web\" /Y >nul
xcopy "config.py" "temp_deploy\flask_web\" /Y >nul
xcopy "ec2_clean_install.sh" "temp_deploy\flask_web\" /Y >nul

if exist "templates" (
    xcopy "templates" "temp_deploy\flask_web\templates\" /E /Y >nul
)

if exist "..\..\_2_Orchestrator_And_ML_Python" (
    xcopy "..\..\_2_Orchestrator_And_ML_Python" "temp_deploy\_2_Orchestrator_And_ML_Python\" /E /Y >nul
)

echo Uploading files to EC2...
scp -i "%KEY_FILE%" -r temp_deploy/* %DEPLOY_USER@%PUBLIC_IP%:~/ 2>nul
if errorlevel 1 (
    echo Trying with ec2-user...
    scp -i "%KEY_FILE%" -r temp_deploy/* ec2-user@%PUBLIC_IP%:~/ 2>nul
    if errorlevel 1 (
        echo ERROR: Failed to upload files. Check your connection.
        pause
        exit /b 1
    )
    set "DEPLOY_USER=ec2-user"
)

echo.
echo Running clean install script...
ssh -i "%KEY_FILE%" %DEPLOY_USER@%PUBLIC_IP% "cd ~/flask_web && chmod +x ec2_clean_install.sh && ./ec2_clean_install.sh"

echo.
echo Cleaning up temporary files...
rmdir /s /q "temp_deploy"

echo.
echo ================================================================================
echo Fix script completed!
echo ================================================================================
echo.
echo Your application should now be running at: http://%PUBLIC_IP%
echo.
echo To check status, SSH into your instance and run:
echo   sudo systemctl status trading-algorithm
echo   sudo journalctl -u trading-algorithm -f
echo.
echo Or run these commands directly:
echo   ssh -i "%KEY_FILE%" %DEPLOY_USER@%PUBLIC_IP% "sudo systemctl status trading-algorithm"
echo   ssh -i "%KEY_FILE%" %DEPLOY_USER@%PUBLIC_IP% "sudo journalctl -u trading-algorithm -f"
echo.
pause 