@echo off
echo ================================================================================
echo              FLASK APP STATUS CHECK - EC2 INSTANCE
echo ================================================================================
echo Started at: %date% %time%
echo ================================================================================
echo.

REM Check if we have the EC2 configuration
if not exist "ec2_config.txt" (
    echo [ERROR] ec2_config.txt not found. Please run the deployment first.
    pause
    exit /b 1
)

REM Read EC2 configuration
for /f "tokens=1,2 delims==" %%a in (ec2_config.txt) do (
    if "%%a"=="PUBLIC_IP" set EC2_IP=%%b
    if "%%a"=="KEY_FILE" set KEY_FILE=%%b
    if "%%a"=="DEPLOY_USER" set USER=%%b
)

if "%EC2_IP%"=="" (
    echo [ERROR] PUBLIC_IP not found in ec2_config.txt
    pause
    exit /b 1
)

if "%KEY_FILE%"=="" (
    echo [ERROR] KEY_FILE not found in ec2_config.txt
    pause
    exit /b 1
)

if "%USER%"=="" set USER=ec2-user

echo [INFO] Connecting to EC2 instance: %USER%@%EC2_IP%
echo [INFO] Using key file: %KEY_FILE%
echo.

REM Upload the status check script to the EC2 instance
echo [INFO] Uploading status check script...
scp -i "%KEY_FILE%" -o StrictHostKeyChecking=no check_app_status.sh %USER%@%EC2_IP%:~/check_app_status.sh
if errorlevel 1 (
    echo [ERROR] Failed to upload check_app_status.sh to EC2 instance.
    pause
    exit /b 1
)

echo [INFO] Running status check on EC2 instance...
echo.

REM Run the status check script on EC2
ssh -i "%KEY_FILE%" -o StrictHostKeyChecking=no %USER%@%EC2_IP% "chmod +x ~/check_app_status.sh && bash ~/check_app_status.sh"

echo.
echo [INFO] Status check completed. Check the output above.
echo.
echo [INFO] If the app is running but stock data still doesn't load:
echo   - Try accessing the web interface directly at http://35.89.15.141
echo   - Check if there are any JavaScript errors in the browser console
echo   - The issue might be in the frontend, not the backend
echo.
pause 