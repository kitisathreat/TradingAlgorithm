@echo off
echo ================================================================================
echo              STOCK DATA LOADING DIAGNOSTIC - EC2 INSTANCE
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

REM Upload the diagnostic shell script to the EC2 instance
scp -i "%KEY_FILE%" -o StrictHostKeyChecking=no diagnose_stock_data.sh %USER%@%EC2_IP%:~/diagnose_stock_data.sh
if errorlevel 1 (
    echo [ERROR] Failed to upload diagnose_stock_data.sh to EC2 instance.
    pause
    exit /b 1
)

echo [INFO] Running diagnostic script on EC2 instance...
ssh -i "%KEY_FILE%" -o StrictHostKeyChecking=no %USER%@%EC2_IP% "chmod +x ~/diagnose_stock_data.sh && bash ~/diagnose_stock_data.sh"

echo.
echo [INFO] Diagnostic completed. Check the output above for issues.
echo.
pause 