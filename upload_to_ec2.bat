@echo off
echo ================================================================================
echo                    TRADING ALGORITHM - EC2 FILE UPLOAD SCRIPT
echo ================================================================================
echo.

echo [INFO] Starting file upload to EC2 instance...
echo [INFO] EC2 Instance: 35.89.15.141
echo [INFO] Key File: C:\Users\KitKumar\Downloads\KKumar06202025.pem
echo.

REM Upload all flask_web files
echo [INFO] Uploading Flask web application files...
scp -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" -r "0_Launchers\flask_web\*" ec2-user@35.89.15.141:/home/ec2-user/flask_web/

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to upload Flask web files
    pause
    exit /b 1
)

REM Upload templates directory
echo [INFO] Uploading templates directory...
scp -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" -r "0_Launchers\flask_web\templates" ec2-user@35.89.15.141:/home/ec2-user/flask_web/

if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Failed to upload templates directory (may already exist)
)

REM Upload ML orchestrator files
echo [INFO] Uploading ML orchestrator files...
scp -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" -r "_2_Orchestrator_And_ML_Python" ec2-user@35.89.15.141:/home/ec2-user/flask_web/

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to upload ML orchestrator files
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo [SUCCESS] All files uploaded successfully!
echo ================================================================================
echo.
echo Next steps:
echo 1. SSH into your EC2 instance:
echo    ssh -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" ec2-user@35.89.15.141
echo.
echo 2. Navigate to flask_web directory:
echo    cd /home/ec2-user/flask_web
echo.
echo 3. Run the setup script:
echo    chmod +x ec2_setup_amazon_linux.sh
echo    ./ec2_setup_amazon_linux.sh
echo.
echo ================================================================================
pause 