@echo off
echo ================================================================================
echo                    TRADING ALGORITHM - EC2 FILE UPLOAD SCRIPT
echo ================================================================================
echo.

echo [INFO] Starting file upload to EC2 instance...
echo [INFO] EC2 Instance: 35.89.15.141
echo [INFO] Key File: C:\Users\KitKumar\Downloads\KKumar06202025.pem
echo.

REM Upload the clean installation script
echo [INFO] Uploading clean installation script...
scp -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" "0_Launchers\flask_web\ec2_clean_install.sh" ec2-user@35.89.15.141:/home/ec2-user/flask_web/
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to upload clean installation script
    pause
    exit /b 1
)

REM Upload the updated setup script first
echo [INFO] Uploading updated setup script...
scp -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" "0_Launchers\flask_web\ec2_setup_amazon_linux.sh" ec2-user@35.89.15.141:/home/ec2-user/flask_web/
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to upload updated setup script
    pause
    exit /b 1
)

REM Upload the complete requirements.txt from project root
echo [INFO] Uploading complete requirements.txt...
scp -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" "requirements.txt" ec2-user@35.89.15.141:/home/ec2-user/flask_web/
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to upload complete requirements.txt
    pause
    exit /b 1
)

REM Upload all files from flask_web (contents only, not the directory itself)
echo [INFO] Uploading all Flask web application files (including templates)...
scp -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" -r 0_Launchers/flask_web/* ec2-user@35.89.15.141:/home/ec2-user/flask_web/
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to upload Flask web files
    pause
    exit /b 1
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
echo Files uploaded include:
echo - Flask application files (flask_app.py, wsgi.py, etc.)
echo - Complete requirements.txt with all dependencies:
echo   * pandas, numpy, scikit-learn
echo   * matplotlib, plotly, seaborn
echo   * tensorflow (full version), onnxruntime
echo   * Flask-SocketIO, gunicorn, eventlet
echo   * All other necessary packages
echo - Updated setup script with all fixes:
echo   * Disk space cleanup before installation
echo   * Full TensorFlow for better performance
echo   * Curl package conflict resolution
echo   * Enhanced error handling
echo - Templates directory
echo - ML orchestrator files (_2_Orchestrator_And_ML_Python)
echo.

REM Ask user for installation preference
echo ================================================================================
echo                    INSTALLATION OPTIONS
echo ================================================================================
echo.
echo Choose your installation method:
echo.
echo [1] CLEAN INSTALL (Recommended for fresh start)
echo     - Completely removes previous installation
echo     - Cleans all caches and temporary files
echo     - Fresh virtual environment setup
echo     - Best for resolving persistent issues
echo.
echo [2] REGULAR INSTALL (Update existing installation)
echo     - Updates existing packages
echo     - Preserves existing data and configurations
echo     - Faster installation
echo.
set /p choice="Enter your choice (1 or 2): "

if "%choice%"=="1" (
    echo.
    echo ================================================================================
    echo                    STARTING CLEAN INSTALL
    echo ================================================================================
    echo.
    echo [INFO] Connecting to EC2 instance and running clean installation...
    echo [INFO] This will completely remove the previous installation and start fresh.
    echo.
    echo [WARNING] This process will:
    echo - Stop and remove existing services
    echo - Clean all caches and temporary files
    echo - Remove old virtual environments
    echo - Install fresh dependencies
    echo - Set up new services
    echo.
    echo [INFO] Starting clean installation...
    echo [INFO] Press Ctrl+C or close this window to cancel the installation.
    echo.
    ssh -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" ec2-user@35.89.15.141 "cd /home/ec2-user/flask_web && chmod +x ec2_clean_install.sh && ./ec2_clean_install.sh"
    echo.
    echo ================================================================================
    echo [INFO] Clean installation completed!
    echo ================================================================================
    echo.
    echo Next steps:
    echo 1. Check service status: sudo systemctl status trading-algorithm
    echo 2. View logs: sudo journalctl -u trading-algorithm -f
    echo 3. Access application: http://35.89.15.141
    echo.
    goto :end
) else if "%choice%"=="2" (
    echo.
    echo ================================================================================
    echo                    STARTING REGULAR INSTALL
    echo ================================================================================
    echo.
    echo [INFO] Connecting to EC2 instance and running regular installation...
    echo [INFO] This will update existing packages and restart services.
    echo.
    echo [INFO] Starting regular installation...
    ssh -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" ec2-user@35.89.15.141 "cd /home/ec2-user/flask_web && chmod +x ec2_setup_amazon_linux.sh && ./ec2_setup_amazon_linux.sh"
    echo.
    echo ================================================================================
    echo [INFO] Regular installation completed!
    echo ================================================================================
    echo.
    echo Next steps:
    echo 1. Check service status: sudo systemctl status trading-algorithm
    echo 2. View logs: sudo journalctl -u trading-algorithm -f
    echo 3. Access application: http://35.89.15.141
    echo.
) else (
    echo.
    echo [ERROR] Invalid choice. Please run the script again and select 1 or 2.
    echo.
)

:end
echo ================================================================================
pause