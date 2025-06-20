@echo off
echo ================================================================================
echo                    TRADING ALGORITHM - EC2 FILE UPLOAD SCRIPT
echo ================================================================================
echo.

echo [INFO] Starting file upload to EC2 instance...
echo [INFO] EC2 Instance: 35.89.15.141
echo [INFO] Key File: C:\Users\KitKumar\Downloads\KKumar06202025.pem
echo.

REM Ask user for installation preference first
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
echo     - Uploads all files with proper directory structure
echo.
echo [2] REGULAR INSTALL (Update existing installation)
echo     - Updates existing packages
echo     - Preserves existing data and configurations
echo     - Faster installation
echo     - Uploads only essential files (optimized)
echo.
set /p choice="Enter your choice (1 or 2): "

if "%choice%"=="1" (
    echo.
    echo ================================================================================
    echo                    STARTING CLEAN INSTALL UPLOAD
    echo ================================================================================
    echo.
    echo [INFO] Clean install selected - uploading all files with proper structure...
    goto :upload_all_files
) else if "%choice%"=="2" (
    echo.
    echo ================================================================================
    echo                    STARTING REGULAR INSTALL UPLOAD
    echo ================================================================================
    echo.
    echo [INFO] Regular install selected - uploading essential files only...
    goto :upload_essential_files
) else (
    echo.
    echo [ERROR] Invalid choice. Please run the script again and select 1 or 2.
    echo.
    pause
    exit /b 1
)

:upload_all_files
REM Clean up existing files for clean install
echo [INFO] Cleaning up existing files on EC2 for fresh install...
ssh -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" ec2-user@35.89.15.141 "rm -rf /home/ec2-user/flask_web/* /home/ec2-user/trading-algorithm 2>/dev/null || true"
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Failed to clean up some existing files, continuing anyway...
)

REM Create proper directory structure on EC2
echo [INFO] Creating proper directory structure on EC2...
ssh -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" ec2-user@35.89.15.141 "mkdir -p /home/ec2-user/trading-algorithm/flask_web /home/ec2-user/trading-algorithm/_2_Orchestrator_And_ML_Python"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create directory structure
    pause
    exit /b 1
)

REM Upload the clean installation script to the correct location
echo [INFO] Uploading clean installation script...
scp -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" "0_Launchers\flask_web\ec2_clean_install.sh" ec2-user@35.89.15.141:/home/ec2-user/trading-algorithm/flask_web/
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to upload clean installation script
    pause
    exit /b 1
)

REM Upload all Flask web files to the correct location
echo [INFO] Uploading all Flask web application files...
scp -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" -r 0_Launchers/flask_web/* ec2-user@35.89.15.141:/home/ec2-user/trading-algorithm/flask_web/
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to upload Flask web files
    pause
    exit /b 1
)

REM Upload ML orchestrator files to the correct location
echo [INFO] Uploading ML orchestrator files...
scp -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" -r "_2_Orchestrator_And_ML_Python" ec2-user@35.89.15.141:/home/ec2-user/trading-algorithm/
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to upload ML orchestrator files
    pause
    exit /b 1
)

goto :verify_and_install

:upload_essential_files
REM Create proper directory structure on EC2 if it doesn't exist
echo [INFO] Creating proper directory structure on EC2...
ssh -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" ec2-user@35.89.15.141 "mkdir -p /home/ec2-user/trading-algorithm/flask_web /home/ec2-user/trading-algorithm/_2_Orchestrator_And_ML_Python"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create directory structure
    pause
    exit /b 1
)

REM Upload only essential files for regular install with smart checking
echo [INFO] Uploading essential files for regular install (smart mode)...
echo [INFO] Checking and uploading only changed/new files...

REM Always upload setup script and requirements.txt for regular installs
echo [INFO] Uploading setup script and requirements.txt (always needed)...
scp -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" "0_Launchers\flask_web\ec2_setup_amazon_linux.sh" ec2-user@35.89.15.141:/home/ec2-user/trading-algorithm/flask_web/
scp -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" "requirements.txt" ec2-user@35.89.15.141:/home/ec2-user/trading-algorithm/flask_web/

REM Check each Flask file individually and upload only if needed
echo [INFO] Checking Flask application files...

REM Check flask_app.py
ssh -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" ec2-user@35.89.15.141 "test -f /home/ec2-user/trading-algorithm/flask_web/flask_app.py && echo 'EXISTS' || echo 'MISSING'" > temp_check.txt
set /p file_status=<temp_check.txt
if "%file_status%"=="MISSING" (
    echo [INFO] flask_app.py missing - uploading...
    scp -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" "0_Launchers\flask_web\flask_app.py" ec2-user@35.89.15.141:/home/ec2-user/trading-algorithm/flask_web/
) else (
    echo [INFO] flask_app.py exists - uploading (overwriting for updates)...
    scp -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" "0_Launchers\flask_web\flask_app.py" ec2-user@35.89.15.141:/home/ec2-user/trading-algorithm/flask_web/
)

REM Check wsgi.py
ssh -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" ec2-user@35.89.15.141 "test -f /home/ec2-user/trading-algorithm/flask_web/wsgi.py && echo 'EXISTS' || echo 'MISSING'" > temp_check.txt
set /p file_status=<temp_check.txt
if "%file_status%"=="MISSING" (
    echo [INFO] wsgi.py missing - uploading...
    scp -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" "0_Launchers\flask_web\wsgi.py" ec2-user@35.89.15.141:/home/ec2-user/trading-algorithm/flask_web/
) else (
    echo [INFO] wsgi.py exists - uploading (overwriting for updates)...
    scp -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" "0_Launchers\flask_web\wsgi.py" ec2-user@35.89.15.141:/home/ec2-user/trading-algorithm/flask_web/
)

REM Check gunicorn.conf.py
ssh -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" ec2-user@35.89.15.141 "test -f /home/ec2-user/trading-algorithm/flask_web/gunicorn.conf.py && echo 'EXISTS' || echo 'MISSING'" > temp_check.txt
set /p file_status=<temp_check.txt
if "%file_status%"=="MISSING" (
    echo [INFO] gunicorn.conf.py missing - uploading...
    scp -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" "0_Launchers\flask_web\gunicorn.conf.py" ec2-user@35.89.15.141:/home/ec2-user/trading-algorithm/flask_web/
) else (
    echo [INFO] gunicorn.conf.py exists - uploading (overwriting for updates)...
    scp -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" "0_Launchers\flask_web\gunicorn.conf.py" ec2-user@35.89.15.141:/home/ec2-user/trading-algorithm/flask_web/
)

REM Check config.py
ssh -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" ec2-user@35.89.15.141 "test -f /home/ec2-user/trading-algorithm/flask_web/config.py && echo 'EXISTS' || echo 'MISSING'" > temp_check.txt
set /p file_status=<temp_check.txt
if "%file_status%"=="MISSING" (
    echo [INFO] config.py missing - uploading...
    scp -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" "0_Launchers\flask_web\config.py" ec2-user@35.89.15.141:/home/ec2-user/trading-algorithm/flask_web/
) else (
    echo [INFO] config.py exists - uploading (overwriting for updates)...
    scp -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" "0_Launchers\flask_web\config.py" ec2-user@35.89.15.141:/home/ec2-user/trading-algorithm/flask_web/
)

REM Check templates directory
ssh -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" ec2-user@35.89.15.141 "test -d /home/ec2-user/trading-algorithm/flask_web/templates && echo 'EXISTS' || echo 'MISSING'" > temp_check.txt
set /p file_status=<temp_check.txt
if "%file_status%"=="MISSING" (
    echo [INFO] templates directory missing - uploading...
    scp -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" -r "0_Launchers\flask_web\templates" ec2-user@35.89.15.141:/home/ec2-user/trading-algorithm/flask_web/
) else (
    echo [INFO] templates directory exists - uploading (overwriting for updates)...
    scp -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" -r "0_Launchers\flask_web\templates" ec2-user@35.89.15.141:/home/ec2-user/trading-algorithm/flask_web/
)

REM Clean up temporary files
del temp_check.txt 2>nul

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to upload essential files
    pause
    exit /b 1
)

goto :verify_and_install

:verify_and_install
REM Verify files were uploaded correctly
echo [INFO] Verifying uploaded files...
ssh -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" ec2-user@35.89.15.141 "ls -la /home/ec2-user/trading-algorithm/flask_web/"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to verify uploaded files
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo [SUCCESS] All files uploaded successfully!
echo ================================================================================
echo.

if "%choice%"=="1" (
    echo Files uploaded include:
    echo - Flask application files (flask_app.py, wsgi.py, etc.)
    echo - Complete requirements.txt with all dependencies
    echo - Clean installation script
    echo - Templates directory
    echo - ML orchestrator files (_2_Orchestrator_And_ML_Python)
    echo - Proper directory structure maintained
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
    echo - Set up new services with proper directory structure
    echo.
    echo [INFO] Starting clean installation...
    echo [INFO] Press Ctrl+C or close this window to cancel the installation.
    echo.
    ssh -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" ec2-user@35.89.15.141 "cd /home/ec2-user/trading-algorithm/flask_web && chmod +x ec2_clean_install.sh && ./ec2_clean_install.sh"
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
    echo Essential files uploaded:
    echo - Flask application files (flask_app.py, wsgi.py, gunicorn.conf.py, config.py)
    echo - Updated requirements.txt
    echo - Templates directory
    echo - Setup script
    echo - Proper directory structure maintained
    echo.
    echo ================================================================================
    echo                    STARTING REGULAR INSTALL
    echo ================================================================================
    echo.
    echo [INFO] Connecting to EC2 instance and running regular installation...
    echo [INFO] This will update existing packages and restart services.
    echo.
    echo [INFO] Starting regular installation...
    ssh -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" ec2-user@35.89.15.141 "cd /home/ec2-user/trading-algorithm/flask_web && chmod +x ec2_setup_amazon_linux.sh && ./ec2_setup_amazon_linux.sh"
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
    goto :end
)

:end
echo ================================================================================
echo [INFO] Installation completed! SSH command to reconnect:
echo.
echo ssh -i "C:\Users\KitKumar\Downloads\KKumar06202025.pem" ec2-user@35.89.15.141
echo.
echo ================================================================================
pause