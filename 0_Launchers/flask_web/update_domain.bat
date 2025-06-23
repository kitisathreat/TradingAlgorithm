@echo off
echo ================================================================================
echo                    UPDATING DOMAIN CONFIGURATION
echo ================================================================================
echo.

set /p "DOMAIN_NAME=Enter your domain name (e.g., trading-algorithm.com): "

if "%DOMAIN_NAME%"=="" (
    echo [ERROR] Domain name is required.
    pause
    exit /b 1
)

echo.
echo [INFO] Updating ec2_config.txt with domain name...
echo.

if exist "ec2_config.txt" (
    REM Create backup
    copy "ec2_config.txt" "ec2_config.txt.backup"
    
    REM Update the file using PowerShell
    powershell -Command "(Get-Content ec2_config.txt) -replace 'PUBLIC_IP=.*', 'PUBLIC_IP=%DOMAIN_NAME%' | Set-Content ec2_config.txt"
    
    echo [SUCCESS] Configuration updated!
    echo.
    echo Old configuration backed up to: ec2_config.txt.backup
    echo New domain: %DOMAIN_NAME%
    echo.
    echo Your application should now be accessible at:
    echo http://%DOMAIN_NAME%
    echo.
    echo Next steps:
    echo 1. Wait 5-60 minutes for DNS propagation
    echo 2. Test your domain: http://%DOMAIN_NAME%
    echo 3. Update any bookmarks or documentation
    echo 4. Consider setting up HTTPS for security
    echo.
) else (
    echo [ERROR] ec2_config.txt not found.
    echo Please run the deployment first.
    pause
    exit /b 1
)

echo ================================================================================
echo                           CONFIGURATION UPDATED!
echo ================================================================================
pause 