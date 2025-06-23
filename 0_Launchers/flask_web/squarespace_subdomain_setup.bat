@echo off
echo ================================================================================
echo                    SQUARESPACE SUBDOMAIN SETUP
echo ================================================================================
echo.
echo Domain: mylensandi.com (managed by Squarespace)
echo EC2 IP: 35.89.15.141
echo.
echo This guide will help you set up a subdomain on your Squarespace-managed domain.
echo.
echo Suggested subdomain names:
echo - trading.mylensandi.com
echo - algo.mylensandi.com
echo - neural.mylensandi.com
echo - ai.mylensandi.com
echo - stock.mylensandi.com
echo.
set /p "SUBDOMAIN=Enter your desired subdomain (without .mylensandi.com): "

if "%SUBDOMAIN%"=="" (
    echo [ERROR] Subdomain name is required.
    pause
    exit /b 1
)

set "FULL_DOMAIN=%SUBDOMAIN%.mylensandi.com"

echo.
echo ================================================================================
echo                    SQUARESPACE DNS CONFIGURATION
echo ================================================================================
echo.
echo IMPORTANT: Squarespace DNS management is different from regular domain registrars.
echo.
echo Step-by-step instructions for Squarespace:
echo.
echo 1. Log into your Squarespace account
echo 2. Go to Settings → Domains
echo 3. Click on "mylensandi.com"
echo 4. Click "DNS Settings" or "Advanced DNS"
echo 5. Look for "DNS Records" or "Custom Records"
echo.
echo If you see "DNS Records" section:
echo - Click "Add Record"
echo - Record Type: A
echo - Name: %SUBDOMAIN%
echo - Points to: 35.89.15.141
echo - TTL: 300 (or leave default)
echo.
echo If you don't see DNS Records option:
echo - You may need to use Squarespace's "External Links" feature
echo - Or contact Squarespace support to enable DNS management
echo.
echo ALTERNATIVE METHOD (if DNS Records not available):
echo.
echo 1. In Squarespace, go to Pages
echo 2. Add a new page called "%SUBDOMAIN%"
echo 3. Set it as "Not Linked" (hidden from navigation)
echo 4. Add a "Link" block
echo 5. Set the link to: http://35.89.15.141
echo 6. This will redirect %SUBDOMAIN%.mylensandi.com to your EC2 instance
echo.
echo COMMON SQUARESPACE ISSUES:
echo.
echo - DNS Records may not be available on all Squarespace plans
echo - You might need to upgrade to a higher plan for DNS management
echo - Some Squarespace plans only allow subdomains for Squarespace sites
echo.
echo If DNS Records are not available, you can:
echo 1. Use the redirect method above
echo 2. Contact Squarespace support to enable DNS management
echo 3. Consider transferring domain to another registrar (GoDaddy, Namecheap, etc.)
echo.
echo After setting up the DNS record or redirect, come back here to update your configuration.
echo.
pause

echo.
echo ================================================================================
echo                           UPDATE CONFIGURATION
echo ================================================================================
echo.
echo [INFO] Updating ec2_config.txt with subdomain...
echo.

if exist "ec2_config.txt" (
    REM Create backup
    copy "ec2_config.txt" "ec2_config.txt.backup"
    
    REM Update the file using PowerShell
    powershell -Command "(Get-Content ec2_config.txt) -replace 'PUBLIC_IP=.*', 'PUBLIC_IP=%FULL_DOMAIN%' | Set-Content ec2_config.txt"
    
    echo [SUCCESS] Configuration updated!
    echo.
    echo Old configuration backed up to: ec2_config.txt.backup
    echo New subdomain: %FULL_DOMAIN%
    echo.
    echo Your application will be accessible at:
    echo http://%FULL_DOMAIN%
    echo.
    echo Next steps:
    echo 1. Wait 5-60 minutes for DNS propagation
    echo 2. Test your subdomain: http://%FULL_DOMAIN%
    echo 3. Update any bookmarks or documentation
    echo 4. Consider setting up HTTPS for security
    echo.
    echo To test DNS propagation, run:
    echo nslookup %FULL_DOMAIN%
    echo.
    echo To check connectivity, run:
    echo check_domain.bat
    echo.
) else (
    echo [ERROR] ec2_config.txt not found.
    echo Please run the deployment first.
    pause
    exit /b 1
)

echo ================================================================================
echo                           SETUP COMPLETE!
echo ================================================================================
echo.
echo Your trading algorithm will be accessible at:
echo http://%FULL_DOMAIN%
echo.
echo Remember:
echo - DNS propagation can take 5-60 minutes
echo - Test your subdomain before sharing
echo - Consider HTTPS for security
echo - Squarespace may have limitations on DNS management
echo.
pause 