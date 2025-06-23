@echo off
echo ================================================================================
echo                    SUBDOMAIN SETUP FOR mylensandi.com
echo ================================================================================
echo.
echo Current EC2 IP: 35.89.15.141
echo Your Domain: mylensandi.com
echo.
echo This script will help you set up a subdomain to point to your EC2 instance.
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
echo                           DNS CONFIGURATION
echo ================================================================================
echo.
echo You need to add a DNS record in your domain registrar's control panel:
echo.
echo DNS Record Type: A Record
echo Name/Host: %SUBDOMAIN%
echo Points to/Value: 35.89.15.141
echo TTL: 300 (or 5 minutes)
echo.
echo Step-by-step instructions:
echo.
echo 1. Log into your domain registrar (where you bought mylensandi.com)
echo 2. Find "DNS Management" or "DNS Settings"
echo 3. Look for "A Records" or "DNS Records"
echo 4. Add a new A record with the above settings
echo 5. Save the changes
echo.
echo Common domain registrars and where to find DNS settings:
echo.
echo - GoDaddy: My Domains → Manage → DNS → Manage Zones
echo - Namecheap: Domain List → Manage → Advanced DNS
echo - Google Domains: Select domain → DNS → Manage Custom Records
echo - AWS Route 53: Hosted Zones → mylensandi.com → Create Record
echo - Cloudflare: Select domain → DNS → Records
echo.
echo After adding the DNS record, come back here to update your configuration.
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
echo - You can create multiple subdomains if needed
echo.
pause 