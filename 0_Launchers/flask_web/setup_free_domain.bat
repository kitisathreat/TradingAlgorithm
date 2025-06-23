@echo off
echo ================================================================================
echo                    FREE DOMAIN SETUP FOR EC2 INSTANCE
echo ================================================================================
echo.
echo Current EC2 IP: 35.89.15.141
echo.
echo Choose your free domain option:
echo.
echo [1] Freenom (.tk, .ml, .ga, .cf domains)
echo [2] InfinityFree (.epizy.com subdomain)
echo [3] Custom domain (you already have one)
echo [4] Exit
echo.
set /p "CHOICE=Enter your choice (1-4): "

if "%CHOICE%"=="1" goto freenom
if "%CHOICE%"=="2" goto infinityfree
if "%CHOICE%"=="3" goto custom
if "%CHOICE%"=="4" goto exit
goto invalid

:freenom
echo.
echo ================================================================================
echo                           FREENOM DOMAIN SETUP
echo ================================================================================
echo.
echo Steps to get a free Freenom domain:
echo.
echo 1. Visit: https://www.freenom.com
echo 2. Create a free account
echo 3. Search for available domains (.tk, .ml, .ga, .cf)
echo 4. Register your chosen domain (FREE for 12 months)
echo 5. Go to "Manage Domain" → "Manage Freenom DNS"
echo 6. Add A Record:
echo    - Name: @ (or leave blank)
echo    - Target: 35.89.15.141
echo    - TTL: 300
echo.
echo After setting up DNS, come back here to update your configuration.
echo.
pause
goto update_config

:infinityfree
echo.
echo ================================================================================
echo                        INFINITYFREE DOMAIN SETUP
echo ================================================================================
echo.
echo Steps to get a free InfinityFree subdomain:
echo.
echo 1. Visit: https://infinityfree.net
echo 2. Create a free account
echo 3. Go to "Subdomains" section
echo 4. Choose your subdomain name (e.g., trading-algo.epizy.com)
echo 5. In DNS settings, add A Record:
echo    - Name: @ (or leave blank)
echo    - Target: 35.89.15.141
echo    - TTL: 300
echo.
echo After setting up DNS, come back here to update your configuration.
echo.
pause
goto update_config

:custom
echo.
echo ================================================================================
echo                           CUSTOM DOMAIN SETUP
echo ================================================================================
echo.
echo If you already have a domain, you can use the existing update script:
echo.
echo Running update_domain.bat...
call update_domain.bat
goto exit

:update_config
echo.
echo ================================================================================
echo                           UPDATE CONFIGURATION
echo ================================================================================
echo.
set /p "DOMAIN_NAME=Enter your domain name (e.g., myapp.tk or trading.epizy.com): "

if "%DOMAIN_NAME%"=="" (
    echo [ERROR] Domain name is required.
    pause
    goto update_config
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
    echo DNS Propagation Check:
    echo You can check if DNS is working by running:
    echo nslookup %DOMAIN_NAME%
    echo.
) else (
    echo [ERROR] ec2_config.txt not found.
    echo Please run the deployment first.
    pause
    exit /b 1
)

goto exit

:invalid
echo.
echo [ERROR] Invalid choice. Please enter 1, 2, 3, or 4.
echo.
pause
goto exit

:exit
echo ================================================================================
echo                           SETUP COMPLETE!
echo ================================================================================
echo.
echo Remember:
echo - DNS propagation can take 5-60 minutes
echo - Test your domain before sharing
echo - Consider HTTPS for security
echo - Free domains may have limitations
echo.
pause 