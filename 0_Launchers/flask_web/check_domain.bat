@echo off
echo ================================================================================
echo                        DOMAIN CONNECTIVITY CHECKER
echo ================================================================================
echo.

set /p "DOMAIN_NAME=Enter your domain name to check: "

if "%DOMAIN_NAME%"=="" (
    echo [ERROR] Domain name is required.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo                           CHECKING DOMAIN: %DOMAIN_NAME%
echo ================================================================================
echo.

echo [1/4] Checking DNS resolution...
nslookup %DOMAIN_NAME% 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] DNS lookup failed. Domain may not be configured yet.
) else (
    echo [SUCCESS] DNS resolution working.
)

echo.
echo [2/4] Checking if domain points to EC2 IP (35.89.15.141)...
for /f "tokens=2 delims=:" %%a in ('nslookup %DOMAIN_NAME% 2^>nul ^| findstr "Address"') do (
    set "RESOLVED_IP=%%a"
    set "RESOLVED_IP=!RESOLVED_IP: =!"
)
if "!RESOLVED_IP!"=="35.89.15.141" (
    echo [SUCCESS] Domain correctly points to EC2 instance.
) else (
    echo [WARNING] Domain resolves to !RESOLVED_IP! (expected 35.89.15.141)
    echo [INFO] This might be normal if you're using a CDN or proxy.
)

echo.
echo [3/4] Testing HTTP connectivity...
powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://%DOMAIN_NAME%' -TimeoutSec 10 -UseBasicParsing; Write-Host '[SUCCESS] HTTP response:' $response.StatusCode; } catch { Write-Host '[ERROR] HTTP request failed:' $_.Exception.Message; }"

echo.
echo [4/4] Testing application endpoint...
powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://%DOMAIN_NAME%/api/get_status' -TimeoutSec 10 -UseBasicParsing; Write-Host '[SUCCESS] Application API working:' $response.StatusCode; } catch { Write-Host '[WARNING] Application API not responding:' $_.Exception.Message; }"

echo.
echo ================================================================================
echo                           SUMMARY
echo ================================================================================
echo.
echo Domain: %DOMAIN_NAME%
echo Expected IP: 35.89.15.141
echo.
echo If all checks passed, your domain is working correctly!
echo.
echo Troubleshooting tips:
echo - DNS propagation can take 5-60 minutes
echo - Check your domain registrar's DNS settings
echo - Ensure EC2 security group allows port 80
echo - Verify nginx is running on your EC2 instance
echo.
echo To test manually:
echo 1. Open browser: http://%DOMAIN_NAME%
echo 2. Check for your trading algorithm interface
echo 3. Test all functionality
echo.
pause 