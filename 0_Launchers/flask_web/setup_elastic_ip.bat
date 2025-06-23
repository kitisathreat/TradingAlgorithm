@echo off
echo ================================================================================
echo                    SETTING UP ELASTIC IP FOR EC2 INSTANCE
echo ================================================================================
echo.
echo This script will help you allocate and associate an Elastic IP to your EC2 instance.
echo This will give you a static IP address that won't change when you restart your instance.
echo.

REM Read current configuration
if not exist "ec2_config.txt" (
    echo [ERROR] ec2_config.txt not found. Please run deployment first.
    pause
    exit /b 1
)

for /f "tokens=1,2 delims==" %%a in (ec2_config.txt) do (
    if "%%a"=="INSTANCE_ID" set "INSTANCE_ID=%%b"
    if "%%a"=="REGION" set "REGION=%%b"
)

if "%INSTANCE_ID%"=="" (
    echo [ERROR] INSTANCE_ID not found in ec2_config.txt
    pause
    exit /b 1
)

if "%REGION%"=="" set "REGION=us-west-2"

echo Current Configuration:
echo Instance ID: %INSTANCE_ID%
echo Region: %REGION%
echo.

echo ================================================================================
echo                           STEP 1: ALLOCATE ELASTIC IP
echo ================================================================================
echo.

echo [INFO] Allocating new Elastic IP...
for /f "tokens=*" %%i in ('aws ec2 allocate-address --domain vpc --region %REGION% --output text 2^>nul') do set "ALLOCATION_ID=%%i"

if "%ALLOCATION_ID%"=="" (
    echo [ERROR] Failed to allocate Elastic IP. Check your AWS CLI configuration.
    echo Make sure you have AWS CLI installed and configured with appropriate permissions.
    pause
    exit /b 1
)

echo [SUCCESS] Elastic IP allocated successfully!
echo Allocation ID: %ALLOCATION_ID%

echo.
echo [INFO] Getting the Elastic IP address...
for /f "tokens=*" %%i in ('aws ec2 describe-addresses --allocation-ids %ALLOCATION_ID% --region %REGION% --query "Addresses[0].PublicIp" --output text 2^>nul') do set "ELASTIC_IP=%%i"

if "%ELASTIC_IP%"=="" (
    echo [ERROR] Failed to get Elastic IP address.
    pause
    exit /b 1
)

echo [SUCCESS] Elastic IP address: %ELASTIC_IP%

echo.
echo ================================================================================
echo                           STEP 2: ASSOCIATE WITH INSTANCE
echo ================================================================================
echo.

echo [INFO] Associating Elastic IP with your EC2 instance...
for /f "tokens=*" %%i in ('aws ec2 associate-address --instance-id %INSTANCE_ID% --allocation-id %ALLOCATION_ID% --region %REGION% --output text 2^>nul') do set "ASSOCIATION_ID=%%i"

if "%ASSOCIATION_ID%"=="" (
    echo [ERROR] Failed to associate Elastic IP with instance.
    echo You may need to disassociate any existing Elastic IP first.
    pause
    exit /b 1
)

echo [SUCCESS] Elastic IP associated successfully!
echo Association ID: %ASSOCIATION_ID%

echo.
echo ================================================================================
echo                           STEP 3: UPDATE CONFIGURATION
echo ================================================================================
echo.

echo [INFO] Updating ec2_config.txt with new Elastic IP...
powershell -Command "(Get-Content ec2_config.txt) -replace 'PUBLIC_IP=.*', 'PUBLIC_IP=%ELASTIC_IP%' | Set-Content ec2_config.txt"

echo [SUCCESS] Configuration updated!

echo.
echo ================================================================================
echo                           SETUP COMPLETE!
echo ================================================================================
echo.
echo Your EC2 instance now has a static Elastic IP address:
echo.
echo Old IP: 35.89.15.141
echo New IP: %ELASTIC_IP%
echo.
echo Your application is now accessible at:
echo http://%ELASTIC_IP%
echo.
echo Benefits of Elastic IP:
echo - Static address that won't change on instance restart
echo - Can be reassigned to other instances if needed
echo - More professional and memorable URL
echo.
echo Next steps:
echo 1. Test your application: http://%ELASTIC_IP%
echo 2. Update any bookmarks or documentation with the new IP
echo 3. Consider setting up a domain name for even better branding
echo.
echo ================================================================================
pause 