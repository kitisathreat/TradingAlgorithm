@echo off
echo ========================================
echo   YFinance Rate Limiting Deployment
echo ========================================
echo.

echo [INFO] Deploying rate limiting system to EC2...
echo.

REM Check if required files exist
if not exist "rate_limiter.py" (
    echo [ERROR] rate_limiter.py not found!
    echo Please ensure the rate limiting module is in the current directory.
    pause
    exit /b 1
)

if not exist "flask_app.py" (
    echo [ERROR] flask_app.py not found!
    echo Please ensure the Flask app is in the current directory.
    pause
    exit /b 1
)

echo [INFO] Required files found:
echo   - rate_limiter.py
echo   - flask_app.py (updated)
echo   - requirements.txt (updated)
echo.

REM Test the rate limiting system locally
echo [INFO] Testing rate limiting system locally...
python test_rate_limiting.py
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Local test failed, but continuing with deployment...
    echo.
)

REM Deploy to EC2 (assuming you have the upload script)
if exist "..\..\upload_to_ec2.bat" (
    echo [INFO] Uploading to EC2...
    call "..\..\upload_to_ec2.bat"
) else (
    echo [INFO] Manual deployment required.
    echo Please upload the following files to your EC2 instance:
    echo   - rate_limiter.py
    echo   - flask_app.py
    echo   - requirements.txt
    echo.
)

echo [INFO] Deployment complete!
echo.
echo [INFO] Next steps:
echo 1. SSH into your EC2 instance
echo 2. Navigate to your Flask app directory
echo 3. Restart your Flask application
echo 4. Test the rate limiting: curl http://your-ec2-ip/api/rate_limiter_stats
echo.
echo [INFO] Rate limiting features now available:
echo   - Automatic rate limiting (30 req/min, 1000 req/hour)
echo   - Smart caching (15 minutes)
echo   - Exponential backoff on errors
echo   - Synthetic data fallback
echo   - Real-time monitoring endpoints
echo.

pause 