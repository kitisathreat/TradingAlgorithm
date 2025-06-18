@echo off
echo ========================================
echo Flask Deployment Package Creator
echo ========================================
echo.

echo Choose deployment package creator:
echo.
echo 1. Batch version (Windows only, faster)
echo 2. Python version (Cross-platform, more features)
echo.

set /p choice="Enter your choice (1 or 2): "

if "%choice%"=="1" (
    echo.
    echo Running batch version...
    call create_deployment_zip.bat
) else if "%choice%"=="2" (
    echo.
    echo Running Python version...
    python create_deployment_zip.py --help
    echo.
    echo Available options:
    echo   --output deployment.zip    (default output filename)
    echo   --include-static          (include static files)
    echo   --no-cleanup              (keep temp directory)
    echo   --verbose                 (detailed output)
    echo.
    echo Example: python create_deployment_zip.py --include-static --output my_deployment.zip
    echo.
    python create_deployment_zip.py
) else (
    echo Invalid choice. Please run the script again and select 1 or 2.
    pause
    exit /b 1
)

echo.
echo Deployment package creation completed!
pause 