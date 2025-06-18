@echo off
echo ========================================
echo Flask Deployment Package Creator
echo ========================================
echo.

echo Running Python deployment package creator...
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

echo.
echo Deployment package creation completed!
pause 