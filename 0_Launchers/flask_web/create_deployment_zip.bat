@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Trading Algorithm - Flask EB Deployment
echo ========================================
echo.

REM Set colors for output
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "RESET=[0m"

REM Check if running from correct directory
echo %BLUE%[1/8] Checking environment...%RESET%
if not exist "flask_app.py" (
    echo %RED%ERROR: flask_app.py not found. Please run this script from the flask_web directory.%RESET%
    pause
    exit /b 1
)

if not exist "wsgi.py" (
    echo %RED%ERROR: wsgi.py not found. Required for Elastic Beanstalk.%RESET%
    pause
    exit /b 1
)

if not exist "requirements.txt" (
    echo %RED%ERROR: requirements.txt not found. Required for dependency installation.%RESET%
    pause
    exit /b 1
)

echo %GREEN%✓ All required files found%RESET%

REM Check for ML orchestrator components
echo.
echo %BLUE%[2/8] Checking ML orchestrator components...%RESET%
set "MISSING_COMPONENTS="

if not exist "..\..\_2_Orchestrator_And_ML_Python" (
    echo %RED%ERROR: ML orchestrator directory not found!%RESET%
    echo %RED%This deployment will have limited functionality.%RESET%
    set "MISSING_COMPONENTS=1"
) else (
    echo %GREEN%✓ ML orchestrator directory found%RESET%
)

if not exist "..\..\_2_Orchestrator_And_ML_Python\interactive_training_app" (
    echo %RED%ERROR: Interactive training app not found!%RESET%
    set "MISSING_COMPONENTS=1"
) else (
    echo %GREEN%✓ Interactive training app found%RESET%
)

if not exist "..\..\_2_Orchestrator_And_ML_Python\model_bridge.py" (
    echo %RED%ERROR: Model bridge not found!%RESET%
    set "MISSING_COMPONENTS=1"
) else (
    echo %GREEN%✓ Model bridge found%RESET%
)

if not exist "..\..\_2_Orchestrator_And_ML_Python\stock_selection_utils.py" (
    echo %RED%ERROR: Stock selection utilities not found!%RESET%
    set "MISSING_COMPONENTS=1"
) else (
    echo %GREEN%✓ Stock selection utilities found%RESET%
)

if not exist "..\..\_2_Orchestrator_And_ML_Python\date_range_utils.py" (
    echo %RED%ERROR: Date range utilities not found!%RESET%
    set "MISSING_COMPONENTS=1"
) else (
    echo %GREEN%✓ Date range utilities found%RESET%
)

if not exist "..\..\_2_Orchestrator_And_ML_Python\sp100_symbols.json" (
    echo %YELLOW%Warning: SP100 symbols file not found%RESET%
) else (
    echo %GREEN%✓ SP100 symbols file found%RESET%
)

if defined MISSING_COMPONENTS (
    echo.
    echo %RED%CRITICAL: Missing ML components will result in limited functionality!%RESET%
    echo %YELLOW%The deployment will not be able to:%RESET%
    echo %YELLOW%- Make trading predictions%RESET%
    echo %YELLOW%- Train neural networks%RESET%
    echo %YELLOW%- Process user training data%RESET%
    echo %YELLOW%- Provide full trading functionality%RESET%
    echo.
    set /p continue="Continue anyway? (y/N): "
    if /i not "%continue%"=="y" (
        echo Deployment cancelled.
        pause
        exit /b 1
    )
)

REM Clean up any existing deployment files
echo.
echo %BLUE%[3/8] Cleaning up previous deployment files...%RESET%
if exist "deployment.zip" (
    del "deployment.zip"
    echo %YELLOW%Removed old deployment.zip%RESET%
)

if exist "deployment_temp" (
    rmdir /s /q "deployment_temp"
    echo %YELLOW%Removed old temp directory%RESET%
)

REM Create temporary directory for staging
echo.
echo %BLUE%[4/8] Creating deployment package...%RESET%
mkdir "deployment_temp"

REM Copy core Flask files
echo Copying core Flask files...
copy "flask_app.py" "deployment_temp\"
copy "wsgi.py" "deployment_temp\"
copy "requirements.txt" "deployment_temp\"
copy "Procfile" "deployment_temp\"
copy "gunicorn.conf.py" "deployment_temp\"
copy "config.py" "deployment_temp\"

REM Copy .ebextensions directory
echo Copying Elastic Beanstalk configuration...
xcopy ".ebextensions" "deployment_temp\.ebextensions\" /E /I /Y

REM Copy templates directory
if exist "templates" (
    echo Copying templates...
    xcopy "templates" "deployment_temp\templates\" /E /I /Y
)

REM Copy static files if they exist
if exist "static" (
    echo Copying static files...
    xcopy "static" "deployment_temp\static\" /E /I /Y
)

REM Copy ML orchestrator components
echo.
echo %BLUE%[5/8] Copying ML orchestrator components...%RESET%
if exist "..\..\_2_Orchestrator_And_ML_Python" (
    echo Creating ML orchestrator directory...
    mkdir "deployment_temp\_2_Orchestrator_And_ML_Python"
    
    REM Copy core ML files
    echo Copying core ML files...
    copy "..\..\_2_Orchestrator_And_ML_Python\model_bridge.py" "deployment_temp\_2_Orchestrator_And_ML_Python\"
    copy "..\..\_2_Orchestrator_And_ML_Python\stock_selection_utils.py" "deployment_temp\_2_Orchestrator_And_ML_Python\"
    copy "..\..\_2_Orchestrator_And_ML_Python\date_range_utils.py" "deployment_temp\_2_Orchestrator_And_ML_Python\"
    copy "..\..\_2_Orchestrator_And_ML_Python\main.py" "deployment_temp\_2_Orchestrator_And_ML_Python\"
    copy "..\..\_2_Orchestrator_And_ML_Python\__init__.py" "deployment_temp\_2_Orchestrator_And_ML_Python\"
    
    REM Copy data files
    if exist "..\..\_2_Orchestrator_And_ML_Python\sp100_symbols.json" (
        copy "..\..\_2_Orchestrator_And_ML_Python\sp100_symbols.json" "deployment_temp\_2_Orchestrator_And_ML_Python\"
    )
    if exist "..\..\_2_Orchestrator_And_ML_Python\sp100_symbols_with_market_cap.json" (
        copy "..\..\_2_Orchestrator_And_ML_Python\sp100_symbols_with_market_cap.json" "deployment_temp\_2_Orchestrator_And_ML_Python\"
    )
    
    REM Copy interactive training app
    if exist "..\..\_2_Orchestrator_And_ML_Python\interactive_training_app" (
        echo Copying interactive training app...
        xcopy "..\..\_2_Orchestrator_And_ML_Python\interactive_training_app" "deployment_temp\_2_Orchestrator_And_ML_Python\interactive_training_app\" /E /I /Y
    )
    
    REM Copy trading executor
    if exist "..\..\_2_Orchestrator_And_ML_Python\trading_executor.py" (
        copy "..\..\_2_Orchestrator_And_ML_Python\trading_executor.py" "deployment_temp\_2_Orchestrator_And_ML_Python\"
    )
    
    echo %GREEN%✓ ML orchestrator components copied%RESET%
) else (
    echo %RED%✗ ML orchestrator directory not found - limited functionality%RESET%
)

REM Create .ebignore file to exclude unnecessary files
echo.
echo %BLUE%[6/8] Creating .ebignore file...%RESET%
(
echo # Elastic Beanstalk ignore file
echo # Exclude files that shouldn't be deployed
echo *.bat
echo *.sh
echo *.pyc
echo __pycache__/
echo .git/
echo .gitignore
echo README*.md
echo AWS_DEPLOYMENT_GUIDE.md
echo README_DEPLOYMENT.md
echo test_*.py
echo run_*.py
echo setup_*.py
echo deploy_*.bat
echo deploy_*.sh
echo docker-compose.yml
echo Dockerfile
echo .env
echo *.log
echo logs/
echo venv/
echo __pycache__/
echo .pytest_cache/
echo .coverage
echo htmlcov/
echo .tox/
echo .mypy_cache/
echo .vscode/
echo .idea/
echo create_deployment_zip.py
echo create_deployment_zip.bat
echo run_deployment_creator.bat
) > "deployment_temp\.ebignore"

REM Create deployment zip
echo.
echo %BLUE%[7/8] Creating ZIP archive...%RESET%
cd deployment_temp

REM Use PowerShell to create zip with proper structure
powershell -command "Compress-Archive -Path * -DestinationPath '..\deployment.zip' -Force"

cd ..

REM Verify zip was created
if exist "deployment.zip" (
    echo %GREEN%✓ Deployment package created successfully%RESET%
) else (
    echo %RED%✗ Failed to create deployment package%RESET%
    pause
    exit /b 1
)

REM Clean up temp directory
rmdir /s /q "deployment_temp"

REM Validate zip contents
echo.
echo %BLUE%[8/8] Validating deployment package...%RESET%

REM Check zip size
for %%A in (deployment.zip) do set "ZIP_SIZE=%%~zA"
set /a "ZIP_SIZE_MB=%ZIP_SIZE%/1024/1024"
echo Deployment package size: %ZIP_SIZE_MB% MB

if %ZIP_SIZE_MB% LSS 1 (
    echo %YELLOW%Warning: Deployment package seems very small. Please verify contents.%RESET%
)

REM List contents of zip
echo.
echo %BLUE%Deployment package contents:%RESET%
powershell -command "Add-Type -AssemblyName System.IO.Compression.FileSystem; [System.IO.Compression.ZipFile]::OpenRead('deployment.zip').Entries | Select-Object Name, Length | Format-Table -AutoSize"

echo.
echo %GREEN%========================================%RESET%
echo %GREEN%DEPLOYMENT PACKAGE READY!%RESET%
echo %GREEN%========================================%RESET%
echo.
echo %BLUE%Next Steps:%RESET%
echo.
echo 1. %YELLOW%Go to AWS Elastic Beanstalk Console:%RESET%
echo    https://console.aws.amazon.com/elasticbeanstalk/
echo.
echo 2. %YELLOW%Create new application or select existing environment%RESET%
echo.
echo 3. %YELLOW%Upload deployment.zip file%RESET%
echo.
echo 4. %YELLOW%Deploy and wait for completion%RESET%
echo.
echo %BLUE%Environment Configuration:%RESET%
echo - Platform: Python 3.9
echo - Instance type: t3.micro (free tier) or t3.medium
echo - Environment type: Single instance (free tier)
echo.
echo %BLUE%Files included in deployment.zip:%RESET%
echo ✓ flask_app.py (main Flask application)
echo ✓ wsgi.py (WSGI entry point)
echo ✓ requirements.txt (Python dependencies)
echo ✓ Procfile (process definition)
echo ✓ gunicorn.conf.py (Gunicorn configuration)
echo ✓ config.py (application configuration)
echo ✓ .ebextensions/ (Elastic Beanstalk configuration)
echo ✓ templates/ (HTML templates)
echo ✓ .ebignore (deployment exclusions)
if exist "..\..\_2_Orchestrator_And_ML_Python" (
    echo ✓ _2_Orchestrator_And_ML_Python/ (ML orchestrator)
    echo ✓ model_bridge.py (neural network bridge)
    echo ✓ interactive_training_app/ (training interface)
    echo ✓ stock_selection_utils.py (stock utilities)
    echo ✓ date_range_utils.py (date utilities)
    echo ✓ sp100_symbols.json (stock data)
) else (
    echo %RED%✗ ML components missing - limited functionality%RESET%
)
echo.
echo %BLUE%Functionality included:%RESET%
if exist "..\..\_2_Orchestrator_And_ML_Python" (
    echo ✓ Full trading algorithm with neural networks
    echo ✓ Real-time stock data fetching
    echo ✓ User training data collection
    echo ✓ Model training and predictions
    echo ✓ WebSocket support for real-time updates
    echo ✓ Technical analysis and sentiment analysis
) else (
    echo %RED%✗ Basic Flask interface only%RESET%
    echo %RED%✗ No ML capabilities%RESET%
    echo %RED%✗ Limited trading functionality%RESET%
)
echo.
echo %GREEN%Ready for deployment!%RESET%
echo.
pause 