@echo off
echo ========================================
echo Installing AWS CLI without Admin Rights
echo ========================================
echo.

echo [1/4] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found
    pause
    exit /b 1
)
echo ✓ Python found

echo.
echo [2/4] Installing AWS CLI via pip (user install)...
pip install --user awscli

echo.
echo [3/4] Finding AWS CLI installation path...
python -c "import site; print(site.USER_BASE + '\\Scripts')"

echo.
echo [4/4] Adding to PATH for current session...
for /f "delims=" %%i in ('python -c "import site; print(site.USER_BASE + '\\Scripts')"') do set AWS_PATH=%%i
set PATH=%AWS_PATH%;%PATH%

echo.
echo ========================================
echo Testing AWS CLI installation...
echo ========================================
aws --version

echo.
echo ========================================
echo INSTALLATION COMPLETE!
echo ========================================
echo.
echo AWS CLI has been installed to your user directory.
echo.
echo To make it permanent, add this to your PATH:
echo %AWS_PATH%
echo.
echo To configure AWS credentials, run:
echo aws configure
echo.
pause 