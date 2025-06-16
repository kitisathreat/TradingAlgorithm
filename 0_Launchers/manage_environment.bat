@echo off
setlocal enabledelayedexpansion

:: Default virtual environment path (AppData - recommended for most users)
set "DEFAULT_VENV_PATH=%LOCALAPPDATA%\TradingAlgorithm\venv"
set "VENV_PATH=%DEFAULT_VENV_PATH%"
set "VENV_ACTIVATE=%VENV_PATH%\Scripts\activate.bat"

:: Clear screen
cls

echo.
echo ================================================================================
echo                    VIRTUAL ENVIRONMENT MANAGER
echo ================================================================================
echo.
echo  This script helps you manage the Python virtual environment for the
echo  Neural Network Trading System.
echo.
echo  Current Environment Location: %VENV_PATH%
echo.
echo  OPTIONS:
echo    1. Use default location (AppData - recommended)
echo    2. Use project directory (./venv)
echo    3. Use custom location
echo    4. Remove existing environment
echo    5. Show environment info
echo    6. Exit
echo.
echo --------------------------------------------------------------------------------
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto :default_location
if "%choice%"=="2" goto :project_location
if "%choice%"=="3" goto :custom_location
if "%choice%"=="4" goto :remove_environment
if "%choice%"=="5" goto :show_info
if "%choice%"=="6" goto :exit
goto :invalid_choice

:default_location
set "VENV_PATH=%DEFAULT_VENV_PATH%"
set "VENV_ACTIVATE=%VENV_PATH%\Scripts\activate.bat"
echo.
echo [INFO] Using default location: %VENV_PATH%
echo        This location is accessible to all users and requires no admin rights.
goto :create_environment

:project_location
set "VENV_PATH=%~dp0..\venv"
set "VENV_ACTIVATE=%VENV_PATH%\Scripts\activate.bat"
echo.
echo [INFO] Using project directory: %VENV_PATH%
echo        This location is within the project folder.
goto :create_environment

:custom_location
echo.
set /p custom_path="Enter custom path for virtual environment: "
if "%custom_path%"=="" (
    echo [ERROR] No path provided. Using default location.
    goto :default_location
)
set "VENV_PATH=%custom_path%"
set "VENV_ACTIVATE=%VENV_PATH%\Scripts\activate.bat"
echo [INFO] Using custom location: %VENV_PATH%
goto :create_environment

:create_environment
echo.
echo [1/3] Checking Python 3.9...
py -3.9 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.9 is required but not found.
    echo        Please install Python 3.9 and ensure it is registered with the 'py' launcher.
    pause
    exit /b 1
)
echo [OK] Python 3.9 found

echo.
echo [2/3] Creating virtual environment...
if exist "%VENV_PATH%" (
    echo    Found existing environment. Removing...
    rmdir /s /q "%VENV_PATH%" 2>nul
)

:: Create parent directory if needed
for %%i in ("%VENV_PATH%") do set "PARENT_DIR=%%~dpi"
if not exist "%PARENT_DIR%" (
    echo    Creating parent directory...
    mkdir "%PARENT_DIR%"
    if errorlevel 1 (
        echo [ERROR] Could not create directory: %PARENT_DIR%
        echo        Please check permissions and try again.
        pause
        exit /b 1
    )
)

py -3.9 -m venv "%VENV_PATH%"
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    echo        Please check if you have write permissions to: %VENV_PATH%
    pause
    exit /b 1
)
echo [OK] Virtual environment created

echo.
echo [3/3] Installing dependencies...
call "%VENV_ACTIVATE%"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

py -3.9 -m pip install --upgrade pip
py -3.9 -m pip install wheel
py -3.9 -m pip install "numpy==1.23.5" "scikit-learn==1.3.0" "PyQt6==6.4.2" "pyqtgraph==0.13.3" "pandas==2.0.3" "yfinance==0.2.36" "qt-material==2.14" "tensorflow==2.13.0"

if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Environment setup complete!
echo.
echo Environment Details:
echo   Location: %VENV_PATH%
echo   Python: 3.9
echo   Status: Ready to use
echo.
echo To use this environment in other scripts, set:
echo   VENV_PATH=%VENV_PATH%
echo   VENV_ACTIVATE=%VENV_ACTIVATE%
echo.
pause
goto :exit

:remove_environment
echo.
if exist "%VENV_PATH%" (
    echo Removing environment at: %VENV_PATH%
    rmdir /s /q "%VENV_PATH%" 2>nul
    if exist "%VENV_PATH%" (
        echo [ERROR] Could not remove environment. It may be in use.
    ) else (
        echo [OK] Environment removed successfully
    )
) else (
    echo [INFO] No environment found at: %VENV_PATH%
)
pause
goto :exit

:show_info
echo.
echo Environment Information:
echo   Path: %VENV_PATH%
echo   Exists: 
if exist "%VENV_PATH%" (
    echo     Yes
    if exist "%VENV_ACTIVATE%" (
        echo   Activated: 
        call "%VENV_ACTIVATE%" >nul 2>&1
        if errorlevel 1 (
            echo     No (activation failed)
        ) else (
            echo     Yes
            echo   Python Version:
            py -3.9 --version 2>nul
            echo   Installed Packages:
            py -3.9 -m pip list --format=columns | findstr /R "numpy tensorflow PyQt6"
        )
    ) else (
        echo     No (activation script missing)
    )
) else (
    echo     No
)
echo.
pause
goto :exit

:invalid_choice
echo.
echo [ERROR] Invalid choice. Please enter a number between 1 and 6.
pause
goto :exit

:exit
echo.
echo Thank you for using the Environment Manager!
endlocal 