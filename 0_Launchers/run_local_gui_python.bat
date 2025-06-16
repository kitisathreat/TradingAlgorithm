@echo off
setlocal

:: Simple wrapper to run the Python executable version
echo [INFO] Launching Python executable version of run_local_gui...
echo.

:: Try to run with py launcher first (Windows)
py -3.9 "%~dp0run_local_gui.py"
if errorlevel 1 (
    :: Fallback to python command
    python "%~dp0run_local_gui.py"
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to run Python script
        echo        Please ensure Python 3.9 is installed and accessible
        echo.
        pause
        exit /b 1
    )
)

endlocal 