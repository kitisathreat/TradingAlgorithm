@echo off
echo Testing robust setup approach for websocket dependency conflict...
echo.

REM Change to the project root directory
cd /d "%~dp0..\.."

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call "venv\Scripts\activate.bat"
) else (
    echo No virtual environment found, using system Python...
)

REM Test the robust setup script
echo Running robust setup script...
python "0_Launchers\build_tools\setup_and_run_robust.py"

echo.
echo Test completed.
pause 