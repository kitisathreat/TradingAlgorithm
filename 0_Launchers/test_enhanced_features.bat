@echo off
echo Testing Enhanced GUI Features...
echo.
echo This will open a test window with sample data to demonstrate:
echo - Hover tooltips on candlesticks
echo - Expandable metrics panel
echo - Financial indicators calculation
echo.

cd /d "%~dp0"
cd "_3_Networking_and_User_Input\local_gui"

echo Running test...
python test_enhanced_gui.py

pause 