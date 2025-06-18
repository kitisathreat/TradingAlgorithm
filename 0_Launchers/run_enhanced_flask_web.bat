@echo off
echo ================================================================================
echo                    ENHANCED TRADING ALGORITHM - FLASK WEB APP
echo ================================================================================
echo.
echo  UI Improvements:
echo  ✓ Enhanced chart zoom and pan functionality (Chart.js zoom plugin)
echo  ✓ Comprehensive technical indicators (matching local GUI)
echo  ✓ Full S&P 100 stock list + additional stocks
echo  ✓ Improved days selector (30, 90, 180, 365, Custom)
echo  ✓ Better user interface and responsive design
echo  ✓ Interactive tooltips and enhanced chart controls
echo.
echo ================================================================================
echo.

REM Change to the Flask web directory
cd /d "%~dp0flask_web"

REM Run the enhanced Flask web app
python run_flask_web.py

pause 