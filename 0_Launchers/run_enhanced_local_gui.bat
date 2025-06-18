@echo off
echo ================================================================================
echo                    ENHANCED TRADING ALGORITHM - LOCAL GUI
echo ================================================================================
echo.
echo  UI Improvements:
echo  ✓ Enhanced chart zoom and pan functionality
echo  ✓ Comprehensive technical indicators (matching local GUI)
echo  ✓ Full S&P 100 stock list + additional stocks
echo  ✓ Improved days selector (30, 90, 180, 365, Custom)
echo  ✓ Better user interface and tooltips
echo.
echo ================================================================================
echo.

REM Change to the local GUI directory
cd /d "%~dp0local_gui"

REM Run the enhanced local GUI
call run_local_gui.bat

pause 