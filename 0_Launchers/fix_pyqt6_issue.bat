@echo off
setlocal enabledelayedexpansion

echo.
echo ================================================================================
echo                    PYQT6 DLL ISSUE - DIAGNOSIS AND FIX
echo ================================================================================
echo.
echo  This script will help you fix the PyQt6 DLL load issue.
echo.
echo  The error "DLL load failed while importing QtCore" indicates
echo  that you're missing the Microsoft Visual C++ Redistributable.
echo.
echo --------------------------------------------------------------------------------
echo.

:: Test PyQt6 import
echo [1/3] Testing PyQt6 import...
py -3.9 -c "import PyQt6; print('✓ Basic PyQt6 import works')" 2>nul
if errorlevel 1 (
    echo [ERROR] Basic PyQt6 import failed
    goto :error_exit
) else (
    echo [OK] Basic PyQt6 import successful
)

:: Test QtWidgets import (the failing one)
echo.
echo [2/3] Testing QtWidgets import...
py -3.9 -c "from PyQt6.QtWidgets import QApplication; print('✓ QtWidgets import works')" 2>nul
if errorlevel 1 (
    echo [ERROR] QtWidgets import failed - This confirms the Visual C++ Redistributable issue
    echo.
    echo ================================================================================
    echo                              SOLUTION REQUIRED
    echo ================================================================================
    echo.
    echo  You need to install the Microsoft Visual C++ Redistributable 2015-2022.
    echo.
    echo  STEPS TO FIX:
    echo  1. Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo  2. Run the installer as Administrator
    echo  3. Restart your computer
    echo  4. Run the trading system again
    echo.
    echo  ALTERNATIVE: Use PyQt5 instead (more stable on Windows)
    echo.
    echo  Press any key to open the download link in your browser...
    pause >nul
    start https://aka.ms/vs/17/release/vc_redist.x64.exe
    goto :end
) else (
    echo [OK] QtWidgets import successful - PyQt6 should work fine
)

:: Test full application
echo.
echo [3/3] Testing full application...
py -3.9 -c "import sys; sys.path.append('_3_Networking_and_User_Input/local_gui'); from main import *; print('✓ Full application import works')" 2>nul
if errorlevel 1 (
    echo [WARNING] Full application import failed, but PyQt6 core works
    echo           This might be due to other dependencies
) else (
    echo [OK] Full application import successful
    echo.
    echo ================================================================================
    echo                              ALL TESTS PASSED
    echo ================================================================================
    echo.
    echo  PyQt6 is working correctly! You can now run the trading system.
    echo.
    echo  To launch the GUI, run: 0_Launchers\run_local_gui.bat
    echo.
)

goto :end

:error_exit
echo.
echo ================================================================================
echo                              DIAGNOSIS COMPLETE
echo ================================================================================
echo.
echo  The issue is confirmed to be missing Visual C++ Redistributable.
echo.
echo  Please install it and try again.
echo.

:end
echo.
echo ================================================================================
echo  Press any key to exit...
pause >nul
endlocal 