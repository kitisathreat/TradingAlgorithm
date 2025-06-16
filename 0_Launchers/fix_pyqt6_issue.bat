@echo off
setlocal enabledelayedexpansion

echo.
echo ================================================================================
echo                    PYQT5 DIAGNOSIS AND FIX
echo ================================================================================
echo.
echo  This script will help you fix PyQt5 DLL load issues.
echo.
echo  If you see "DLL load failed while importing QtCore" it may indicate

echo  a missing Microsoft Visual C++ Redistributable, but PyQt5 is less likely to need it.
echo.
echo --------------------------------------------------------------------------------
echo.

:: Test PyQt5 import
echo [1/2] Testing PyQt5 import...
py -3.9 -c "from PyQt5.QtWidgets import QApplication; print('✓ PyQt5 import works')" 2>nul
if errorlevel 1 (
    echo [ERROR] PyQt5 import failed - This may indicate a missing or corrupted installation
    echo.
    echo ================================================================================
    echo                              SOLUTION REQUIRED
    echo ================================================================================
    echo.
    echo  Try reinstalling PyQt5:
    echo    py -3.9 -m pip install --force-reinstall PyQt5
    echo.
    echo  If you still see DLL errors, install the Microsoft Visual C++ Redistributable 2015-2022:
    echo    https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo.
    echo  Press any key to open the download link in your browser...
    pause >nul
    start https://aka.ms/vs/17/release/vc_redist.x64.exe
    goto :end
) else (
    echo [OK] PyQt5 import successful - PyQt5 should work fine
)

goto :end

:end
echo.
echo ================================================================================
echo  Press any key to exit...
pause >nul
endlocal 