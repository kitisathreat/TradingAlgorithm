@echo off
set PYTHON_PATH=C:\Users\KitKumar\AppData\Local\Programs\Python\Python39
set PATH=%PYTHON_PATH%;%PYTHON_PATH%\Scripts;%PATH%

echo Using Python from: %PYTHON_PATH%
%PYTHON_PATH%\python.exe --version

echo Creating virtual environment in root directory...
%PYTHON_PATH%\python.exe -m venv ..\venv

echo Activating virtual environment...
call ..\venv\Scripts\activate.bat

echo Upgrading pip and installing wheel...
python -m pip install --upgrade pip
python -m pip install --upgrade wheel

echo Installing requirements...
python -m pip install -r ..\requirements.txt

echo Setup complete! Virtual environment is ready.
echo To activate the environment, run: ..\venv\Scripts\activate.bat
pause 