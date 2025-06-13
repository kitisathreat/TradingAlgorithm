@echo off
REM Set up Python virtual environment and install dependencies
cd /d "%~dp0"

REM Create virtual environment named 'venv'
python -m venv venv

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
pip install -r requirements.txt

echo.
echo Environment setup complete. You can now run your Streamlit app with:
echo streamlit run streamlit_app.py
echo.
pause 