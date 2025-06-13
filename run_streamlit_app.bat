@echo off
REM Activate the virtual environment and run the Streamlit app
cd /d "%~dp0"

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Run the Streamlit app
streamlit run streamlit_app.py

pause 