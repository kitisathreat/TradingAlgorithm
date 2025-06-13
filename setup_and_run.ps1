# Create and activate virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Green
python -m venv venv
.\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Green
python -m pip install -r "networking_and_user_input\web_interface\requirements.txt"

# Start Streamlit
Write-Host "Starting Streamlit app..." -ForegroundColor Green
python "networking_and_user_input\web_interface\start_streamlit.py"

# Note: The script will keep running until you press Ctrl+C to stop the Streamlit server
# To deactivate the virtual environment after stopping, run: deactivate 