# Step 1: Remove existing venv if it exists
if (Test-Path ".\venv") {
    Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force .\venv
}

# Step 2: Run setup_local_env.bat
Write-Host "Running setup_local_env.bat..." -ForegroundColor Green
cmd /c setup_local_env.bat

# Step 3: Run run_streamlit_app.bat
Write-Host "Launching Streamlit app..." -ForegroundColor Green
cmd /c run_streamlit_app.bat

# Note: The script will keep running until you press Ctrl+C to stop the Streamlit server
# To deactivate the virtual environment after stopping, run: deactivate 