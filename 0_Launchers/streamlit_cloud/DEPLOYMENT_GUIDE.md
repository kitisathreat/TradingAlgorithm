# Streamlit Cloud Deployment Guide

## Overview
This guide explains how to deploy the Advanced Neural Network Trading System to Streamlit Cloud while resolving the websocket dependency conflict between TensorFlow and Alpaca Trade API.

## The Problem
- **TensorFlow 2.13.0** requires `websockets >= 13.0`
- **Alpaca Trade API 3.2.0** requires `websockets <= 11.0`
- This creates a dependency conflict that prevents both from working together

## The Solution
We've implemented a multi-layered approach to resolve this conflict:

### 1. Automatic Setup Script
The `streamlit_app.py` automatically runs `setup_streamlit_cloud.py` on first load to:
- Force install `websockets >= 13.0` first
- Install `alpaca-trade-api` with `--no-deps` to ignore the websocket constraint
- Install all other dependencies

### 2. Graceful Error Handling
The app includes robust error handling that:
- Continues to function even if some dependencies fail
- Shows clear status messages about what's working
- Provides fallback functionality

### 3. Manual Installation Script
You can also run `install_requirements.py` manually if needed.

## Deployment Steps

### Step 1: Prepare Your Repository
1. Ensure all files in `0_Launchers/streamlit_cloud/` are committed to your repository
2. The main file for Streamlit Cloud is: **`streamlit_app.py`**

### Step 2: Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository
3. Set the main file path to: `0_Launchers/streamlit_cloud/streamlit_app.py`
4. Set Python version to: `3.9`
5. Deploy

### Step 3: Monitor the Deployment
The app will automatically:
1. Run the setup script to resolve dependencies
2. Show status messages about what's working
3. Continue to function even if some dependencies have issues

## File Structure
```
0_Launchers/streamlit_cloud/
├── streamlit_app.py              # Main Streamlit app (entry point)
├── setup_streamlit_cloud.py      # Automatic dependency setup
├── install_requirements.py       # Manual installation script
├── streamlit_requirements.txt    # Requirements with conflict resolution
├── streamlit_requirements_override.txt  # Alternative requirements
├── constraints.txt               # Version constraints
├── packages.txt                  # System packages
├── runtime.txt                   # Python version
└── DEPLOYMENT_GUIDE.md          # This file
```

## Troubleshooting

### If Dependencies Fail to Install
1. Check the Streamlit Cloud logs for error messages
2. The app will show warnings but continue to function
3. Most features will work even with partial dependency installation

### If Alpaca API Doesn't Work
- The app will show a warning but continue to function
- You can still use the ML training and prediction features
- Stock data loading via yfinance will still work

### If TensorFlow Doesn't Work
- The app will show a warning but continue to function
- You can still use the data analysis and visualization features
- The training interface will be limited

## Expected Behavior
On first load, you should see:
1. 🔧 Setting up dependencies to resolve websocket conflicts...
2. ✅ Dependencies setup completed successfully (or warnings)
3. ✅ Websockets X.X.X - Compatible with TensorFlow
4. ✅ Alpaca Trade API imported successfully
5. ✅ TensorFlow X.X.X imported successfully

## Performance Notes
- The setup script runs only once per session
- Subsequent loads will be faster
- The app caches the setup status in session state

## Security Notes
- No API keys are required for basic functionality
- The app uses dummy keys for testing Alpaca API functionality
- All sensitive operations require user-provided credentials

## Support
If you encounter issues:
1. Check the Streamlit Cloud logs
2. Look for error messages in the app interface
3. The app provides detailed status information about what's working 