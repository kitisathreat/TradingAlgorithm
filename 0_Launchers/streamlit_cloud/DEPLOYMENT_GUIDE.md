# Streamlit Cloud Deployment Guide - Alpaca Websocket Version Fix

## Problem Description
Streamlit Cloud deployment fails due to Alpaca websocket version dependency conflicts. The `alpaca-trade-api==3.2.0` package requires specific websocket versions that conflict with other dependencies.

## Solutions Available

### Solution 1: Use Custom Installation Script (Recommended)
1. **File**: `install_requirements.py`
2. **How to use**: 
   - Add this script to your Streamlit Cloud repository
   - Run it manually during deployment or as a pre-deployment step
   - It installs packages with `--no-deps` flag to avoid conflicts

### Solution 2: Use Modified Requirements File
1. **File**: `streamlit_requirements_override.txt`
2. **How to use**:
   - Rename this to `requirements.txt` in your Streamlit Cloud directory
   - Contains `--no-deps` flags for problematic packages
   - Forces websocket version to 13.0

### Solution 3: Use Setup Script
1. **File**: `setup_streamlit_cloud.py`
2. **How to use**:
   - Run this script during Streamlit Cloud deployment
   - Automatically handles version conflicts
   - Provides detailed logging

### Solution 4: Version Conflict Handler (Already Implemented)
1. **File**: `streamlit_app.py` (lines 18-35)
2. **How to use**: 
   - Automatically runs when the app starts
   - Detects version mismatches
   - Provides user-friendly error messages
   - Allows app to continue functioning

## Step-by-Step Deployment Instructions

### Option A: Automatic Fix (Recommended)
1. **Deploy normally** with existing files
2. **The version conflict handler** will automatically detect issues
3. **App will continue to function** with limited trading features
4. **User will see clear status messages** about what's working

### Option B: Manual Fix with Custom Installation
1. **Add `install_requirements.py`** to your repository
2. **Run the script** during deployment:
   ```bash
   python install_requirements.py
   ```
3. **Deploy normally** after script completion

### Option C: Use Modified Requirements
1. **Replace** `streamlit_requirements.txt` with `streamlit_requirements_override.txt`
2. **Rename** the override file to `streamlit_requirements.txt`
3. **Deploy normally**

## What Each Solution Does

### Version Conflict Handler
- ✅ **Detects** websocket version mismatches
- ✅ **Provides** clear user feedback
- ✅ **Allows** app to continue functioning
- ✅ **No deployment changes** required

### Custom Installation Script
- ✅ **Forces** specific websocket version (13.0)
- ✅ **Installs** Alpaca without dependencies
- ✅ **Handles** all package conflicts
- ✅ **Provides** detailed logging

### Modified Requirements
- ✅ **Uses** pip's `--no-deps` flag
- ✅ **Forces** websocket version first
- ✅ **Simplifies** deployment process

## Expected Behavior After Fix

### With Successful Fix:
- ✅ Alpaca Trade API imports successfully
- ✅ All trading features work normally
- ✅ No version conflict warnings
- ✅ Full functionality available

### With Partial Fix (Version Handler):
- ⚠️ Version mismatch warnings displayed
- ✅ App continues to function
- ⚠️ Some trading features may be limited
- ✅ Core features (data analysis, training) work

### Without Fix:
- ❌ Import errors during startup
- ❌ App may fail to load
- ❌ No trading functionality
- ❌ Poor user experience

## Troubleshooting

### If App Still Fails:
1. **Check Streamlit Cloud logs** for specific error messages
2. **Verify Python version** is set to 3.9.13
3. **Ensure all files** are committed to repository
4. **Try running** `setup_streamlit_cloud.py` manually

### If Trading Features Don't Work:
1. **Check** if Alpaca API keys are configured
2. **Verify** websocket version is 13.0
3. **Test** with synthetic data first
4. **Check** network connectivity

### If You See Version Warnings:
1. **This is normal** - the app will still function
2. **Trading features** may be limited
3. **Core features** (data analysis, training) will work
4. **Consider** using one of the manual fix options

## Files Summary

| File | Purpose | When to Use |
|------|---------|-------------|
| `streamlit_app.py` | Main app with version handler | Always (automatic) |
| `install_requirements.py` | Custom installation script | Manual deployment fix |
| `streamlit_requirements_override.txt` | Modified requirements | Alternative requirements file |
| `setup_streamlit_cloud.py` | Setup script | Pre-deployment setup |
| `DEPLOYMENT_GUIDE.md` | This guide | Reference |

## Quick Start
1. **Deploy normally** - the version handler will work automatically
2. **If issues persist**, use `install_requirements.py`
3. **For best results**, combine automatic handler with manual fix 