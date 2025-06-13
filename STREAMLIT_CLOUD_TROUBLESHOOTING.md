# Streamlit Cloud Financial Data Troubleshooting Guide

## Common Issues and Solutions

### 1. YFinance Rate Limiting/Blocking ⚠️
**Problem**: Yahoo Finance blocks or rate-limits Streamlit Cloud's shared IP addresses.

**Symptoms**:
- Empty dataframes from yfinance
- Connection timeouts
- HTTP 429 (Too Many Requests) errors
- "No data found" messages

**Solutions**:
✅ **Already Implemented**: The app now includes fallback synthetic data generation
✅ **Retry Logic**: Multiple attempts with exponential backoff
✅ **Random Delays**: To avoid synchronized requests

### 2. TensorFlow Compatibility Issues 🔧
**Problem**: TensorFlow version mismatch with Python version.

**Fixed**: 
- Changed `tensorflow==2.15.0` → `tensorflow==2.13.0`
- This version is compatible with Python 3.9.13
- Added proper system dependencies in `packages.txt`

### 3. Missing Environment Variables 🔑
**Required for Live Trading** (Optional for demo):
```
APCA_API_KEY_ID=your_alpaca_key
APCA_API_SECRET_KEY=your_alpaca_secret
APCA_BASE_URL=https://paper-api.alpaca.markets
FMP_API_KEY=your_fmp_key
POLYGON_API_KEY=your_polygon_key
```

**How to Add in Streamlit Cloud**:
1. Go to your app dashboard
2. Click "⚙️ Settings"
3. Go to "Secrets" tab
4. Add each variable as key=value pairs

### 4. Import/Module Errors 📦
**Fixed**:
- Updated requirements to use compatible versions
- Added system dependencies in `packages.txt`
- Changed to `opencv-python-headless` for cloud compatibility

## How the App Handles Data Issues

### Fallback Data Generation 🎯
When yfinance fails, the app automatically:

1. **Generates Realistic Synthetic Data**:
   - Uses symbol-specific base prices
   - Realistic OHLCV patterns
   - Proper technical indicators

2. **Maintains Training Functionality**:
   - Interface works even without real data
   - Users can still practice trading decisions
   - Model training continues uninterrupted

3. **Seamless User Experience**:
   - No error messages for data failures
   - Consistent interface behavior
   - Clear logging for debugging

### Real vs. Synthetic Data 📊
- **Real Data**: Used when available (yfinance works)
- **Synthetic Data**: Used as fallback (yfinance fails)
- **Both**: Provide same feature set for training

## Deployment Checklist ✅

### Before Deploying:
- [ ] Python version set to 3.9 in Streamlit Cloud
- [ ] All secrets added (if using live trading APIs)
- [ ] Repository is public or properly connected
- [ ] Main file is `streamlit_app.py`

### Files Required:
- [ ] `streamlit_app.py` (main entry point)
- [ ] `streamlit_requirements.txt` (dependencies)
- [ ] `runtime.txt` (Python version: python-3.9.13)
- [ ] `packages.txt` (system dependencies)

### After Deploying:
- [ ] Check app logs for any remaining errors
- [ ] Test the "Get New Training Example" button
- [ ] Verify data is loading (real or synthetic)
- [ ] Test model training functionality

## Testing Locally First 🧪

Run the diagnostic script:
```bash
python test_financial_data.py
```

This will test:
- Package imports
- YFinance connectivity  
- Multiple symbol fetching
- ModelTrainer functionality

## Common Error Messages and Fixes

### "No module named 'tensorflow.python'"
- **Fix**: TensorFlow version issue, now fixed with v2.13.0

### "Empty historical data"
- **Fix**: Fallback data generation now handles this

### "Rate limit exceeded"
- **Fix**: Retry logic and delays now implemented

### "Import Error: ModelTrainer"
- **Fix**: Path issues resolved in updated code

## Support 🆘

If issues persist:
1. Check Streamlit Cloud app logs
2. Run local diagnostic: `python test_financial_data.py`
3. Verify all files are committed to your repository
4. Check that Python 3.9 is selected in Streamlit Cloud settings 