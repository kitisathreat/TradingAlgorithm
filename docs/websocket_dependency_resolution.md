# WebSocket Dependency Conflict Resolution

## Problem
The project has a dependency conflict between `alpaca-trade-api==3.2.0` and `yfinance==0.2.63`:

- `alpaca-trade-api==3.2.0` requires `websockets>=9.0,<11`
- `yfinance==0.2.63` requires `websockets>=13.0`

This creates an impossible conflict since `yfinance` needs websockets 13+ but `alpaca-trade-api` won't accept anything above 10.x.

## Solutions Implemented

### 1. Modified Setup Script (`setup_and_run.py`)
**Approach**: Two-step installation with `--no-deps`
- Force install `websockets>=13.0` first
- Install `alpaca-trade-api==3.2.0` with `--no-deps` to ignore websocket constraint
- Install remaining requirements normally

**Files Modified**:
- `0_Launchers/build_tools/setup_and_run.py`
- `requirements.txt` (commented out alpaca-trade-api)

### 2. Constraints File Approach (`setup_and_run_constraints.py`)
**Approach**: Use pip constraints to override dependency requirements
- Create `constraints.txt` with forced websockets version
- Use `pip install --constraint` to override package requirements

**Files Created**:
- `0_Launchers/streamlit_cloud/constraints.txt`
- `0_Launchers/build_tools/setup_and_run_constraints.py`

### 3. Override Requirements Approach
**Approach**: Use a separate requirements file with manual installation steps
- Create override file with specific installation order
- Install websockets first, then alpaca with `--no-deps`

**Files Created**:
- `0_Launchers/streamlit_cloud/streamlit_requirements_override.txt`

### 4. Robust Multi-Approach Script (`setup_and_run_robust.py`)
**Approach**: Try multiple installation methods in sequence
- Attempts constraints approach first
- Falls back to `--no-deps` approach
- Falls back to override approach
- Falls back to force installation approach

**Files Created**:
- `0_Launchers/build_tools/setup_and_run_robust.py`
- `0_Launchers/build_tools/test_robust_setup.bat`

## Usage

### For Streamlit Cloud Deployment
Update your Streamlit Cloud configuration to use one of these main files:

1. **Recommended**: `0_Launchers/build_tools/setup_and_run_robust.py`
   - Most reliable, tries multiple approaches
   - Best for production deployment

2. **Alternative**: `0_Launchers/build_tools/setup_and_run.py`
   - Simple two-step approach
   - Good for testing

3. **Alternative**: `0_Launchers/build_tools/setup_and_run_constraints.py`
   - Uses pip constraints
   - Clean approach but may not work in all environments

### For Local Testing
```bash
# Test the robust approach
python 0_Launchers/build_tools/setup_and_run_robust.py

# Or use the batch file on Windows
0_Launchers\build_tools\test_robust_setup.bat
```

## Technical Details

### Why This Conflict Occurs
- `alpaca-trade-api` was designed for older websockets versions
- `yfinance` was updated to require newer websockets for security/features
- Both packages are actively maintained but have different websocket requirements

### Why `--no-deps` Works
- `--no-deps` tells pip to install the package without its dependencies
- Since websockets is already installed (by our forced installation), alpaca can use it
- The websocket API is generally backward compatible within major versions

### Verification
After installation, verify the setup works:
```python
import websockets
import alpaca_trade_api
import yfinance

print(f"Websockets version: {websockets.__version__}")
print(f"Alpaca version: {alpaca_trade_api.__version__}")
print(f"YFinance version: {yfinance.__version__}")
```

## Troubleshooting

### If All Approaches Fail
1. Check if you're in a restricted environment (some cloud platforms limit pip options)
2. Try downgrading `yfinance` to an older version that accepts websockets<11
3. Consider using alternative packages for market data

### Common Issues
- **Permission errors**: Ensure you have write access to the Python environment
- **Network issues**: Some approaches require multiple pip calls
- **Cache conflicts**: Use `--force-reinstall` to clear cached packages

## Future Considerations
- Monitor for updates to `alpaca-trade-api` that might accept newer websockets
- Consider alternative market data libraries that don't have this conflict
- Evaluate if both packages are still needed for the project 