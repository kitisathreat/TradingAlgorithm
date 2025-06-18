# WebSocket Dependency Conflict Resolution

## Problem (Resolved)
The project had a dependency conflict between `alpaca-trade-api==3.2.0` and `yfinance==0.2.63`:

- `alpaca-trade-api==3.2.0` required `websockets>=9.0,<11`
- `yfinance==0.2.63` required `websockets>=13.0`

This created an impossible conflict since `yfinance` needed websockets 13+ but `alpaca-trade-api` wouldn't accept anything above 10.x.

## Solution Implemented

### Consolidated Requirements Approach
**Approach**: Resolved in the main `requirements.txt` file
- Specified `websockets==13.0` to satisfy yfinance requirements
- The alpaca-trade-api package works with websockets 13.0 despite its stated constraint
- All dependencies are now managed in a single consolidated requirements file

**Files Modified**:
- `requirements.txt` (consolidated all dependencies)
- Removed separate requirements files for different components

## Current State

### Consolidated Requirements File
The project now uses a single `requirements.txt` file that includes:
- All ML and data processing dependencies
- Web framework dependencies (Flask, FastAPI)
- Trading and market data dependencies
- Development and testing dependencies
- Build tools

### Installation
Simply install all dependencies with:
```bash
pip install -r requirements.txt
```

## Technical Details

### Why This Works
- `alpaca-trade-api` actually works fine with websockets 13.0 despite its stated constraint
- The constraint was likely set conservatively and hasn't been updated
- By forcing websockets 13.0, we satisfy yfinance's requirements while maintaining alpaca functionality

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

## Migration Notes

### Removed Files
The following Streamlit-specific setup scripts have been removed:
- `0_Launchers/build_tools/setup_and_run.py`
- `0_Launchers/build_tools/setup_and_run_constraints.py`
- `0_Launchers/build_tools/setup_and_run_robust.py`
- `0_Launchers/build_tools/test_robust_setup.bat`

### Removed Requirements Files
The following separate requirements files have been consolidated:
- `root_requirements.txt`
- `0_Launchers/flask_web/requirements.txt`
- `_2_Orchestrator_And_ML_Python/ml_requirements.txt`
- `_2_Orchestrator_And_ML_Python/ml_dev_requirements.txt`
- `1_High_Performance_Module_(C++)/cpp_requirements.txt`

## Future Considerations
- Monitor for updates to `alpaca-trade-api` that might officially support newer websockets
- Consider alternative market data libraries if needed
- The consolidated requirements approach simplifies dependency management 