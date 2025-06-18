@echo off
echo ================================================================================
echo Installing Alpaca Trading API for Paper Trading
echo ================================================================================
echo.

echo Checking Python version...
python --version
echo.

echo Checking current websockets version...
python -c "import websockets; print(f'Websockets version: {websockets.__version__}')"
echo.

echo Installing alpaca-trade-api with dependency override...
echo This resolves the websockets version conflict with yfinance
echo.
python -m pip install alpaca-trade-api==3.2.0 --no-deps

echo.
echo Verifying installation...
python -c "import alpaca_trade_api; print(f'Alpaca version: {alpaca_trade_api.__version__}')"
python -c "import websockets; print(f'Websockets version: {websockets.__version__}')"
python -c "import yfinance; print(f'YFinance version: {yfinance.__version__}')"

echo.
echo ================================================================================
echo Installation complete!
echo ================================================================================
echo.
echo Next steps:
echo 1. Set your Alpaca API credentials as environment variables:
echo    - ALPACA_API_KEY=your_api_key_here
echo    - ALPACA_SECRET_KEY=your_secret_key_here
echo.
echo 2. Uncomment alpaca-trade-api in requirements.txt
echo.
echo 3. Test the trading executor:
echo    python _2_Orchestrator_And_ML_Python/trading_executor.py
echo.
pause 