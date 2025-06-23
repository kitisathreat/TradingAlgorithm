#!/usr/bin/env python3
"""
Simple test script to check yfinance functionality on EC2
"""

import sys
import traceback
from datetime import datetime, timedelta

print("=== YFINANCE SIMPLE TEST ===")
print(f"Python version: {sys.version}")
print(f"Started at: {datetime.now()}")

try:
    print("\n1. Testing yfinance import...")
    import yfinance as yf
    print(f"✓ yfinance imported successfully, version: {yf.__version__}")
except Exception as e:
    print(f"✗ yfinance import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n2. Testing pandas import...")
    import pandas as pd
    print(f"✓ pandas imported successfully, version: {pd.__version__}")
except Exception as e:
    print(f"✗ pandas import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n3. Testing basic yfinance ticker creation...")
    ticker = yf.Ticker('AAPL')
    print("✓ Ticker created successfully")
except Exception as e:
    print(f"✗ Ticker creation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n4. Testing stock data fetch...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    
    print(f"Fetching AAPL data from {start_date.date()} to {end_date.date()}...")
    data = ticker.history(start=start_date, end=end_date)
    
    if data.empty:
        print("✗ No data returned - this is the problem!")
        print("This suggests yfinance cannot fetch data from Yahoo Finance")
    else:
        print(f"✓ SUCCESS: Retrieved {len(data)} rows of data")
        print(f"Columns: {list(data.columns)}")
        print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"Latest close: ${data['Close'].iloc[-1]:.2f}")
        print(f"Latest volume: {data['Volume'].iloc[-1]:,}")
        
except Exception as e:
    print(f"✗ Data fetch failed: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n5. Testing stock info fetch...")
    info = ticker.info
    if info:
        print("✓ Stock info fetched successfully")
        print(f"Company: {info.get('longName', 'Unknown')}")
        print(f"Market Cap: ${info.get('marketCap', 0):,}")
    else:
        print("✗ No stock info returned")
except Exception as e:
    print(f"✗ Stock info fetch failed: {e}")
    traceback.print_exc()

print("\n=== TEST COMPLETED ===")
print("If step 4 failed, this explains why your app can't load stock data.")
print("Common causes:")
print("- Network connectivity issues")
print("- Yahoo Finance API changes")
print("- Rate limiting")
print("- Firewall blocking outbound connections") 