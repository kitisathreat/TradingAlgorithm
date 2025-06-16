#!/usr/bin/env python3
"""
Test script to verify yfinance date fixes and Polygon API
"""

import sys
import os
from datetime import datetime, timedelta
import time

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '_2_Orchestrator_And_ML_Python', 'interactive_training_app', 'backend'))

try:
    import yfinance as yf
    from model_trainer import ModelTrainer
    import requests
    
    # Try to load dotenv if available
    try:
        from dotenv import load_dotenv
        dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(dotenv_path=dotenv_path)
        print(f"✅ Successfully imported required modules and loaded .env file from {dotenv_path}")
    except ImportError:
        print("✅ Successfully imported required modules (dotenv not available)")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_current_date():
    """Test that current date is correct"""
    now = datetime.now()
    print(f"Current system time: {now}")
    
    if now.year > 2024:
        print("⚠️  Warning: System date appears to be in the future!")
        return False
    elif now.year < 2020:
        print("⚠️  Warning: System date appears to be too far in the past!")
        return False
    else:
        print("✅ System date appears reasonable")
        return True

def test_polygon_api():
    """Test Polygon API functionality for recent data (15 to 10 days ago)."""
    print("\n🔷 Testing Polygon API...")
    
    # Get API key from environment
    api_key = os.environ.get('Polygon_API_KEY')
    if not api_key:
        print("❌ Polygon_API_KEY not found in environment variables")
        return False
    
    print(f"✅ Found Polygon API key: {api_key[:8]}...")
    
    # Use a recent date range (15 to 10 days ago)
    today = datetime.now().date()
    start_date = (today - timedelta(days=15)).strftime('%Y-%m-%d')
    end_date = (today - timedelta(days=10)).strftime('%Y-%m-%d')
    symbol = "AAPL"
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?apiKey={api_key}"
    
    try:
        print(f"Fetching {symbol} data from Polygon for {start_date} to {end_date}...")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('results'):
                print(f"✅ Polygon API: Got {len(data['results'])} days of data for {symbol}")
                print(f"   Data range: {data['results'][0]['t']} to {data['results'][-1]['t']}")
                return True
            else:
                print(f"❌ Polygon API: No results returned for {symbol}")
                return False
        else:
            print(f"❌ Polygon API: HTTP {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Polygon API test failed: {e}")
        return False

def test_yfinance_basic():
    """Test basic yfinance functionality with proper dates"""
    print("\n📊 Testing yfinance with proper date ranges...")
    
    try:
        # Test with a known reliable symbol
        symbol = "AAPL"
        ticker = yf.Ticker(symbol)
        
        # Use a reference date that yfinance should definitely have data for
        # Since today is June 16, 2025, let's use a date from 2024 that we know exists
        reference_date = datetime(2024, 12, 20)  # December 20, 2024 - should have data
        end_date = reference_date
        start_date = end_date - timedelta(days=10)  # 10 days before that
        
        print(f"Fetching data for {symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (using 2024 reference date)")
        
        data = ticker.history(start=start_date, end=end_date)
        
        if not data.empty:
            print(f"✅ Successfully fetched {len(data)} days of data")
            print(f"   Data range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            print(f"   Latest price: ${data['Close'].iloc[-1]:.2f}")
            
            # Check that data ends at our reference date
            if data.index.max() <= end_date:
                print("✅ Data ends at reference date (as expected)")
                return True
            else:
                print("⚠️  Warning: Data contains dates newer than expected!")
                return False
        else:
            print("❌ No data returned")
            return False
            
    except Exception as e:
        print(f"❌ yfinance test failed: {e}")
        return False

def test_model_trainer():
    """Test ModelTrainer with proper date handling"""
    print("\n🤖 Testing ModelTrainer date handling...")
    
    try:
        trainer = ModelTrainer()
        print(f"✅ ModelTrainer initialized with {len(trainer.symbols)} symbols")
        
        # Test with a single symbol
        symbol = "AAPL"
        print(f"Testing data fetch for {symbol}...")
        
        data = trainer.get_historical_stock_data(symbol, years_back=1)  # Just 1 year
        
        if not data.empty:
            print(f"✅ Successfully fetched {len(data)} days of data")
            print(f"   Data range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            
            # Check for future dates
            if data.index.max() > datetime.now():
                print("⚠️  Warning: ModelTrainer data contains future dates!")
                return False
            else:
                print("✅ ModelTrainer data has no future dates")
                return True
        else:
            print("❌ ModelTrainer returned empty data")
            return False
            
    except Exception as e:
        print(f"❌ ModelTrainer test failed: {e}")
        return False

def test_yahoo_data_any():
    """Try to fetch any data for AAPL, MSFT, GOOG for a range in 2023."""
    print("\n🔎 Testing if any data is available from Yahoo Finance for 2023...")
    symbols = ["AAPL", "MSFT", "GOOG"]
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    found_any = False
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            print(f"Trying {symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            data = ticker.history(start=start_date, end=end_date)
            if not data.empty:
                print(f"✅ {symbol}: Got {len(data)} days of data. Range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
                found_any = True
            else:
                print(f"❌ {symbol}: No data returned.")
        except Exception as e:
            print(f"❌ {symbol}: Exception: {e}")
    return found_any

def test_yahoo_rate_limit():
    """Test if yfinance is being rate limited by making multiple rapid requests."""
    print("\n🚦 Testing for Yahoo Finance rate limiting...")
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "SPY", "QQQ"]
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)
    errors = 0
    for i in range(3):  # Reduced to 3 iterations for faster testing
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                print(f"[{i+1}] Fetching {symbol}...")
                data = ticker.history(start=start_date, end=end_date)
                if data.empty:
                    print(f"   ❌ {symbol}: No data returned.")
                    errors += 1
                else:
                    print(f"   ✅ {symbol}: Got {len(data)} days of data.")
            except Exception as e:
                print(f"   ❌ {symbol}: Exception: {e}")
                errors += 1
        time.sleep(1)  # Short pause to avoid instant hammering
    if errors == 0:
        print("✅ No rate limiting detected (all requests succeeded)")
        return True
    else:
        print(f"⚠️  Detected {errors} errors/empty responses. Possible rate limiting or data issue.")
        return False

def test_custom_date_range():
    """Test yfinance with the specific date range: 06/01/2024 to 06/01/2025"""
    print("\n📊 Testing yfinance with custom date range: 06/01/2024 to 06/01/2025...")
    
    try:
        # Test with multiple symbols
        symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "SPY", "QQQ"]
        start_date = datetime(2024, 6, 1)  # 06/01/2024
        end_date = datetime(2025, 6, 1)    # 06/01/2025
        
        print(f"Testing date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Testing symbols: {', '.join(symbols)}")
        
        successful_symbols = []
        failed_symbols = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                print(f"\nFetching {symbol}...")
                
                data = ticker.history(start=start_date, end=end_date)
                
                if not data.empty:
                    print(f"✅ {symbol}: Got {len(data)} days of data")
                    print(f"   Data range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
                    print(f"   Latest price: ${data['Close'].iloc[-1]:.2f}")
                    print(f"   Highest price: ${data['High'].max():.2f}")
                    print(f"   Lowest price: ${data['Low'].min():.2f}")
                    successful_symbols.append(symbol)
                else:
                    print(f"❌ {symbol}: No data returned")
                    failed_symbols.append(symbol)
                    
            except Exception as e:
                print(f"❌ {symbol}: Exception - {e}")
                failed_symbols.append(symbol)
        
        print(f"\n📈 RESULTS SUMMARY:")
        print(f"✅ Successful symbols ({len(successful_symbols)}): {', '.join(successful_symbols)}")
        if failed_symbols:
            print(f"❌ Failed symbols ({len(failed_symbols)}): {', '.join(failed_symbols)}")
        
        return len(successful_symbols) > 0
        
    except Exception as e:
        print(f"❌ Custom date range test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("           FINANCIAL DATA SOURCE TEST")
    print("=" * 60)
    
    # Test system date
    date_ok = test_current_date()
    
    # Test custom date range (06/01/2024 to 06/01/2025)
    custom_range_ok = test_custom_date_range()
    
    # Test Polygon API
    polygon_ok = test_polygon_api()
    
    # Test yfinance
    yf_ok = test_yfinance_basic()
    
    # Test ModelTrainer
    trainer_ok = test_model_trainer()
    
    # Test if any data is available at all from Yahoo
    yahoo_ok = test_yahoo_data_any()
    
    # Test for rate limiting
    rate_limit_ok = test_yahoo_rate_limit()
    
    print("\n" + "=" * 60)
    print("                    RESULTS")
    print("=" * 60)
    
    if date_ok and custom_range_ok and polygon_ok and yf_ok and trainer_ok and yahoo_ok and rate_limit_ok:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Date handling is working correctly")
        print("✅ Custom date range (06/01/2024 to 06/01/2025) is working")
        print("✅ Polygon API is working")
        print("✅ yfinance is fetching proper data")
        print("✅ ModelTrainer is working with correct dates")
        print("✅ Yahoo Finance is returning data for 2023")
        print("✅ No rate limiting detected")
    else:
        print("❌ SOME TESTS FAILED!")
        if not date_ok:
            print("❌ System date issue detected")
        if not custom_range_ok:
            print("❌ Custom date range test failed")
        if not polygon_ok:
            print("❌ Polygon API issue detected")
        if not yf_ok:
            print("❌ yfinance date issue detected")
        if not trainer_ok:
            print("❌ ModelTrainer date issue detected")
        if not yahoo_ok:
            print("❌ Yahoo Finance is not returning any data for 2023!")
        if not rate_limit_ok:
            print("❌ Possible Yahoo Finance rate limiting or data issue!")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 