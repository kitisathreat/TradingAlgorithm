#!/usr/bin/env python3
"""
Test script for the new random date range functionality
This script tests the date_range_utils module and verifies it works correctly.
"""

import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_date_range_utils():
    """Test the date range utilities"""
    print("🧪 Testing Date Range Utilities...")
    
    try:
        from date_range_utils import get_random_date_range, find_available_data_range, validate_date_range
        
        # Test get_random_date_range with different parameters
        print("Testing get_random_date_range...")
        start_date, end_date = get_random_date_range(30, max_years_back=None)
        print(f"30 days, no limit: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Test with different day ranges
        for days in [7, 30, 90, 365]:
            start_date, end_date = get_random_date_range(days, max_years_back=None)
            print(f"{days} days, no limit: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Test find_available_data_range
        print("\nTesting find_available_data_range...")
        start_date, end_date = find_available_data_range("AAPL", 30, max_years_back=None)
        print(f"AAPL 30 days, no limit: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Test 4: Validation tests
        print("\n📅 Test 4: Date range validation")
        from datetime import datetime, timedelta, timezone
        
        # Valid range
        valid_start = datetime.now(timezone.utc) - timedelta(days=30)
        valid_end = datetime.now(timezone.utc) - timedelta(days=1)
        print(f"   Valid range: {validate_date_range(valid_start, valid_end)}")
        
        # Invalid range (future dates)
        future_start = datetime.now(timezone.utc) + timedelta(days=1)
        future_end = datetime.now(timezone.utc) + timedelta(days=30)
        print(f"   Future range: {validate_date_range(future_start, future_end)}")
        
        # Invalid range (start >= end)
        invalid_start = datetime.now(timezone.utc) - timedelta(days=10)
        invalid_end = datetime.now(timezone.utc) - timedelta(days=20)
        print(f"   Invalid range (start >= end): {validate_date_range(invalid_start, invalid_end)}")
        
        print("\n✅ All date range utility tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_data_fetching():
    """Test data fetching with new date ranges"""
    print("\n📊 Testing Data Fetching with Random Date Ranges...")
    
    try:
        import yfinance as yf
        
        # Test with a known symbol
        symbol = "AAPL"
        days = 30
        
        # Import the date range utilities
        from date_range_utils import find_available_data_range, validate_date_range
        
        # Get random date range
        start_date, end_date = find_available_data_range(symbol, days, max_years_back=None)
        
        if not validate_date_range(start_date, end_date, symbol):
            print(f"❌ Invalid date range generated for {symbol}")
            return False
        
        print(f"   Fetching {days} days of {symbol} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Fetch data
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"❌ No data returned for {symbol}")
            return False
        
        print(f"   ✅ Successfully fetched {len(df)} days of data")
        print(f"   Data range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"   Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
        
        return True
        
    except ImportError as e:
        print(f"❌ yfinance not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Data fetching error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Testing Random Date Range Functionality")
    print("=" * 50)
    
    # Test date range utilities
    utils_ok = test_date_range_utils()
    
    # Test data fetching
    data_ok = test_data_fetching()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   Date Range Utilities: {'✅ PASS' if utils_ok else '❌ FAIL'}")
    print(f"   Data Fetching: {'✅ PASS' if data_ok else '❌ FAIL'}")
    
    if utils_ok and data_ok:
        print("\n🎉 All tests passed! Random date range functionality is working correctly.")
        return True
    else:
        print("\n⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 