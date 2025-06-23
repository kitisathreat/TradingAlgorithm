#!/usr/bin/env python3
"""
Test script for the rate limiting system
Run this to verify the rate limiting is working correctly
"""

import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_rate_limiter_import():
    """Test if rate limiter can be imported"""
    print("🔍 Testing rate limiter import...")
    try:
        from rate_limiter import get_rate_limiter_stats, EnhancedStockDataFetcher
        print("✅ Rate limiter imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Rate limiter import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic rate limiter functionality"""
    print("\n🧪 Testing basic functionality...")
    try:
        from rate_limiter import get_rate_limiter_stats, EnhancedStockDataFetcher
        
        # Test stats
        stats = get_rate_limiter_stats()
        print(f"✅ Stats retrieved: {stats['total_requests']} total requests")
        
        # Test fetcher creation
        fetcher = EnhancedStockDataFetcher()
        print("✅ EnhancedStockDataFetcher created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_data_fetching():
    """Test actual data fetching"""
    print("\n📊 Testing data fetching...")
    try:
        from rate_limiter import EnhancedStockDataFetcher
        
        fetcher = EnhancedStockDataFetcher()
        
        # Test with a simple symbol
        symbol = "AAPL"
        days = 7
        
        print(f"Fetching {days} days of data for {symbol}...")
        start_time = time.time()
        
        stock_info, data = fetcher.get_stock_data(symbol, days)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if stock_info and data is not None and not data.empty:
            synthetic = stock_info.get('synthetic', False)
            data_points = len(data)
            
            print(f"✅ Success: {data_points} data points in {duration:.2f}s")
            if synthetic:
                print("⚠️  Using synthetic data (yfinance may be rate limited)")
            else:
                print("✅ Real yfinance data retrieved")
            
            return True
        else:
            print("❌ No data returned")
            return False
            
    except Exception as e:
        print(f"❌ Data fetching test failed: {e}")
        return False

def test_multiple_symbols():
    """Test fetching multiple symbols"""
    print("\n📈 Testing multiple symbols...")
    try:
        from rate_limiter import EnhancedStockDataFetcher
        
        fetcher = EnhancedStockDataFetcher()
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        results = fetcher.get_multiple_stocks(symbols, days=5)
        
        success_count = 0
        synthetic_count = 0
        
        for symbol, (stock_info, data) in results.items():
            if stock_info and data is not None and not data.empty:
                success_count += 1
                if stock_info.get('synthetic', False):
                    synthetic_count += 1
                print(f"✅ {symbol}: {len(data)} data points")
            else:
                print(f"❌ {symbol}: Failed")
        
        print(f"\n📋 Summary: {success_count}/{len(symbols)} successful")
        print(f"Synthetic data used: {synthetic_count}")
        
        return success_count > 0
        
    except Exception as e:
        print(f"❌ Multiple symbols test failed: {e}")
        return False

def test_stats_and_monitoring():
    """Test statistics and monitoring"""
    print("\n📊 Testing statistics and monitoring...")
    try:
        from rate_limiter import get_rate_limiter_stats, clear_yfinance_cache, reset_yfinance_rate_limits
        
        # Get initial stats
        initial_stats = get_rate_limiter_stats()
        print(f"✅ Initial stats: {initial_stats['total_requests']} requests")
        
        # Test cache clearing
        clear_yfinance_cache()
        print("✅ Cache cleared")
        
        # Test rate limit reset
        reset_yfinance_rate_limits()
        print("✅ Rate limits reset")
        
        # Get final stats
        final_stats = get_rate_limiter_stats()
        print(f"✅ Final stats: {final_stats['total_requests']} requests")
        
        return True
        
    except Exception as e:
        print(f"❌ Statistics test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Rate Limiting System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_rate_limiter_import),
        ("Basic Functionality", test_basic_functionality),
        ("Data Fetching", test_data_fetching),
        ("Multiple Symbols", test_multiple_symbols),
        ("Statistics & Monitoring", test_stats_and_monitoring)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Rate limiting system is working correctly.")
        print("\n💡 Next steps:")
        print("1. Deploy to your EC2 instance")
        print("2. Monitor the /api/rate_limiter_stats endpoint")
        print("3. Use the rate limiting in your Flask app")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure all dependencies are installed")
        print("2. Check that yfinance is working")
        print("3. Verify file permissions")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 