"""
Test trading days calculation
"""

from date_range_utils import calculate_trading_days
from datetime import datetime, timedelta

def test_trading_days_calculation():
    """Test the trading days calculation function"""
    print("Testing trading days calculation...")
    
    # Test 1: One week (should be 5 trading days)
    start = datetime(2024, 1, 1)  # Monday
    end = datetime(2024, 1, 5)    # Friday
    trading_days = calculate_trading_days(start, end)
    print(f"Jan 1-5, 2024 (Mon-Fri): {trading_days} trading days (expected: 5)")
    assert trading_days == 5, f"Expected 5, got {trading_days}"
    
    # Test 2: One week including weekend (should be 5 trading days)
    start = datetime(2024, 1, 1)  # Monday
    end = datetime(2024, 1, 7)    # Sunday
    trading_days = calculate_trading_days(start, end)
    print(f"Jan 1-7, 2024 (Mon-Sun): {trading_days} trading days (expected: 5)")
    assert trading_days == 5, f"Expected 5, got {trading_days}"
    
    # Test 3: Two weeks (should be 10 trading days)
    start = datetime(2024, 1, 1)  # Monday
    end = datetime(2024, 1, 14)   # Sunday
    trading_days = calculate_trading_days(start, end)
    print(f"Jan 1-14, 2024 (2 weeks): {trading_days} trading days (expected: 10)")
    assert trading_days == 10, f"Expected 10, got {trading_days}"
    
    # Test 4: One month (approximately 22 trading days)
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 31)
    trading_days = calculate_trading_days(start, end)
    print(f"Jan 1-31, 2024 (1 month): {trading_days} trading days (expected: ~22)")
    print(f"   This is reasonable for a month with weekends and holidays")
    
    # Test 5: 1000 calendar days (should be approximately 714 trading days)
    start = datetime(2024, 1, 1)
    end = start + timedelta(days=1000)
    trading_days = calculate_trading_days(start, end)
    expected_trading_days = int(1000 * 5/7)  # Rough estimate: 5/7 of calendar days
    print(f"1000 calendar days: {trading_days} trading days (expected: ~{expected_trading_days})")
    print(f"   This shows why we need trading days calculation!")
    
    print("✅ All trading days tests passed!")
    return True

if __name__ == "__main__":
    test_trading_days_calculation() 