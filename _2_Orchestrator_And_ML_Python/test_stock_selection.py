#!/usr/bin/env python3
"""
Test script for enhanced stock selection features
"""

import sys
from pathlib import Path

# Add the orchestrator path
ORCHESTRATOR_PATH = Path(__file__).parent
sys.path.append(str(ORCHESTRATOR_PATH))

from stock_selection_utils import StockSelectionManager

def test_stock_selection():
    """Test the enhanced stock selection features"""
    print("🧪 Testing Enhanced Stock Selection Features")
    print("=" * 50)
    
    try:
        # Initialize stock selection manager
        stock_manager = StockSelectionManager()
        print("✅ StockSelectionManager initialized successfully")
        
        # Test 1: Get random stock
        print("\n🎲 Testing Random Stock Selection:")
        symbol, name = stock_manager.get_random_stock()
        print(f"   Random pick: {symbol} ({name})")
        
        # Test 2: Get optimized pick
        print("\n🚀 Testing Optimized Pick:")
        symbol, name = stock_manager.get_optimized_pick()
        print(f"   Optimized pick: {symbol} ({name})")
        
        # Test 3: Validate custom ticker (valid)
        print("\n✏️ Testing Custom Ticker Validation (Valid):")
        is_valid, message, company_name = stock_manager.validate_custom_ticker("AAPL")
        print(f"   AAPL: {message}")
        if company_name:
            print(f"   Company: {company_name}")
        
        # Test 4: Validate custom ticker (invalid)
        print("\n✏️ Testing Custom Ticker Validation (Invalid):")
        is_valid, message, company_name = stock_manager.validate_custom_ticker("INVALID123")
        print(f"   INVALID123: {message}")
        
        # Test 5: Get stock options
        print("\n📋 Testing Stock Options:")
        options = stock_manager.get_stock_options()
        print(f"   Total options: {len(options)}")
        print(f"   Special options: {len([opt for opt in options if opt['symbol'] in ['random', 'optimized', 'custom']])}")
        print(f"   S&P100 stocks: {len([opt for opt in options if opt['symbol'] not in ['random', 'optimized', 'custom']])}")
        
        # Test 6: Display names and descriptions
        print("\n📝 Testing Display Names and Descriptions:")
        test_symbols = ["random", "optimized", "custom", "AAPL", "MSFT"]
        for symbol in test_symbols:
            display_name = stock_manager.get_stock_display_name(symbol)
            description = stock_manager.get_stock_description(symbol)
            print(f"   {symbol}: {display_name} - {description}")
        
        # Test 7: S&P100 data loading
        print("\n📊 Testing S&P100 Data:")
        print(f"   Total S&P100 stocks: {len(stock_manager.sp100_data)}")
        print(f"   Top 5 by market cap:")
        for i, stock in enumerate(stock_manager.sp100_data[:5]):
            market_cap_billions = stock['market_cap'] / 1_000_000_000
            print(f"   {i+1}. {stock['symbol']} - {stock['name']} (${market_cap_billions:.1f}B)")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_stock_selection() 