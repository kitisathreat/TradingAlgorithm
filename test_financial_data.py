#!/usr/bin/env python3
"""
Financial Data Test Script - Run this to test financial data fetching and identify issues.
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_imports():
    """Test if all required libraries can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import yfinance as yf
        print("✅ yfinance imported successfully")
    except ImportError as e:
        print(f"❌ yfinance import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    return True

def test_yfinance_basic():
    """Test basic yfinance functionality"""
    print("\n📊 Testing yfinance basic functionality...")
    
    try:
        import yfinance as yf
        
        # Test simple ticker creation
        ticker = yf.Ticker("AAPL")
        print("✅ Ticker creation successful")
        
        # Test historical data (most important for our app)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        hist = ticker.history(start=start_date, end=end_date)
        
        if not hist.empty:
            print(f"✅ Historical data retrieval successful: {len(hist)} days")
            print(f"   Data range: {hist.index[0].date()} to {hist.index[-1].date()}")
            print(f"   Columns: {list(hist.columns)}")
            return True
        else:
            print("❌ Historical data retrieval returned empty dataframe")
            return False
            
    except Exception as e:
        print(f"❌ yfinance test failed: {e}")
        return False

def test_multiple_symbols():
    """Test fetching data for multiple symbols"""
    print("\n📈 Testing multiple symbols...")
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    success_count = 0
    
    try:
        import yfinance as yf
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)  # Short period for quick test
                
                hist = ticker.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    print(f"✅ {symbol}: {len(hist)} days")
                    success_count += 1
                else:
                    print(f"❌ {symbol}: No data returned")
                    
            except Exception as e:
                print(f"❌ {symbol}: {e}")
        
        print(f"\n📊 Success rate: {success_count}/{len(symbols)} ({success_count/len(symbols)*100:.1f}%)")
        return success_count > 0
        
    except Exception as e:
        print(f"❌ Multiple symbols test failed: {e}")
        return False

def test_market_data():
    """Test market indices (VIX, S&P 500)"""
    print("\n📉 Testing market data...")
    
    try:
        import yfinance as yf
        
        # Test VIX
        try:
            vix = yf.download('^VIX', period='5d', progress=False)
            if not vix.empty:
                print(f"✅ VIX data: {len(vix)} days")
            else:
                print("❌ VIX data: Empty")
        except Exception as e:
            print(f"❌ VIX data failed: {e}")
        
        # Test S&P 500
        try:
            spy = yf.download('^GSPC', period='5d', progress=False)
            if not spy.empty:
                print(f"✅ S&P 500 data: {len(spy)} days")
            else:
                print("❌ S&P 500 data: Empty")
        except Exception as e:
            print(f"❌ S&P 500 data failed: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ Market data test failed: {e}")
        return False

def test_model_trainer():
    """Test the ModelTrainer class"""
    print("\n🤖 Testing ModelTrainer...")
    
    try:
        # Add the orchestrator directory to Python path
        REPO_ROOT = Path(__file__).parent
        ORCHESTRATOR_PATH = REPO_ROOT / "_2_Orchestrator_And_ML_Python"
        sys.path.append(str(ORCHESTRATOR_PATH))
        
        from interactive_training_app.backend.model_trainer import ModelTrainer
        
        # Create trainer instance
        trainer = ModelTrainer()
        print("✅ ModelTrainer created successfully")
        
        # Test getting random stock data
        print("🔄 Testing random stock data generation...")
        features, feature_vector, actual_result = trainer.get_random_stock_data()
        
        if features is not None:
            print("✅ Random stock data generated successfully")
            print(f"   Symbol: {features['Symbol']}")
            print(f"   Date: {features['Date']}")
            print(f"   Current Price: ${features['Current Price']:.2f}")
            print(f"   Feature vector shape: {feature_vector.shape}")
            return True
        else:
            print("❌ Random stock data generation failed")
            return False
            
    except ImportError as e:
        print(f"❌ ModelTrainer import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ ModelTrainer test failed: {e}")
        return False

def test_environment_variables():
    """Test environment variables"""
    print("\n🔑 Testing environment variables...")
    
    required_vars = [
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY", 
        "APCA_BASE_URL",
        "FMP_API_KEY",
        "POLYGON_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var}: Set")
        else:
            print(f"❌ {var}: Missing")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️ Missing variables: {', '.join(missing_vars)}")
        print("Note: These are optional for yfinance but required for live trading")
    else:
        print("\n✅ All environment variables are set")
    
    return len(missing_vars) == 0

def main():
    """Run all tests"""
    print("🧪 Financial Data Diagnostic Test")
    print("=" * 50)
    
    results = []
    
    # Run all tests
    results.append(("Imports", test_imports()))
    results.append(("Environment Variables", test_environment_variables()))
    results.append(("YFinance Basic", test_yfinance_basic()))
    results.append(("Multiple Symbols", test_multiple_symbols()))
    results.append(("Market Data", test_market_data()))
    results.append(("ModelTrainer", test_model_trainer()))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed ({passed/len(results)*100:.1f}%)")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if not results[0][1]:  # Imports failed
        print("- Install missing dependencies: pip install -r streamlit_requirements.txt")
    
    if not results[2][1]:  # YFinance basic failed
        print("- YFinance is being blocked/rate-limited")
        print("- This is common on Streamlit Cloud due to shared IP addresses")
        print("- The app now uses fallback synthetic data when yfinance fails")
    
    if not results[3][1]:  # Multiple symbols failed
        print("- Consider using alternative data sources (Alpha Vantage, IEX, etc.)")
        print("- Implement caching to reduce API calls")
    
    if not results[1][1]:  # Environment variables missing
        print("- Add missing environment variables to Streamlit Cloud secrets")
        print("- Go to app settings > Secrets and add the API keys")
    
    print("\n🎯 For Streamlit Cloud deployment:")
    print("- The app will now use synthetic data if yfinance fails")
    print("- This ensures the training interface works even with API issues")
    print("- Real data will be used when available, fallback when not")

if __name__ == "__main__":
    main() 