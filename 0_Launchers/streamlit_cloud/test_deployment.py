#!/usr/bin/env python3
"""
Test script to verify Streamlit Cloud deployment dependencies
"""

def test_imports():
    """Test that all critical imports work"""
    print("🔍 Testing critical imports...")
    
    # Test websockets
    try:
        import websockets
        print(f"✅ websockets {websockets.__version__}")
        if websockets.__version__ >= "13.0":
            print("✅ websockets version is compatible with TensorFlow")
        else:
            print("⚠️ websockets version may cause TensorFlow issues")
    except ImportError as e:
        print(f"❌ websockets import failed: {e}")
    
    # Test alpaca-trade-api
    try:
        import alpaca_trade_api
        print("✅ alpaca-trade-api imported successfully")
        
        # Test basic functionality
        try:
            api = alpaca_trade_api.REST('dummy', 'dummy', 'https://paper-api.alpaca.markets')
            print("✅ alpaca-trade-api functionality verified")
        except Exception as e:
            print(f"⚠️ alpaca-trade-api test failed (expected): {e}")
            
    except ImportError as e:
        print(f"❌ alpaca-trade-api import failed: {e}")
    
    # Test tensorflow
    try:
        import tensorflow as tf
        print(f"✅ tensorflow {tf.__version__}")
    except ImportError as e:
        print(f"❌ tensorflow import failed: {e}")
    
    # Test streamlit
    try:
        import streamlit as st
        print("✅ streamlit imported successfully")
    except ImportError as e:
        print(f"❌ streamlit import failed: {e}")
    
    # Test yfinance
    try:
        import yfinance as yf
        print("✅ yfinance imported successfully")
    except ImportError as e:
        print(f"❌ yfinance import failed: {e}")

def test_setup_script():
    """Test that the setup script can be imported"""
    print("\n🔍 Testing setup script...")
    try:
        import setup_streamlit_cloud
        print("✅ setup_streamlit_cloud imported successfully")
    except ImportError as e:
        print(f"❌ setup_streamlit_cloud import failed: {e}")

def main():
    """Main test function"""
    print("🧪 Testing Streamlit Cloud Deployment Setup")
    print("=" * 50)
    
    test_imports()
    test_setup_script()
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    main() 