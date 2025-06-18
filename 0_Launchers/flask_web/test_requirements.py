#!/usr/bin/env python3
"""
Test script to verify all requirements work properly
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        if package_name:
            importlib.import_module(package_name)
        else:
            importlib.import_module(module_name)
        print(f"✅ {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {module_name}: {e}")
        return False

def main():
    """Test all required modules"""
    print("Testing Trading Algorithm Web Interface Requirements")
    print("=" * 50)
    
    # Core Flask dependencies
    print("\n🔧 Core Flask Dependencies:")
    flask_modules = [
        "Flask",
        "Flask-SocketIO",
        "python-socketio",
        "python-engineio",
        "Werkzeug",
        "python-dotenv",
        "requests"
    ]
    
    flask_success = 0
    for module in flask_modules:
        if test_import(module):
            flask_success += 1
    
    # Production server
    print("\n🚀 Production Server:")
    prod_modules = [
        "gunicorn",
        "eventlet"
    ]
    
    prod_success = 0
    for module in prod_modules:
        if test_import(module):
            prod_success += 1
    
    # ML and data processing
    print("\n🤖 ML and Data Processing:")
    ml_modules = [
        ("tensorflow", "tensorflow"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("joblib", "joblib")
    ]
    
    ml_success = 0
    for module, package in ml_modules:
        if test_import(module, package):
            ml_success += 1
    
    # Trading and market data
    print("\n📈 Trading and Market Data:")
    trading_modules = [
        ("yfinance", "yfinance"),
        ("lxml", "lxml"),
        ("tqdm", "tqdm"),
        ("websockets", "websockets")
    ]
    
    trading_success = 0
    for module, package in trading_modules:
        if test_import(module, package):
            trading_success += 1
    
    # Sentiment analysis
    print("\n💭 Sentiment Analysis:")
    sentiment_modules = [
        ("vaderSentiment", "vaderSentiment"),
        ("textblob", "textblob")
    ]
    
    sentiment_success = 0
    for module, package in sentiment_modules:
        if test_import(module, package):
            sentiment_success += 1
    
    # System and performance
    print("\n⚙️ System and Performance:")
    system_modules = [
        ("psutil", "psutil"),
        ("tenacity", "tenacity")
    ]
    
    system_success = 0
    for module, package in system_modules:
        if test_import(module, package):
            system_success += 1
    
    # Security and utilities
    print("\n🔒 Security and Utilities:")
    security_modules = [
        ("cryptography", "cryptography")
    ]
    
    security_success = 0
    for module, package in security_modules:
        if test_import(module, package):
            security_success += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    print(f"Flask Dependencies: {flask_success}/{len(flask_modules)}")
    print(f"Production Server: {prod_success}/{len(prod_modules)}")
    print(f"ML and Data Processing: {ml_success}/{len(ml_modules)}")
    print(f"Trading and Market Data: {trading_success}/{len(trading_modules)}")
    print(f"Sentiment Analysis: {sentiment_success}/{len(sentiment_modules)}")
    print(f"System and Performance: {system_success}/{len(system_modules)}")
    print(f"Security and Utilities: {security_success}/{len(security_modules)}")
    
    total_modules = len(flask_modules) + len(prod_modules) + len(ml_modules) + len(trading_modules) + len(sentiment_modules) + len(system_modules) + len(security_modules)
    total_success = flask_success + prod_success + ml_success + trading_success + sentiment_success + system_success + security_success
    
    print(f"\nOverall: {total_success}/{total_modules} modules imported successfully")
    
    if total_success == total_modules:
        print("🎉 All requirements are working correctly!")
        return 0
    else:
        print("⚠️  Some requirements failed to import. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 