#!/usr/bin/env python3
"""
Test script to verify Streamlit Cloud deployment fixes
"""

def test_imports():
    """Test critical imports"""
    print("🧪 Testing critical imports...")
    
    # Test websockets
    try:
        import websockets
        print(f"✅ Websockets {websockets.__version__} imported successfully")
        
        # Check version compatibility
        if "9.0" <= websockets.__version__ < "11.0":
            print("✅ Websockets version is compatible with Alpaca Trade API")
        else:
            print(f"⚠️ Websockets version {websockets.__version__} may cause conflicts")
    except ImportError as e:
        print(f"❌ Failed to import websockets: {e}")
    
    # Test alpaca-trade-api
    try:
        import alpaca_trade_api
        print("✅ Alpaca Trade API imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import alpaca-trade-api: {e}")
    
    # Test TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import TensorFlow: {e}")
    
    # Test Streamlit
    try:
        import streamlit as st
        print(f"✅ Streamlit {st.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Streamlit: {e}")

def test_dependency_conflicts():
    """Test for dependency conflicts"""
    print("\n🔍 Checking for dependency conflicts...")
    
    try:
        import pkg_resources
        
        # Check websocket dependencies
        websocket_packages = [d for d in pkg_resources.working_set if 'websocket' in d.project_name.lower()]
        if websocket_packages:
            print("📦 Websocket-related packages found:")
            for pkg in websocket_packages:
                print(f"   - {pkg.project_name} {pkg.version}")
        
        # Check for conflicts
        try:
            pkg_resources.check_requirements(['websockets>=9.0,<11.0', 'alpaca-trade-api==3.2.0'])
            print("✅ No dependency conflicts detected")
        except pkg_resources.DistributionNotFound as e:
            print(f"⚠️ Missing dependency: {e}")
        except pkg_resources.VersionConflict as e:
            print(f"❌ Version conflict: {e}")
            
    except Exception as e:
        print(f"⚠️ Could not check dependencies: {e}")

if __name__ == "__main__":
    print("🚀 Testing Streamlit Cloud deployment fixes...")
    test_imports()
    test_dependency_conflicts()
    print("\n✅ Test completed!") 