#!/usr/bin/env python3
"""
Test script to verify the financial modeler integration with the main GUI.
"""

import sys
import os
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_financial_modeler_import():
    """Test if the financial modeler can be imported."""
    try:
        from financial_modeler_widget import FinancialModelerWidget
        print("✅ FinancialModelerWidget imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import FinancialModelerWidget: {e}")
        return False

def test_financial_modeler_creation():
    """Test if the financial modeler widget can be created."""
    try:
        from financial_modeler_widget import FinancialModelerWidget
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        widget = FinancialModelerWidget()
        print("✅ FinancialModelerWidget created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create FinancialModelerWidget: {e}")
        return False

def test_main_window_integration():
    """Test if the main window can be created with the financial modeler tab."""
    try:
        from main import MainWindow
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        window = MainWindow()
        print("✅ MainWindow with financial modeler integration created successfully")
        
        # Check if the financial modeler tab exists
        tab_count = window.main_tab_widget.count()
        print(f"   Number of tabs: {tab_count}")
        
        for i in range(tab_count):
            tab_text = window.main_tab_widget.tabText(i)
            print(f"   Tab {i}: {tab_text}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to create MainWindow with integration: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are available."""
    dependencies = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('yfinance', 'yfinance'),
        ('openpyxl', 'openpyxl'),
        ('sec_api', 'sec-api')
    ]
    
    all_available = True
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            print(f"✅ {package_name} available")
        except ImportError:
            print(f"❌ {package_name} not available")
            all_available = False
    
    return all_available

def main():
    """Run all tests."""
    print("=" * 60)
    print("FINANCIAL MODELER INTEGRATION TEST")
    print("=" * 60)
    
    # Test dependencies
    print("\n1. Testing dependencies...")
    deps_ok = test_dependencies()
    
    # Test import
    print("\n2. Testing import...")
    import_ok = test_financial_modeler_import()
    
    # Test widget creation
    print("\n3. Testing widget creation...")
    widget_ok = test_financial_modeler_creation()
    
    # Test main window integration
    print("\n4. Testing main window integration...")
    integration_ok = test_main_window_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    tests = [
        ("Dependencies", deps_ok),
        ("Import", import_ok),
        ("Widget Creation", widget_ok),
        ("Main Window Integration", integration_ok)
    ]
    
    all_passed = True
    for test_name, passed in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("The financial modeler is ready to use.")
        print("\nTo launch the GUI:")
        print("python main.py")
    else:
        print("⚠️  SOME TESTS FAILED!")
        print("Please check the error messages above.")
        print("\nTo install missing dependencies:")
        print("pip install pandas numpy yfinance openpyxl sec-api")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 