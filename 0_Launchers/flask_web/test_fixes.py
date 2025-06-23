#!/usr/bin/env python3
"""
Test script to verify the fixes for the Flask web interface
"""

import requests
import json
import time

def test_stock_data_loading():
    """Test stock data loading with different parameters"""
    print("Testing stock data loading...")
    
    # Test 1: Basic stock loading
    print("\n1. Testing basic stock loading (AAPL, 30 days)...")
    response = requests.post('http://localhost:5000/api/load_stock', 
                           json={'symbol': 'AAPL', 'days': 30})
    data = response.json()
    
    if data.get('success'):
        print(f"✓ Success: Loaded {data['stock_info']['data_points']} data points")
        print(f"  Date range: {data['stock_info']['start_date']} to {data['stock_info']['end_date']}")
        print(f"  Current stock: {data['stock_info']['symbol']}")
    else:
        print(f"✗ Failed: {data.get('error', 'Unknown error')}")
    
    # Test 2: Different days parameter
    print("\n2. Testing different days parameter (90 days)...")
    response = requests.post('http://localhost:5000/api/load_stock', 
                           json={'symbol': 'MSFT', 'days': 90})
    data = response.json()
    
    if data.get('success'):
        print(f"✓ Success: Loaded {data['stock_info']['data_points']} data points")
        print(f"  Date range: {data['stock_info']['start_date']} to {data['stock_info']['end_date']}")
    else:
        print(f"✗ Failed: {data.get('error', 'Unknown error')}")
    
    # Test 3: Random stock selection
    print("\n3. Testing random stock selection...")
    response = requests.post('http://localhost:5000/api/load_stock', 
                           json={'symbol': 'random', 'days': 30})
    data = response.json()
    
    if data.get('success'):
        print(f"✓ Success: Random stock selected: {data['stock_info']['symbol']}")
        print(f"  Data points: {data['stock_info']['data_points']}")
    else:
        print(f"✗ Failed: {data.get('error', 'Unknown error')}")

def test_model_status():
    """Test model status and training functionality"""
    print("\nTesting model status...")
    
    response = requests.get('http://localhost:5000/api/get_model_status')
    data = response.json()
    
    if data.get('available'):
        print(f"✓ Model trainer available")
        print(f"  Training examples: {data.get('training_examples', 0)}")
        print(f"  Model state: {data.get('model_state', {}).get('is_trained', False)}")
    else:
        print("✗ Model trainer not available")

def test_stock_options():
    """Test stock options loading"""
    print("\nTesting stock options...")
    
    response = requests.get('http://localhost:5000/api/get_stock_options')
    data = response.json()
    
    if data.get('success'):
        options = data.get('options', [])
        print(f"✓ Success: Loaded {len(options)} stock options")
        
        # Check for special options
        special_options = [opt for opt in options if opt.get('type') == 'special']
        stock_options = [opt for opt in options if opt.get('type') == 'stock']
        
        print(f"  Special options: {len(special_options)}")
        print(f"  Stock options: {len(stock_options)}")
        
        # Show first few stocks
        if stock_options:
            print("  Sample stocks:", [opt['symbol'] for opt in stock_options[:5]])
    else:
        print(f"✗ Failed: {data.get('error', 'Unknown error')}")

def test_chart_data_format():
    """Test that chart data is in the correct format for candlestick charts"""
    print("\nTesting chart data format...")
    
    response = requests.post('http://localhost:5000/api/load_stock', 
                           json={'symbol': 'AAPL', 'days': 30})
    data = response.json()
    
    if data.get('success') and data.get('chart_data'):
        chart_data = data['chart_data']
        print(f"✓ Chart data loaded: {len(chart_data)} points")
        
        # Check first data point structure
        if chart_data:
            first_point = chart_data[0]
            required_fields = ['x', 'o', 'h', 'l', 'c']
            missing_fields = [field for field in required_fields if field not in first_point]
            
            if not missing_fields:
                print("✓ Chart data has correct candlestick format")
                print(f"  Sample point: {first_point}")
            else:
                print(f"✗ Chart data missing fields: {missing_fields}")
        else:
            print("✗ No chart data points")
    else:
        print("✗ Failed to load chart data")

def main():
    """Run all tests"""
    print("=" * 60)
    print("           FLASK WEB INTERFACE FIXES TEST")
    print("=" * 60)
    
    try:
        # Test basic connectivity
        print("Testing server connectivity...")
        response = requests.get('http://localhost:5000/')
        if response.status_code == 200:
            print("✓ Server is running")
        else:
            print(f"✗ Server returned status code: {response.status_code}")
            return
        
        # Run tests
        test_stock_options()
        test_stock_data_loading()
        test_chart_data_format()
        test_model_status()
        
        print("\n" + "=" * 60)
        print("           TEST COMPLETED")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server. Make sure the Flask app is running on localhost:5000")
    except Exception as e:
        print(f"✗ Test failed with error: {e}")

if __name__ == '__main__':
    main() 