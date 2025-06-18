#!/usr/bin/env python3
"""
Test script to verify WebSocket connections work properly
"""

import socketio
import time
import requests
import json

def test_websocket_connection():
    """Test WebSocket connection to the Flask app"""
    
    # Test HTTP endpoints first
    print("Testing HTTP endpoints...")
    try:
        response = requests.get('http://localhost:5000/')
        print(f"✓ Main page: {response.status_code}")
        
        response = requests.get('http://localhost:5000/api/get_model_status')
        print(f"✓ API endpoint: {response.status_code}")
        
    except requests.exceptions.ConnectionError:
        print("✗ Flask app not running on localhost:5000")
        print("Please start the Flask app first with: python flask_app.py")
        return False
    
    # Test Socket.IO connection
    print("\nTesting Socket.IO connection...")
    
    # Create Socket.IO client
    sio = socketio.Client()
    
    connected = False
    received_message = False
    
    @sio.event
    def connect():
        nonlocal connected
        connected = True
        print("✓ Connected to Socket.IO server")
    
    @sio.event
    def disconnect():
        print("✗ Disconnected from Socket.IO server")
    
    @sio.event
    def connected(data):
        nonlocal received_message
        received_message = True
        print(f"✓ Received connection confirmation: {data}")
    
    @sio.event
    def error(data):
        print(f"✗ Received error: {data}")
    
    try:
        # Connect to Socket.IO server
        sio.connect('http://localhost:5000')
        
        # Wait for connection and message
        timeout = 10
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if connected and received_message:
                print("✓ WebSocket connection test passed!")
                break
            time.sleep(0.1)
        else:
            print("✗ WebSocket connection test failed - timeout")
            return False
        
        # Test ping/pong
        print("\nTesting ping/pong...")
        sio.emit('ping')
        
        # Wait for pong response
        time.sleep(2)
        
        # Test status request
        print("\nTesting status request...")
        sio.emit('get_status')
        
        # Wait for status response
        time.sleep(2)
        
        # Disconnect
        sio.disconnect()
        
        print("\n✓ All WebSocket tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ WebSocket connection failed: {e}")
        return False

def test_production_endpoints():
    """Test production endpoints"""
    print("\nTesting production endpoints...")
    
    base_url = "http://tradingalgorithm-env.eba-dppmvzbf.us-west-2.elasticbeanstalk.com"
    
    try:
        # Test main page
        response = requests.get(f"{base_url}/", timeout=10)
        print(f"✓ Production main page: {response.status_code}")
        
        # Test API endpoint
        response = requests.get(f"{base_url}/api/get_model_status", timeout=10)
        print(f"✓ Production API: {response.status_code}")
        
        # Test Socket.IO endpoint
        response = requests.get(f"{base_url}/socket.io/", timeout=10)
        print(f"✓ Production Socket.IO: {response.status_code}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Production test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("           WebSocket Connection Test")
    print("=" * 60)
    
    # Test local connection
    local_success = test_websocket_connection()
    
    # Test production connection
    production_success = test_production_endpoints()
    
    print("\n" + "=" * 60)
    print("                    Test Results")
    print("=" * 60)
    print(f"Local WebSocket: {'✓ PASS' if local_success else '✗ FAIL'}")
    print(f"Production HTTP: {'✓ PASS' if production_success else '✗ FAIL'}")
    
    if local_success and production_success:
        print("\n✓ All tests passed! Ready for deployment.")
    else:
        print("\n✗ Some tests failed. Please check the issues above.") 