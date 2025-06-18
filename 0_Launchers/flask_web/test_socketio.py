#!/usr/bin/env python3
"""
Test script to verify SocketIO configuration
"""

import requests
import json

def test_socketio_connection():
    """Test SocketIO connection"""
    base_url = "http://localhost:8000"
    
    try:
        # Test basic Flask app
        response = requests.get(f"{base_url}/")
        print(f"Flask app response: {response.status_code}")
        
        # Test SocketIO endpoint
        response = requests.get(f"{base_url}/socket.io/")
        print(f"SocketIO endpoint response: {response.status_code}")
        
        # Test API endpoints
        response = requests.get(f"{base_url}/api/get_stock_options")
        print(f"Stock options API response: {response.status_code}")
        
        response = requests.get(f"{base_url}/api/get_model_status")
        print(f"Model status API response: {response.status_code}")
        
    except Exception as e:
        print(f"Error testing connection: {e}")

if __name__ == "__main__":
    test_socketio_connection() 