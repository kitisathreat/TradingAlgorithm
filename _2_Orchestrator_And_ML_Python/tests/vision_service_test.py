import pytest
from fastapi.testclient import TestClient
import sys
sys.path.insert(0, '../deployment_services')
from vision_service import app # Import the FastAPI app instance

@pytest.fixture
def client():
    """Provides a test client for the FastAPI app."""
    return TestClient(app)

def test_analyze_vision_endpoint(client):
    """
    Tests the /analyze_vision endpoint.
    We don't test the accuracy of deepface itself, but we test that our
    API endpoint correctly receives files and returns a structured JSON response.
    """
    # Create dummy image data (e.g., a simple 10x10 black image)
    dummy_image_bytes = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09'

    # The files must be sent in a specific tuple format for TestClient
    files = {
        'image_left': ('test_left.jpg', dummy_image_bytes, 'image/jpeg'),
        'image_right': ('test_right.jpg', dummy_image_bytes, 'image/jpeg')
    }

    # Make a POST request to the endpoint
    response = client.post("/analyze_vision", files=files)
    
    # Assert that the request was successful
    assert response.status_code == 200
    
    # Assert that the response has the expected JSON structure
    data = response.json()
    assert "dominant_emotion" in data
    assert "confidence" in data
    
    # Because deepface can't find a face in dummy bytes, we expect 'not_found'
    assert data["dominant_emotion"] == "not_found"
    assert data["confidence"] == 0.0

def test_read_main_root_endpoint(client):
    """A simple test for the root endpoint, if you choose to add one."""
    # This is useful for health checks to see if the service is running.
    # To make this pass, add this to vision_service.py:
    # @app.get("/")
    # def read_root():
    #     return {"status": "Vision Service is running"}
    
    # response = client.get("/")
    # assert response.status_code == 200
    # assert response.json() == {"status": "Vision Service is running"}
    pass # Remove this pass and uncomment above lines if you add a root endpoint
