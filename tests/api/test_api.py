"""
Integration tests for ML Prediction API.
"""

import pytest
import requests
import time
import subprocess
import signal
import os


BASE_URL = "http://localhost:8000"


@pytest.fixture(scope="module")
def api_server():
    """Start API server for testing."""
    # Start the server
    proc = subprocess.Popen(
        ["python", "src/app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(5)
    
    yield proc
    
    # Cleanup: kill the server
    proc.send_signal(signal.SIGTERM)
    proc.wait()


def test_health_endpoint(api_server):
    """Test the health check endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    assert data['model_loaded'] == True
    assert 'model_type' in data


def test_root_endpoint(api_server):
    """Test the root endpoint."""
    response = requests.get(f"{BASE_URL}/")
    
    assert response.status_code == 200
    data = response.json()
    assert 'message' in data
    assert 'version' in data
    assert 'endpoints' in data


def test_predict_endpoint_valid_input(api_server):
    """Test prediction endpoint with valid input."""
    payload = {
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert 'prediction' in data
    assert 'confidence' in data
    assert 'class_name' in data
    assert 0 <= data['prediction'] <= 2
    assert 0 <= data['confidence'] <= 1
    assert data['class_name'] in ['setosa', 'versicolor', 'virginica']


def test_predict_endpoint_invalid_feature_count(api_server):
    """Test prediction endpoint with incorrect number of features."""
    payload = {
        "features": [5.1, 3.5, 1.4]  # Only 3 features instead of 4
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    
    assert response.status_code == 400
    data = response.json()
    assert 'detail' in data


def test_predict_endpoint_different_classes(api_server):
    """Test prediction endpoint for different iris classes."""
    test_cases = [
        [5.1, 3.5, 1.4, 0.2],  # setosa
        [6.2, 2.9, 4.3, 1.3],  # versicolor
        [6.3, 3.3, 6.0, 2.5]   # virginica
    ]
    
    for features in test_cases:
        payload = {"features": features}
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert 'prediction' in data
        assert 'confidence' in data
        assert 'class_name' in data


def test_predict_endpoint_high_confidence(api_server):
    """Test that predictions have reasonable confidence."""
    payload = {
        "features": [5.0, 3.4, 1.5, 0.2]
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data['confidence'] > 0.5  # Should have reasonable confidence
