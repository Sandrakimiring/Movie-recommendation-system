from fastapi.testclient import TestClient
from src.api.app import app
import pytest

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "models" in data

def test_recommend_cf():
    response = client.get("/recommend/1")
    if response.status_code == 200:
        data = response.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0
    else:
        assert response.status_code in [404, 503]

def test_recommend_content():
    response = client.get("/recommend/content/?title=Toy Story")
    if response.status_code == 200:
        data = response.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0
    else:
        assert response.status_code in [404, 503]

def test_recommend_hybrid():
    response = client.get("/recommend/hybrid/1")
    if response.status_code == 200:
        data = response.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0
    else:
        assert response.status_code in [404, 503]
