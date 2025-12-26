import requests
import json
import pytest


BASE_URL = "http://localhost:8000"


@pytest.fixture(scope="session", autouse=True)
def setup_model():
    run_id = "acae974042f640869d5682af0d550f3a"

    response = requests.post(f"{BASE_URL}/update-model", json={"run_id": run_id})

    if response.status_code != 200:
        pytest.skip(f"Could not load model: {response.text}")


def test_predict_setosa():
    test_data = {"features": [[5.1, 3.5, 1.4, 0.2]]}
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result
    assert result["predictions"][0] == 0


def test_predict_versicolor():
    test_data = {"features": [[6.2, 2.9, 4.3, 1.3]]}
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result
    assert result["predictions"][0] == 1


def test_predict_virginica():
    test_data = {"features": [[6.3, 3.3, 6.0, 2.5]]}
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result
    assert result["predictions"][0] == 2


def test_predict_multiple():
    test_data = {
        "features": [[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3], [6.3, 3.3, 6.0, 2.5]]
    }
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result
    assert len(result["predictions"]) == 3


def test_predict_invalid_features():
    test_data = {"features": [[5.1, 3.5, 1.4]]}
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    assert response.status_code == 400


def test_predict_edge_case_min():
    test_data = {"features": [[4.3, 3.0, 1.1, 0.1]]}
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result


def test_predict_edge_case_max():
    test_data = {"features": [[7.9, 4.4, 6.9, 2.5]]}
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result


def test_predict_empty_features():
    test_data = {"features": []}
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    assert response.status_code == 400


def test_predict_missing_features():
    test_data = {}
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    assert response.status_code == 400
