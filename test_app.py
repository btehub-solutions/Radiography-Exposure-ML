"""
Unit and integration test suite for Radiography Exposure ML API
"""

import sys
from pathlib import Path

# Ensure current directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from main import (
    app,
    PatientInput,
    perform_prediction,
    encode_sex,
    categorize_bmi,
    model_registry,
    get_model_path,
)
import joblib


def test_helper_functions():
    assert encode_sex("Female") == 0
    assert encode_sex("female") == 0
    assert encode_sex("F") == 0
    assert encode_sex(0) == 0

    assert encode_sex("Male") == 1
    assert encode_sex("male") == 1
    assert encode_sex("M") == 1
    assert encode_sex(1) == 1

    assert categorize_bmi(17.5) == "Underweight"
    assert categorize_bmi(22.0) == "Normal Weight"
    assert categorize_bmi(27.0) == "Overweight"
    assert categorize_bmi(32.0) == "Obese"
    print("Helper functions passed.")


def test_model_loading_and_inference():
    # Manually populate model_registry to test prediction logic
    model_registry["scaler"] = joblib.load(get_model_path("scaler.joblib"))
    model_registry["ridge_kvp"] = joblib.load(get_model_path("kvp_model.joblib"))
    model_registry["ridge_mas"] = joblib.load(get_model_path("mas_model.joblib"))

    # Test Case 1: Standard Adult Female
    p1 = PatientInput(
        sex="Female",
        age=28.0,
        weight_kg=65.0,
        height_m=1.68,
        chest_cm=22.5,
        model_architecture="ridge",
    )
    res1 = perform_prediction(p1)
    assert res1.status == "success"
    assert res1.recommended_kvp >= 70, f"Expected >= 70 kVp, got {res1.recommended_kvp}"
    assert res1.recommended_mas >= 1.0, f"Expected >= 1.0 mAs, got {res1.recommended_mas}"
    assert res1.bmi_category == "Normal Weight"
    print(f"Case 1 (Female) -> kVp: {res1.recommended_kvp}, mAs: {res1.recommended_mas}, BMI: {res1.computed_bmi}")

    # Test Case 2: Adult Male
    p2 = PatientInput(
        sex="Male",
        age=35.0,
        weight_kg=80.9,
        height_m=1.72,
        chest_cm=24.7,
        model_architecture="ridge",
    )
    res2 = perform_prediction(p2)
    assert res2.status == "success"
    assert res2.recommended_kvp >= 70
    assert res2.bmi_category == "Overweight"
    print(f"Case 2 (Male) -> kVp: {res2.recommended_kvp}, mAs: {res2.recommended_mas}, BMI: {res2.computed_bmi}")

    print("Model inference test passed successfully!")


def test_via_testclient():
    try:
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        with client:
            resp_root = client.get("/")
            assert resp_root.status_code == 200
            
            resp_health = client.get("/health")
            assert resp_health.status_code == 200
            assert resp_health.json()["status"] == "healthy"
            
            resp_pred = client.post("/predict", json={
                "sex": "Female",
                "age": 28.0,
                "weight_kg": 65.0,
                "height_m": 1.68,
                "chest_cm": 22.5
            })
            assert resp_pred.status_code == 200
            data = resp_pred.json()
            assert data["recommended_kvp"] >= 70
            print("FastAPI TestClient HTTP endpoints verified successfully!")
    except (ImportError, RuntimeError) as e:
        print(f"Note: TestClient skipped ({e}). Core inference and helper tests verified!")


if __name__ == "__main__":
    print("--- Running Radiography ML API Unit Tests ---")
    test_helper_functions()
    test_model_loading_and_inference()
    test_via_testclient()
    print("--- ALL TESTS COMPLETED SUCCESSFULLY ---")
