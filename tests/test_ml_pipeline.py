import os
import sys
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(REPO_ROOT, "Frontend_BackEnd")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import Frontend_BackEnd.server as server_module

@pytest.fixture(scope="session", autouse=True)
def load_models():
    server_module.load_all_artifacts()

@pytest.fixture
def client():
    return TestClient(server_module.app)

@pytest.fixture
def sample_customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 1026.0,
        "selected_model": "XGBoost"
    }

def test_models_loaded():
    assert "XGBoost" in server_module.MODELS
    assert "Logistic Regression" in server_module.MODELS
    assert "PyTorch MLP" in server_module.MODELS
    assert server_module.CLUSTER_MODEL is not None
    assert len(server_module.COLUMNS) > 0

def test_multi_model_scoring(client, sample_customer):
    res = client.post("/api/predict", json=sample_customer)
    assert res.status_code == 200
    data = res.json()
    
    assert "churn_probability" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
    
    scores = data["all_model_scores"]
    for model_name in ["XGBoost", "Random Forest", "Logistic Regression", "PyTorch MLP"]:
        assert model_name in scores
        assert 0.0 <= scores[model_name] <= 1.0

def test_explainability_and_actionable_insights(client, sample_customer):
    res = client.post("/api/predict", json=sample_customer)
    assert res.status_code == 200
    data = res.json()
    
    assert "factor_insights" in data and len(data["factor_insights"]) > 0
    assert "recommendations" in data and len(data["recommendations"]) > 0
    assert "phone_script" in data and len(data["phone_script"]) > 10
    assert "cluster_info" in data
    assert "name" in data["cluster_info"]

def test_batch_prediction_csv_and_json(client):
    # CSV batch
    csv_content = (
        "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges\n"
        "CUST-001,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85\n"
        "CUST-002,Male,0,No,No,34,Yes,No,DSL,Yes,No,Yes,No,No,No,One year,No,Mailed check,56.95,1889.5\n"
    )
    res_csv = client.post("/api/batch-predict", files={"file": ("batch.csv", csv_content.encode("utf-8"), "text/csv")})
    assert res_csv.status_code == 200
    batch_data = res_csv.json()
    assert batch_data["total_processed"] == 2
    assert "high_risk_count" in batch_data
    assert len(batch_data["records"]) == 2

    # JSON batch
    json_content = b'[{"tenure": 5, "MonthlyCharges": 70.0, "Contract": "Month-to-month"}]'
    res_json = client.post("/api/batch-predict", files={"file": ("batch.json", json_content, "application/json")})
    assert res_json.status_code == 200
    assert res_json.json()["total_processed"] == 1
