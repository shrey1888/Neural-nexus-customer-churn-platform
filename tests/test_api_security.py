import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure repo root and Backend directory are in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(REPO_ROOT, "Frontend_BackEnd")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import Frontend_BackEnd.server as server_module

@pytest.fixture(scope="session", autouse=True)
def setup_models():
    """Ensures all ML models and artifacts are loaded into memory for the test session."""
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

def test_root_serves_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "TelcoPulse" in response.text

def test_security_headers_and_csp(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    headers = response.headers
    assert headers.get("x-content-type-options") == "nosniff"
    assert headers.get("x-frame-options") == "DENY"
    assert headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert headers.get("permissions-policy") == "camera=(), microphone=(), geolocation=()"
    
    csp = headers.get("content-security-policy", "")
    assert "script-src 'self' https://fonts.googleapis.com;" in csp
    script_part = csp.split("script-src")[1].split(";")[0]
    assert "'unsafe-inline'" not in script_part
    assert "'unsafe-eval'" not in csp

def test_cors_preflight_and_headers(client):
    # Preflight OPTIONS request
    res = client.options("/api/status", headers={
        "Origin": "http://localhost:8000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type, X-API-Key, Authorization"
    })
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://localhost:8000"
    allowed_hdrs = res.headers.get("access-control-allow-headers", "").lower()
    assert "x-api-key" in allowed_hdrs or "*" in allowed_hdrs

def test_production_docs_disabled_by_default(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404

def test_auth_full_matrix(client, sample_customer):
    original_auth_req = server_module.REQUIRE_AUTH
    original_secrets = server_module.VALID_AUTH_SECRETS
    try:
        server_module.REQUIRE_AUTH = True
        server_module.VALID_AUTH_SECRETS = ["test-bearer-token", "test-api-key-999"]

        # 1. No credentials -> 401
        res_no_key = client.post("/api/predict", json=sample_customer)
        assert res_no_key.status_code == 401
        assert "WWW-Authenticate" in res_no_key.headers

        # 2. Wrong API Key -> 401
        res_wrong_key = client.post("/api/predict", json=sample_customer, headers={"X-API-Key": "bad-key"})
        assert res_wrong_key.status_code == 401
        assert "invalid_token" in res_wrong_key.headers.get("WWW-Authenticate", "")

        # 3. Wrong Bearer -> 401
        res_wrong_bearer = client.post("/api/predict", json=sample_customer, headers={"Authorization": "Bearer bad-bearer"})
        assert res_wrong_bearer.status_code == 401
        assert "invalid_token" in res_wrong_bearer.headers.get("WWW-Authenticate", "")

        # 4. Valid API Key -> 200
        res_valid_key = client.post("/api/predict", json=sample_customer, headers={"X-API-Key": "test-api-key-999"})
        assert res_valid_key.status_code == 200
        assert "churn_probability" in res_valid_key.json()

        # 5. Valid Bearer -> 200
        res_valid_bearer = client.post("/api/predict", json=sample_customer, headers={"Authorization": "Bearer test-bearer-token"})
        assert res_valid_bearer.status_code == 200
        assert "churn_probability" in res_valid_bearer.json()

        # 6. Protected customer directory endpoint
        res_cust_unauth = client.get("/api/customers")
        assert res_cust_unauth.status_code == 401
        res_cust_auth = client.get("/api/customers?limit=5", headers={"X-API-Key": "test-api-key-999"})
        assert res_cust_auth.status_code == 200
        assert "customers" in res_cust_auth.json()

        # 7. Protected email endpoint
        res_email_unauth = client.post("/api/generate-email", json={"customer_name": "Test", "customer_data": sample_customer, "churn_probability": 0.6})
        assert res_email_unauth.status_code == 401
        res_email_auth = client.post("/api/generate-email", json={"customer_name": "Test", "customer_data": sample_customer, "churn_probability": 0.6}, headers={"Authorization": "Bearer test-bearer-token"})
        assert res_email_auth.status_code == 200
        assert "subject" in res_email_auth.json()
    finally:
        server_module.REQUIRE_AUTH = original_auth_req
        server_module.VALID_AUTH_SECRETS = original_secrets

def test_input_validation(client, sample_customer):
    # Invalid categorical value
    bad_cust = dict(sample_customer, Contract="PermanentNeverEnding")
    res = client.post("/api/predict", json=bad_cust)
    assert res.status_code == 422

    # Negative tenure
    bad_tenure = dict(sample_customer, tenure=-5)
    res_tenure = client.post("/api/predict", json=bad_tenure)
    assert res_tenure.status_code == 422

def test_batch_upload_file_restrictions(client):
    # Non-CSV/JSON extension
    bad_ext = client.post("/api/batch-predict", files={"file": ("malicious.exe", b"binarycontent", "application/octet-stream")})
    assert bad_ext.status_code == 400
    assert "Unsupported file format" in bad_ext.json().get("detail", "")

    # Oversized payload (> 2MB MAX_UPLOAD_BYTES)
    oversized_data = b"tenure,MonthlyCharges\n" + (b"1,50.0\n" * 400000)  # ~2.8MB
    res_oversized = client.post("/api/batch-predict", files={"file": ("big.csv", oversized_data, "text/csv")})
    assert res_oversized.status_code == 413

    # Batch rows exceed 150 limit
    row = "7590-VHVEG,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85\n"
    csv_160_rows = "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges\n" + (row * 160)
    res_row_limit = client.post("/api/batch-predict", files={"file": ("too_many_rows.csv", csv_160_rows.encode("utf-8"), "text/csv")})
    assert res_row_limit.status_code == 400
    assert "exceeds maximum limit of 150 records" in res_row_limit.json().get("detail", "")

def test_rate_limiting(client):
    headers = {"X-Forwarded-For": "192.0.2.111"}
    hit_429 = False
    for _ in range(25):
        res = client.post("/api/batch-predict", files={"file": ("mini.csv", b"tenure,MonthlyCharges\n1,50", "text/csv")}, headers=headers)
        if res.status_code == 429:
            hit_429 = True
            assert "Retry-After" in res.headers
            assert int(res.headers["Retry-After"]) > 0
            break
    assert hit_429, "Rate limiter did not throttle requests beyond threshold"

def test_error_handling_sanitization(client):
    # Corrupt upload
    err_res = client.post("/api/batch-predict", files={"file": ("corrupt.json", b"{not valid json...", "application/json")})
    assert err_res.status_code == 400
    body = str(err_res.json())
    assert "Traceback" not in body
    assert "/Users/" not in body
    assert ".py" not in body
