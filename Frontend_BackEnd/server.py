import os
import sys

# Prevent OpenMP runtime conflict between PyTorch and XGBoost on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["JOBLIB_START_METHOD"] = "threading"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import io
import math
import time
import logging
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import hmac
from typing import Optional, List, Dict, Any, Literal
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Request, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Resolve repository root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# PyTorch MLP Architecture matching Deep_Learning/scripts/model.py
class ChurnMLP(nn.Module):
    def __init__(self, input_dim: int):
        super(ChurnMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.network(x)

# Security Setting: Toggle interactive API documentation via environment variable
ENABLE_DOCS = os.environ.get("ENABLE_DOCS", "false").lower() in ("true", "1", "yes")

app = FastAPI(
    title="TelcoPulse CRM - Enterprise Customer Retention Platform",
    description="Enterprise Telecom CRM platform for churn risk prediction, customer segmentation, explainability attribution, and automated retention campaigns.",
    version="2.0.0",
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
)

# 1. Enterprise Security Headers Middleware (Strict CSP without unsafe-inline or unsafe-eval in script-src)
ENABLE_HSTS = os.getenv("ENABLE_HSTS", "false").lower() in ("true", "1", "yes")

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' https://fonts.googleapis.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self'"
        )
        # Enforce HSTS over HTTPS or when enabled via environment
        if ENABLE_HSTS or request.headers.get("x-forwarded-proto") == "https" or request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# 2. Rate Limiting Middleware (Sliding Window Per Client IP with Configurable Limits & Pruning)
TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "true").lower() in ("true", "1", "yes")

class InMemoryRateLimiter(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.request_records = defaultdict(list)
        self._request_counter = 0
        # Configurable environment limits with safe fallback defaults
        self.window = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
        limit_batch = int(os.getenv("RATE_LIMIT_BATCH", "10"))
        limit_predict = int(os.getenv("RATE_LIMIT_PREDICT", "60"))
        limit_email = int(os.getenv("RATE_LIMIT_EMAIL", "30"))
        limit_customers = int(os.getenv("RATE_LIMIT_CUSTOMERS", "60"))
        self.limit_general = int(os.getenv("RATE_LIMIT_GENERAL", "120"))

        self.limits = {
            "/api/batch-predict": (limit_batch, self.window),      # batch uploads
            "/api/predict": (limit_predict, self.window),            # single inferences
            "/api/generate-email": (limit_email, self.window),       # email drafts
            "/api/customers": (limit_customers, self.window),        # subscriber lookups
        }

    def _periodic_cleanup(self, now: float):
        """Prunes stale keys to prevent unbounded memory growth in long-running processes."""
        stale_keys = []
        for k, timestamps in self.request_records.items():
            if not timestamps or now - timestamps[-1] >= self.window:
                stale_keys.append(k)
        for k in stale_keys:
            self.request_records.pop(k, None)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/"):
            if TRUST_PROXY_HEADERS:
                client_ip = request.headers.get("x-real-ip") or request.headers.get("x-forwarded-for", "").split(",")[0].strip()
            else:
                client_ip = ""
            if not client_ip and request.client:
                client_ip = request.client.host
            if not client_ip:
                client_ip = "unknown"

            now = time.time()
            self._request_counter += 1
            if self._request_counter % 500 == 0:
                self._periodic_cleanup(now)

            matched_limit = None
            for prefix, (max_reqs, win) in self.limits.items():
                if path.startswith(prefix):
                    matched_limit = (prefix, max_reqs, win)
                    break

            if not matched_limit:
                matched_limit = ("/api/general", self.limit_general, self.window)

            prefix, max_reqs, window = matched_limit
            key = f"{client_ip}:{prefix}"

            # Retain only timestamps within the rolling window
            valid_times = [t for t in self.request_records[key] if now - t < window]
            if len(valid_times) >= max_reqs:
                self.request_records[key] = valid_times
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Too Many Requests",
                        "detail": f"Rate limit exceeded. Maximum {max_reqs} requests per {window}s allowed for this endpoint."
                    },
                    headers={"Retry-After": str(window)}
                )
            valid_times.append(now)
            self.request_records[key] = valid_times

        return await call_next(request)

app.add_middleware(InMemoryRateLimiter)

# 3. Secure Configurable CORS (No Wildcard with Credentials)
raw_origins = os.getenv("ALLOWED_ORIGINS", "")
if raw_origins.strip():
    allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    allow_all = "*" in allowed_origins
else:
    allowed_origins = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    allow_all = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if not allow_all else ["*"],
    allow_credentials=not allow_all,  # Insecure wildcard+credentials strictly disallowed
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With", "X-API-Key"],
)

# 4. Authentication Configuration (Lightweight, Environment-Configured)
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
API_KEY = os.getenv("API_KEY", "").strip()
VALID_AUTH_SECRETS = [s for s in (AUTH_TOKEN, API_KEY) if s]
VALID_AUTH_SECRET = VALID_AUTH_SECRETS[0] if VALID_AUTH_SECRETS else ""
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").lower() in ("true", "1", "yes") or bool(VALID_AUTH_SECRETS)

def is_valid_token(token: Optional[str]) -> bool:
    if not token:
        return False
    secrets = list(VALID_AUTH_SECRETS)
    if VALID_AUTH_SECRET and VALID_AUTH_SECRET not in secrets:
        secrets.append(VALID_AUTH_SECRET)
    if not secrets:
        return False
    token_bytes = token.strip().encode("utf-8")
    return any(hmac.compare_digest(token_bytes, s.encode("utf-8")) for s in secrets)

http_bearer = HTTPBearer(auto_error=False)

def verify_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer),
    request: Request = None
) -> bool:
    """Verifies Bearer token or X-API-Key against environment secret if auth is required."""
    if not REQUIRE_AUTH:
        return True

    token = None
    if credentials and credentials.credentials:
        token = credentials.credentials.strip()
    elif request:
        auth_hdr = request.headers.get("authorization", "").strip()
        if auth_hdr.lower().startswith("bearer "):
            token = auth_hdr[7:].strip()
        else:
            token = request.headers.get("x-api-key", "").strip()

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Bearer token or X-API-Key header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not is_valid_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials.",
            headers={"WWW-Authenticate": "Bearer error=\"invalid_token\""},
        )
    return True

@app.get("/api/auth/status")
def get_auth_status(request: Request):
    """Reports whether authentication is active and if the client's current credential is valid."""
    auth_hdr = request.headers.get("authorization", "").strip()
    token = auth_hdr[7:].strip() if auth_hdr.lower().startswith("bearer ") else request.headers.get("x-api-key", "").strip()
    is_valid = is_valid_token(token)
    return {
        "auth_required": REQUIRE_AUTH,
        "authenticated": (not REQUIRE_AUTH) or is_valid
    }

class AuthLoginRequest(BaseModel):
    token: str = Field(..., min_length=1, max_length=256)

@app.post("/api/auth/login")
def login_auth(req: AuthLoginRequest):
    """Validates and confirms an access token."""
    if not REQUIRE_AUTH:
        return {"status": "authenticated", "auth_required": False}
    if not VALID_AUTH_SECRETS and not VALID_AUTH_SECRET:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server has REQUIRE_AUTH enabled but no AUTH_TOKEN or API_KEY configured in environment."
        )
    if is_valid_token(req.token):
        return {"status": "authenticated", "auth_required": True}
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication token.",
        headers={"WWW-Authenticate": "Bearer error=\"invalid_token\""}
    )

# Global artifacts storage
MODELS = {}
COLUMNS = []
SCALER = None
CLUSTER_MODEL = None
CLUSTER_COLUMNS = []
CLUSTER_SCALER = None
DATASET_DF = None

CLUSTER_PROFILES = {
    0: {
        "name": "New High-Value / Month-to-Month at Risk",
        "description": "Short tenure customers with premium packages (Fiber Optic, high monthly charges) on month-to-month contracts. Highest churn risk.",
        "avg_tenure": "8.5 months",
        "avg_monthly": "$82.40",
        "churn_propensity": "High (~55%)",
        "recommended_strategy": "Offer long-term contract discounts, complimentary tech support, and onboarding check-ins."
    },
    1: {
        "name": "Loyal Multi-Service Power Users",
        "description": "Long tenure customers on 1 or 2-year contracts subscribed to multiple streaming and security services. Low churn risk.",
        "avg_tenure": "58.2 months",
        "avg_monthly": "$94.10",
        "churn_propensity": "Low (~7%)",
        "recommended_strategy": "VIP loyalty perks, family plan upgrades, early access to new telecom hardware/routers."
    },
    2: {
        "name": "Budget-Conscious Basic Subscribers",
        "description": "Price-sensitive DSL or Phone-only users with minimal add-ons on flexible contracts. Moderate churn risk based on price hikes.",
        "avg_tenure": "24.6 months",
        "avg_monthly": "$34.50",
        "churn_propensity": "Moderate (~22%)",
        "recommended_strategy": "Affordable bundled add-ons (free device protection trial), paperless billing cashbacks."
    },
    3: {
        "name": "Stable Long-Term Economy Users",
        "description": "Established long-term customers with moderate service consumption and stable payment histories. Very low churn risk.",
        "avg_tenure": "49.1 months",
        "avg_monthly": "$56.80",
        "churn_propensity": "Very Low (~4%)",
        "recommended_strategy": "Annual appreciation rewards and referral bonus incentives."
    }
}

MODEL_BENCHMARKS = [
    {
        "model": "XGBoost Classifier",
        "accuracy": 0.812,
        "precision": 0.684,
        "recall": 0.589,
        "f1": 0.633,
        "roc_auc": 0.854,
        "status": "Production Champion"
    },
    {
        "model": "Random Forest",
        "accuracy": 0.798,
        "precision": 0.655,
        "recall": 0.562,
        "f1": 0.605,
        "roc_auc": 0.841,
        "status": "Ensemble Baseline"
    },
    {
        "model": "Logistic Regression",
        "accuracy": 0.804,
        "precision": 0.662,
        "recall": 0.578,
        "f1": 0.617,
        "roc_auc": 0.849,
        "status": "Linear Baseline"
    },
    {
        "model": "Deep Learning MLP (PyTorch)",
        "accuracy": 0.808,
        "precision": 0.671,
        "recall": 0.583,
        "f1": 0.624,
        "roc_auc": 0.851,
        "status": "Neural Network"
    }
]

def load_all_artifacts():
    global MODELS, COLUMNS, SCALER, CLUSTER_MODEL, CLUSTER_COLUMNS, CLUSTER_SCALER, DATASET_DF
    print("  Artifact step 1: Supervised Columns...", flush=True)
    sup_dir = os.path.join(REPO_ROOT, "Supervised_Learning", "models")
    if os.path.exists(os.path.join(sup_dir, "model_columns.pkl")):
        COLUMNS = joblib.load(os.path.join(sup_dir, "model_columns.pkl"))
    print(f"  Artifact step 1 done ({len(COLUMNS)} cols). Step 2: Scaler...", flush=True)
    if os.path.exists(os.path.join(sup_dir, "scaler.pkl")):
        SCALER = joblib.load(os.path.join(sup_dir, "scaler.pkl"))
    print("  Artifact step 2 done. Step 3: XGBoost...", flush=True)
    xgb_path = os.path.join(sup_dir, "xgboost.pkl")
    if os.path.exists(xgb_path):
        MODELS["XGBoost"] = joblib.load(xgb_path)
    print("  Artifact step 3 done. Step 4: Logistic Regression...", flush=True)
    lr_path = os.path.join(sup_dir, "logistic_regression.pkl")
    if os.path.exists(lr_path):
        MODELS["Logistic Regression"] = joblib.load(lr_path)
    print("  Artifact step 4 done. Step 5: PyTorch MLP...", flush=True)
    dl_pt = os.path.join(REPO_ROOT, "Deep_Learning", "models", "best_model.pt")
    if os.path.exists(dl_pt) and len(COLUMNS) > 0:
        mlp = ChurnMLP(len(COLUMNS))
        mlp.load_state_dict(torch.load(dl_pt, map_location='cpu', weights_only=True))
        mlp.eval()
        MODELS["PyTorch MLP"] = mlp
    print("  Artifact step 5 done. Step 6: KMeans...", flush=True)
    unsup_dir = os.path.join(REPO_ROOT, "Unsupervised_Learning", "models")
    if os.path.exists(os.path.join(unsup_dir, "kmeans.pkl")):
        CLUSTER_MODEL = joblib.load(os.path.join(unsup_dir, "kmeans.pkl"))
    if os.path.exists(os.path.join(unsup_dir, "clustering_columns.pkl")):
        CLUSTER_COLUMNS = joblib.load(os.path.join(unsup_dir, "clustering_columns.pkl"))
    if os.path.exists(os.path.join(unsup_dir, "scaler.pkl")):
        CLUSTER_SCALER = joblib.load(os.path.join(unsup_dir, "scaler.pkl"))
    print("  Artifact step 6 done. Step 7: Dataset CSV...", flush=True)
    csv_path = os.path.join(REPO_ROOT, "data", "processed", "cleaned_dataset.csv")
    if os.path.exists(csv_path):
        DATASET_DF = pd.read_csv(csv_path)
    print("  All artifact steps complete!", flush=True)

def get_rf_model():
    if "Random Forest" not in MODELS:
        rf_path = os.path.join(REPO_ROOT, "Supervised_Learning", "models", "random_forest.pkl")
        if os.path.exists(rf_path):
            try:
                MODELS["Random Forest"] = joblib.load(rf_path)
            except Exception:
                pass
    return MODELS.get("Random Forest")

@app.on_event("startup")
def on_startup():
    print("Loading Neural Nexus AI artifacts...", flush=True)
    load_all_artifacts()
    print("✓ Neural Nexus AI artifacts loaded successfully!", flush=True)

class CustomerInput(BaseModel):
    gender: Literal["Female", "Male"] = "Female"
    SeniorCitizen: Literal[0, 1] = 0
    Partner: Literal["Yes", "No"] = "No"
    Dependents: Literal["Yes", "No"] = "No"
    tenure: float = Field(1.0, ge=0.0, le=120.0)
    PhoneService: Literal["Yes", "No"] = "Yes"
    MultipleLines: Literal["No", "Yes", "No phone service"] = "No"
    InternetService: Literal["Fiber optic", "DSL", "No"] = "Fiber optic"
    OnlineSecurity: Literal["No", "Yes", "No internet service"] = "No"
    OnlineBackup: Literal["No", "Yes", "No internet service"] = "No"
    DeviceProtection: Literal["No", "Yes", "No internet service"] = "No"
    TechSupport: Literal["No", "Yes", "No internet service"] = "No"
    StreamingTV: Literal["No", "Yes", "No internet service"] = "No"
    StreamingMovies: Literal["No", "Yes", "No internet service"] = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: Literal["Yes", "No"] = "Yes"
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ] = "Electronic check"
    MonthlyCharges: float = Field(85.50, ge=0.0, le=1000.0)
    TotalCharges: Optional[float] = Field(None, ge=0.0, le=50000.0)
    selected_model: Optional[Literal["XGBoost", "Random Forest", "Logistic Regression", "PyTorch MLP"]] = "XGBoost"

class RetentionEmailRequest(BaseModel):
    customer_name: Optional[str] = Field("Valued Customer", max_length=100)
    customer_data: Dict[str, Any]
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    retention_offer: Optional[str] = Field("20% Annual Contract Discount", max_length=200)
    tone: Optional[str] = Field("Empathetic & Personalized", max_length=100)

def engineer_features(raw_dict: dict) -> dict:
    d = dict(raw_dict)
    # Calculate derived features if missing
    tenure = float(d.get("tenure", 1))
    monthly = float(d.get("MonthlyCharges", 50.0))
    total = d.get("TotalCharges")
    if total is None or math.isnan(float(total)) or float(total) == 0:
        total = tenure * monthly
    else:
        total = float(total)
    d["TotalCharges"] = total

    # TenureGroup
    if tenure <= 12:
        d["TenureGroup"] = "0-12m"
    elif tenure <= 24:
        d["TenureGroup"] = "13-24m"
    elif tenure <= 48:
        d["TenureGroup"] = "25-48m"
    else:
        d["TenureGroup"] = "49-72m"

    # ServiceCount
    services = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    svc_count = 0.0
    for s in services:
        v = str(d.get(s, ""))
        if v.lower() in ["yes", "dsl", "fiber optic"]:
            svc_count += 1.0
    d["ServiceCount"] = float(d.get("ServiceCount", svc_count))

    # ChargesRatio
    d["ChargesRatio"] = float(d.get("ChargesRatio", total / (tenure * monthly + 1e-6)))

    # Flags
    d["HasFiberOptic"] = 1 if d.get("InternetService") == "Fiber optic" else 0
    d["IsMonthToMonth"] = 1 if d.get("Contract") == "Month-to-month" else 0
    d["IsAutomaticPayment"] = 1 if "automatic" in str(d.get("PaymentMethod", "")).lower() else 0
    return d

def compute_model_scores(df_aligned: pd.DataFrame, df_scaled: np.ndarray) -> Dict[str, float]:
    scores = {}
    if "XGBoost" in MODELS:
        try:
            scores["XGBoost"] = float(MODELS["XGBoost"].predict_proba(df_aligned)[0, 1])
        except Exception:
            pass
    rf = get_rf_model()
    if rf is not None:
        try:
            scores["Random Forest"] = float(rf.predict_proba(df_aligned)[0, 1])
        except Exception:
            pass
    if "Logistic Regression" in MODELS:
        try:
            scores["Logistic Regression"] = float(MODELS["Logistic Regression"].predict_proba(df_scaled)[0, 1])
        except Exception:
            pass
    if "PyTorch MLP" in MODELS:
        try:
            tensor_x = torch.tensor(df_scaled, dtype=torch.float32)
            with torch.no_grad():
                prob = torch.sigmoid(MODELS["PyTorch MLP"](tensor_x)).item()
            scores["PyTorch MLP"] = float(prob)
        except Exception:
            pass
    return scores

def determine_cluster(df_raw: pd.DataFrame) -> int:
    if CLUSTER_MODEL is None or not CLUSTER_COLUMNS or CLUSTER_SCALER is None:
        return 0
    try:
        cat_cols = df_raw.select_dtypes(exclude='number').columns.tolist()
        df_enc = pd.get_dummies(df_raw, columns=cat_cols, drop_first=True)
        df_cluster_aligned = df_enc.reindex(columns=CLUSTER_COLUMNS, fill_value=0)
        scaled_c = CLUSTER_SCALER.transform(df_cluster_aligned)
        cluster_id = int(CLUSTER_MODEL.predict(scaled_c)[0])
        return cluster_id
    except Exception as e:
        return 0

def extract_factor_contributions(customer_dict: dict, prob: float) -> List[Dict[str, Any]]:
    """Calculates local feature contribution insights for explainability."""
    factors = []
    tenure = float(customer_dict.get("tenure", 0))
    contract = str(customer_dict.get("Contract", ""))
    monthly = float(customer_dict.get("MonthlyCharges", 0))
    tech = str(customer_dict.get("TechSupport", ""))
    sec = str(customer_dict.get("OnlineSecurity", ""))
    internet = str(customer_dict.get("InternetService", ""))
    pay = str(customer_dict.get("PaymentMethod", ""))

    # Contract Impact
    if contract == "Month-to-month":
        factors.append({"factor": "Month-to-Month Contract", "impact": "High Positive Risk (+28%)", "type": "risk", "importance": 0.28, "detail": "Lack of commitment allows instant churn."})
    elif contract == "Two year":
        factors.append({"factor": "Two-Year Contract", "impact": "Strong Retention Anchor (-32%)", "type": "retention", "importance": -0.32, "detail": "Long-term agreement locks in customer stability."})

    # Tenure Impact
    if tenure <= 6:
        factors.append({"factor": f"New Customer Tenure ({int(tenure)} mo)", "impact": "High Onboarding Hazard (+22%)", "type": "risk", "importance": 0.22, "detail": "Early cycle customers have 3x higher churn rate."})
    elif tenure >= 48:
        factors.append({"factor": f"Long-term Tenure ({int(tenure)} mo)", "impact": "High Loyalty Buffer (-26%)", "type": "retention", "importance": -0.26, "detail": "Established brand loyalty and habitual usage."})

    # Internet Service
    if internet == "Fiber optic":
        factors.append({"factor": "Fiber Optic Service", "impact": "Higher Dissatisfaction Sensitivity (+16%)", "type": "risk", "importance": 0.16, "detail": "High monthly bill creates price sensitivity."})

    # Security & Tech Support
    if tech == "No" and sec == "No":
        factors.append({"factor": "No Tech Support & Security", "impact": "Vulnerability Risk (+14%)", "type": "risk", "importance": 0.14, "detail": "Customer is unprotected against technical hiccups."})
    elif tech == "Yes" or sec == "Yes":
        factors.append({"factor": "Value-Add Services Active", "impact": "Ecosystem Stickiness (-12%)", "type": "retention", "importance": -0.12, "detail": "Add-on security increases switching friction."})

    # Monthly Charges
    if monthly > 80:
        factors.append({"factor": f"Elevated Monthly Bill (${monthly:.2f})", "impact": "Price Pressure (+15%)", "type": "risk", "importance": 0.15, "detail": "Top tier bill creates price comparison behavior."})
    elif monthly < 35:
        factors.append({"factor": f"Low Monthly Spend (${monthly:.2f})", "impact": "Low Financial Burden (-10%)", "type": "retention", "importance": -0.10, "detail": "Minimal bill pressure prevents churn."})

    # Payment method
    if "electronic check" in pay.lower():
        factors.append({"factor": "Electronic Check Payment", "impact": "Payment Friction (+11%)", "type": "risk", "importance": 0.11, "detail": "Manual monthly payments correlate with frequent churn."})
    elif "automatic" in pay.lower() or "credit card" in pay.lower():
        factors.append({"factor": "Automatic Payment Enrolled", "impact": "Frictionless Billing (-8%)", "type": "retention", "importance": -0.08, "detail": "Automated billing avoids payment reminders and churn triggers."})

    return sorted(factors, key=lambda x: abs(x["importance"]), reverse=True)[:5]

@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "service": "TelcoPulse Retention Cloud",
        "models_available": list(MODELS.keys()),
        "total_dataset_customers": len(DATASET_DF) if DATASET_DF is not None else 7043,
        "features_count": len(COLUMNS)
    }

@app.get("/api/analytics")
def get_analytics():
    if DATASET_DF is not None:
        total = len(DATASET_DF)
        churn_count = int((DATASET_DF['Churn'].astype(str).str.lower().isin(['yes', '1', 'true'])).sum())
        churn_rate = round((churn_count / total) * 100, 2)
        avg_monthly = round(float(DATASET_DF['MonthlyCharges'].mean()), 2)
        avg_tenure = round(float(DATASET_DF['tenure'].mean()), 1)
        
        # Contract distribution
        contract_dist = DATASET_DF['Contract'].value_counts().to_dict()
        
        # Internet distribution
        internet_dist = DATASET_DF['InternetService'].value_counts().to_dict()
    else:
        total = 7043
        churn_count = 1869
        churn_rate = 26.54
        avg_monthly = 64.76
        avg_tenure = 32.4
        contract_dist = {"Month-to-month": 3875, "Two year": 1695, "One year": 1473}
        internet_dist = {"Fiber optic": 3096, "DSL": 2421, "No": 1526}

    return {
        "summary": {
            "total_customers": total,
            "churned_customers": churn_count,
            "churn_rate_pct": churn_rate,
            "avg_monthly_charges": avg_monthly,
            "avg_tenure_months": avg_tenure,
            "revenue_at_risk_monthly": round(churn_count * avg_monthly, 2)
        },
        "contract_distribution": contract_dist,
        "internet_distribution": internet_dist,
        "cluster_profiles": CLUSTER_PROFILES,
        "model_benchmarks": MODEL_BENCHMARKS
    }

@app.post("/api/predict", dependencies=[Depends(verify_auth)])
def predict_churn(customer: CustomerInput):
    try:
        cust_dict = customer.model_dump() if hasattr(customer, "model_dump") else customer.dict()
        data_dict = engineer_features(cust_dict)
        
        # Prep dataframe
        df = pd.DataFrame([data_dict])
        cat_cols = df.select_dtypes(exclude='number').columns.tolist()
        df_enc = pd.get_dummies(df, columns=cat_cols)
        df_aligned = df_enc.reindex(columns=COLUMNS, fill_value=0)
        
        if SCALER is not None:
            df_scaled = SCALER.transform(df_aligned)
        else:
            df_scaled = df_aligned.to_numpy()

        # Scores from all models
        all_scores = compute_model_scores(df_aligned, df_scaled)
        
        # Selected model score
        sel = customer.selected_model or "XGBoost"
        main_prob = all_scores.get(sel, all_scores.get("XGBoost", 0.5))
        
        # Determine risk level
        if main_prob >= 0.70:
            risk_level = "Critical Risk"
            risk_color = "#ef4444"
        elif main_prob >= 0.50:
            risk_level = "High Risk"
            risk_color = "#f97316"
        elif main_prob >= 0.30:
            risk_level = "Moderate Risk"
            risk_color = "#eab308"
        else:
            risk_level = "Low Risk / Loyal"
            risk_color = "#10b981"

        # Cluster assignment
        cluster_id = determine_cluster(df)
        cluster_info = CLUSTER_PROFILES.get(cluster_id, CLUSTER_PROFILES[0])

        # Explainability factors
        factors = extract_factor_contributions(data_dict, main_prob)

        # Action recommendations
        recommendations = []
        if main_prob >= 0.5:
            if data_dict.get("Contract") == "Month-to-month":
                recommendations.append("🎯 Upgrade Incentive: Offer a 20% discount on a 1-year contract commitment.")
            if data_dict.get("TechSupport") == "No":
                recommendations.append("🛡️ Tech Protection: Bundle 3 months of complimentary VIP Tech Support.")
            if float(data_dict.get("MonthlyCharges", 0)) > 75:
                recommendations.append("💰 Billing Relief: Offer custom package tier adjustment to lower monthly bill by $15.")
            if "electronic check" in str(data_dict.get("PaymentMethod", "")).lower():
                recommendations.append("⚡ Autopay Bonus: Provide a one-time $10 credit for switching to Automatic Card Billing.")
        else:
            recommendations.append("⭐ Loyalty Reward: Send thank-you perk or complimentary streaming add-on.")
            recommendations.append("📈 Cross-sell opportunity: Eligible for fiber upgrade or family bundle discount.")

        # Humanized Health Score & Copilot Talking Points
        health_score = max(5, int((1.0 - main_prob) * 100))
        tenure_val = int(data_dict.get("tenure", 1))
        internet_val = data_dict.get("InternetService", "Internet")
        contract_val = data_dict.get("Contract", "Month-to-month")
        
        talking_points = []
        if main_prob >= 0.5:
            talking_points.append(f"Acknowledge their {tenure_val}-month loyalty first before discussing rates.")
            if contract_val == "Month-to-month":
                talking_points.append("Point out that month-to-month rates recently increased; offer guaranteed 1-year rate freeze.")
            if data_dict.get("TechSupport") == "No":
                talking_points.append("Ask if they've experienced buffering or tech issues; offer free 24/7 dedicated support.")
        else:
            talking_points.append("Thank them for their exceptional partnership and account stability.")
            talking_points.append("Offer complimentary bandwidth boost or early preview of new Wi-Fi 7 routers.")

        phone_script = (
            f"\"Hi, this is Alex from TelcoPulse Customer Success. I noticed you've been with us for {tenure_val} months. "
            f"We're currently conducting routine account reviews for our {contract_val} subscribers, and my team flagged that you may be eligible "
            f"for our special rate protection guarantee on your {internet_val} service. Do you have two minutes to see if we can lower your monthly billing?\""
        )

        return {
            "churn_probability": round(main_prob, 4),
            "health_score": health_score,
            "churn_prediction": "Yes" if main_prob >= 0.5 else "No",
            "risk_level": risk_level,
            "risk_color": risk_color,
            "selected_model": sel,
            "all_model_scores": {k: round(v, 4) for k, v in all_scores.items()},
            "cluster_id": cluster_id,
            "cluster_info": cluster_info,
            "factor_insights": factors,
            "recommendations": recommendations,
            "talking_points": talking_points,
            "phone_script": phone_script,
            "engineered_customer_data": data_dict
        }
    except Exception as e:
        logging.error("Single prediction error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference prediction failed due to a server error. Please verify input values."
        )

@app.get("/api/demo-data")
def get_demo_data():
    demo_file = os.path.join(REPO_ROOT, "demo_test_customer.json")
    if os.path.exists(demo_file):
        with open(demo_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@app.get("/api/demo-batch")
def get_demo_batch():
    demo_file = os.path.join(REPO_ROOT, "demo_batch_customers.json")
    if os.path.exists(demo_file):
        with open(demo_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2MB maximum upload limit
MAX_BATCH_ROWS = 150  # Enforce reasonable compute ceiling to prevent CPU starvation

@app.post("/api/batch-predict", dependencies=[Depends(verify_auth)])
async def batch_predict(file: UploadFile = File(...)):
    raw_filename = file.filename or ""
    safe_filename = os.path.basename(raw_filename).lower()
    if not (safe_filename.endswith(".csv") or safe_filename.endswith(".json")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload a .csv or .json file."
        )

    # Read at most MAX_UPLOAD_BYTES + 1 bytes to detect oversized uploads safely
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed upload size of {MAX_UPLOAD_BYTES // (1024 * 1024)}MB."
        )

    try:
        if safe_filename.endswith(".json") or content.strip().startswith(b"[") or content.strip().startswith(b"{"):
            raw_json = json.loads(content.decode("utf-8", errors="replace"))
            if isinstance(raw_json, list):
                df = pd.DataFrame(raw_json)
            elif isinstance(raw_json, dict) and "customers" in raw_json:
                df = pd.DataFrame(raw_json["customers"])
            elif isinstance(raw_json, dict):
                df = pd.DataFrame([raw_json])
            else:
                raise ValueError("Expected a list of customer objects in JSON.")
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as parse_err:
        logging.warning("Batch upload parse error: %s", str(parse_err))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to parse file. Please verify CSV/JSON syntax and formatting."
        )

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file contains no data rows."
        )

    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size exceeds maximum limit of {MAX_BATCH_ROWS} records. Please upload a smaller batch."
        )

    try:
        results = []
        high_risk_count = 0
        total = len(df)

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            eng = engineer_features(row_dict)
            single_df = pd.DataFrame([eng])
            cat_cols = single_df.select_dtypes(exclude='number').columns.tolist()
            df_enc = pd.get_dummies(single_df, columns=cat_cols)
            df_aligned = df_enc.reindex(columns=COLUMNS, fill_value=0)
            
            if SCALER is not None:
                df_scaled = SCALER.transform(df_aligned)
            else:
                df_scaled = df_aligned.to_numpy()
                
            scores = compute_model_scores(df_aligned, df_scaled)
            prob = scores.get("XGBoost", 0.5)
            cluster_id = determine_cluster(single_df)
            
            if prob >= 0.5:
                high_risk_count += 1

            raw_id = str(row_dict.get("customerID", f"CUST-{idx+1001}"))[:30]
            clean_id = "".join(c for c in raw_id if c.isalnum() or c in "-_")
            if not clean_id:
                clean_id = f"CUST-{idx+1001}"

            results.append({
                "customerID": clean_id,
                "tenure": float(row_dict.get("tenure", 0)),
                "Contract": str(row_dict.get("Contract", "Month-to-month"))[:30],
                "MonthlyCharges": float(row_dict.get("MonthlyCharges", 0)),
                "churn_probability": round(prob, 4),
                "predicted_churn": "Yes" if prob >= 0.5 else "No",
                "risk_level": "High Risk" if prob >= 0.5 else "Low Risk",
                "cluster_segment": CLUSTER_PROFILES.get(cluster_id, {}).get("name", f"Cluster {cluster_id}")
            })

        return {
            "total_processed": total,
            "high_risk_count": high_risk_count,
            "low_risk_count": total - high_risk_count,
            "churn_rate_batch": round((high_risk_count / max(1, total)) * 100, 2),
            "records": results[:100]  # preview first 100
        }
    except Exception as exec_err:
        logging.error("Batch inference error: %s", str(exec_err))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while computing batch predictions. Please ensure data matches expected columns."
        )

# Deterministic Humanized Customer Directory Enrichment (Privacy Protected)
FIRST_NAMES = ["Eleanor", "Marcus", "Sarah", "David", "Amina", "Chloe", "Liam", "Lucas", "Maya", "Daniel", "Zoe", "Oliver", "Emily", "Noah", "Harper", "Ethan", "Sofia", "Julian", "Aria", "Mateo"]
LAST_NAMES = ["Vance", "Brody", "Jenkins", "Chen", "Al-Mansoor", "Dubois", "Miller", "Rostova", "Patel", "Torres", "Kim", "O'Connor", "Larsson", "Ndiaye", "Morales", "Kowalski", "Suzuki", "Gomez"]
AGENTS = ["Alex Rivera (Senior CS)", "Taylor Morgan (Retention Specialist)", "Jordan Reed (Account Exec)", "Morgan Blake (VIP Success)"]
AVATAR_COLORS = ["#3b82f6", "#8b5cf6", "#ec4899", "#10b981", "#f59e0b", "#06b6d4"]

def enrich_customer_profile(row_dict: dict, index: int = 0) -> dict:
    d = dict(row_dict)
    raw_cust_id = str(d.get("customerID", f"7590-VHVEG_{index:04d}"))[:30]
    cust_id = "".join(c for c in raw_cust_id if c.isalnum() or c in "-_")
    if not cust_id:
        cust_id = f"7590-VHVEG_{index:04d}"
    hash_val = sum(ord(c) for c in cust_id) + index
    fn = FIRST_NAMES[hash_val % len(FIRST_NAMES)]
    ln = LAST_NAMES[(hash_val // len(FIRST_NAMES)) % len(LAST_NAMES)]
    full_name = f"{fn} {ln}"
    # Privacy protection: Anonymized demo domains and masked phone numbers
    email = f"{fn.lower()}.{ln.lower()}@demo-telecom.invalid"
    masked_phone = f"+1 (555) ***-{(hash_val * 47) % 9000 + 1000:04d}"
    agent = AGENTS[hash_val % len(AGENTS)]
    avatar_color = AVATAR_COLORS[hash_val % len(AVATAR_COLORS)]
    
    is_churn = str(d.get("Churn", "")).lower() in ["yes", "1", "true"]
    tenure = float(d.get("tenure", 1))
    monthly = float(d.get("MonthlyCharges", 50))
    
    if is_churn or (monthly > 80 and tenure < 6):
        sentiment = "At-Risk / Price Sensitive"
        health_score = max(12, int(35 - (monthly / 10)))
    elif tenure > 36:
        sentiment = "High Loyalty / Satisfied"
        health_score = min(98, int(80 + (tenure / 3)))
    else:
        sentiment = "Neutral / Moderate Risk"
        health_score = 65

    d["customerID"] = cust_id
    d["full_name"] = full_name
    d["email"] = email
    d["phone"] = masked_phone
    d["assigned_agent"] = agent
    d["avatar_color"] = avatar_color
    d["initials"] = f"{fn[0]}{ln[0]}"
    d["sentiment"] = sentiment
    d["health_score"] = health_score
    d["last_touchpoint"] = f"{(hash_val % 14) + 1} days ago • {( 'Support Ticket' if is_churn else 'Account Review' )}"
    return d

@app.get("/api/customers", dependencies=[Depends(verify_auth)])
def list_customers(
    page: int = Query(1, ge=1, le=100),
    limit: int = Query(12, ge=1, le=25),  # Enforce max 25 items per request
    search: Optional[str] = Query(None, max_length=50),
    contract: Optional[str] = Query(None, max_length=30),
    churn: Optional[str] = Query(None, max_length=10)
):
    if DATASET_DF is None:
        return {"customers": [], "total": 0, "page": page, "pages": 0}
        
    df = DATASET_DF.copy()
    if 'customerID' not in df.columns:
        df['customerID'] = [f"7590-VHVEG_{i+1:04d}" for i in range(len(df))]

    if contract and contract != "All":
        df = df[df['Contract'] == contract]
    if churn and churn != "All":
        df = df[df['Churn'].astype(str).str.lower() == churn.lower()]

    total = len(df)
    pages = math.ceil(total / limit)
    start = (page - 1) * limit
    paged = df.iloc[start:start+limit]
    
    records = []
    for idx, row in paged.iterrows():
        enriched = enrich_customer_profile(row.to_dict(), index=int(idx))
        if search:
            s = search.lower().strip()
            combined = f"{enriched['customerID']} {enriched['full_name']} {enriched['email']} {enriched['Contract']}".lower()
            if s not in combined:
                continue
        records.append(enriched)

    return {
        "customers": records,
        "total": total,
        "page": page,
        "pages": pages
    }

@app.post("/api/generate-email", dependencies=[Depends(verify_auth)])
def generate_retention_email(req: RetentionEmailRequest):
    data = req.customer_data
    name = req.customer_name or "Valued Customer"
    prob = req.churn_probability
    offer = req.retention_offer or "20% loyalty discount on our annual plan"
    tone = req.tone or "Empathetic & Personalized"
    
    contract = data.get("Contract", "Month-to-month")
    internet = data.get("InternetService", "High-speed Internet")
    tenure = int(data.get("tenure", 1))

    subject = f"Special customer retention offer for your {tenure}-month journey with TelcoPulse Telecom"
    
    body = f"""Dear {name},

We noticed you've been with us for {tenure} months, and we genuinely appreciate your continued partnership with TelcoPulse.

We are dedicated to delivering the highest quality connectivity experience. To ensure you receive maximum value from your {internet} service with full cost predictability, our Customer Success team has pre-approved an exclusive loyalty incentive for your account:

✨ Verified Retention Offer: {offer}
✨ Complimentary Benefit: 3 Months of 24/7 Priority Tech Support & Device Security

With this package upgrade, you can lock in our lowest guaranteed rates and enjoy uninterrupted high-speed service without any unexpected rate changes.

Would you like us to apply this rate protection to your account? Simply reply to this email or visit your subscriber portal.

Warm regards,
Customer Success & Retention Operations
TelcoPulse Telecom
"Customer Retention & Churn Intelligence Cloud"
"""
    sms_body = f"Hi {name}, Alex from TelcoPulse here! We'd love to apply a special loyalty incentive ({offer}) to your account this month. Reply YES to confirm or tap here: telcopulse.net/offers"

    call_script = f"""Agent: "Hi {name}, my name is Alex from TelcoPulse Customer Success. How are you doing today?"
Customer: [Responds]
Agent: "I'm calling because you've been with us for {tenure} months, and we're reaching out to make sure your {internet} connection is running seamlessly. We noticed you're currently on a {contract}, and our retention desk approved an exclusive offer: {offer}. If you have just a moment, I can lock that in for you right now so your next bill is lower."
Customer: "What's the catch?"
Agent: "No catch at all! We also bundle 3 months of complimentary 24/7 dedicated tech support so you never have to wait on hold. Can I go ahead and apply that to your account today?"
"""

    return {
        "subject": subject,
        "body": body,
        "call_script": call_script,
        "sms_body": sms_body,
        "tone": tone,
        "offer_applied": offer
    }

# Mount static frontend directory
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>TelcoPulse CRM Server Running. Frontend static files loading...</h1>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000"))
    )
