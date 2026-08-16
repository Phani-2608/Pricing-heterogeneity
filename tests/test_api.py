"""API integration test: train a small model, load into the app, hit each endpoint."""
import os

import pytest


@pytest.fixture(scope="module")
def trained_registry_dir(tmp_path_factory):
    from pricing_heterogeneity.config import Config
    from pricing_heterogeneity.pipeline import run_pipeline

    out = tmp_path_factory.mktemp("outputs")
    cfg = Config(n_customers=1_500, n_folds=3, out_dir=str(out), model_version="api-test-1.0.0")
    run_pipeline(cfg, save=True)
    return str(out / "registry")


@pytest.fixture()
def client(trained_registry_dir):
    os.environ["REGISTRY_DIR"] = trained_registry_dir
    os.environ["MODEL_VERSION"] = "api-test-1.0.0"
    from fastapi.testclient import TestClient

    from pricing_heterogeneity.api.service import app
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info_endpoint(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "api-test-1.0.0"


def test_predict_single(client):
    payload = {
        "customer_id": "C-1", "age": 34, "ltv": 320, "purchase_freq": 5,
        "tenure_months": 18, "prior_discounts": 2, "sessions_30d": 8,
        "cohort_month": 6, "category": "Electronics", "region": "North",
        "baseline_probability": 0.2,
    }
    r = client.post("/predict-treatment", json=payload)
    assert r.status_code == 200
    body = r.json()
    for key in ("predicted_ite", "action", "confidence", "reason",
                "model_version", "expected_incremental_revenue"):
        assert key in body
    assert body["action"] in ("OFFER_DISCOUNT", "NO_ACTION")


def test_predict_batch_rejects_empty(client):
    r = client.post("/predict-treatment/batch", json={"customers": []})
    assert r.status_code == 400


def test_predict_batch(client):
    payload = {"customers": [
        {"customer_id": "A", "age": 25, "ltv": 100, "purchase_freq": 3,
         "tenure_months": 6, "prior_discounts": 3, "sessions_30d": 5,
         "cohort_month": 1, "category": "Apparel", "region": "South", "baseline_probability": 0.15},
        {"customer_id": "B", "age": 60, "ltv": 900, "purchase_freq": 10,
         "tenure_months": 60, "prior_discounts": 0, "sessions_30d": 12,
         "cohort_month": 8, "category": "Home", "region": "West", "baseline_probability": 0.35},
    ]}
    r = client.post("/predict-treatment/batch", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
