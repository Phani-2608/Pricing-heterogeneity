"""FastAPI prediction service.

Endpoints
---------
GET  /health                      Liveness / readiness
GET  /model/info                  Model version and metadata
POST /predict-treatment           One customer: predicted ITE + policy action
POST /predict-treatment/batch     N customers in one call
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..config import DEFAULT_CONFIG
from ..models.registry import ModelRegistry
from ..optimization.policy_engine import decide_for_customer

logger = logging.getLogger("pricing_heterogeneity.api")


# --- Schemas -----------------------------------------------------------------

class CustomerFeatures(BaseModel):
    customer_id: str | int = Field(..., description="Unique identifier for the customer")
    age: float
    ltv: float
    purchase_freq: float
    tenure_months: float
    prior_discounts: float
    sessions_30d: float
    cohort_month: int = Field(..., ge=1, le=12)
    category: str
    region: str
    baseline_probability: float = Field(
        0.20, ge=0.0, le=1.0,
        description="Prior conversion probability. Used to compute the break-even ITE.",
    )


class PredictBatchRequest(BaseModel):
    customers: list[CustomerFeatures]


class PredictResponse(BaseModel):
    customer_id: str | int
    predicted_ite: float
    ite_lower: float
    ite_upper: float
    expected_conv_lift: float
    customer_value: float
    intervention_cost: float
    expected_incremental_revenue: float
    action: str
    confidence: str
    reason: str
    model_version: str
    latency_ms: float


# --- App lifecycle -----------------------------------------------------------

_STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry_dir = os.environ.get("REGISTRY_DIR", "outputs/registry")
    registry = ModelRegistry(registry_dir)
    version = os.environ.get("MODEL_VERSION") or registry.latest_version()
    if version is None:
        logger.warning("No trained model found in %s; API will serve without a model.", registry_dir)
        _STATE.update({"loaded": False, "version": None})
    else:
        artifact, meta = registry.load(version)
        _STATE.update({
            "loaded": True,
            "version": version,
            "cate_model": artifact["cate_model"],
            "feature_builder": artifact["feature_builder"],
            "metadata": meta,
        })
        logger.info("Loaded model version %s", version)
    yield
    _STATE.clear()


app = FastAPI(
    title="Pricing Heterogeneity Decision Service",
    version="1.0.0",
    description="Serves per-customer treatment-effect predictions and pricing recommendations.",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Response-Time-Ms"] = f"{(time.perf_counter()-start)*1000:.2f}"
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "internal_server_error"})


# --- Routes ------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _STATE.get("loaded", False)}


@app.get("/model/info")
def model_info():
    if not _STATE.get("loaded"):
        raise HTTPException(status_code=503, detail="No model loaded")
    m = _STATE["metadata"]
    return {
        "model_version": m.model_version,
        "trained_at": m.trained_at,
        "n_train": m.n_train,
        "metrics": m.metrics,
        "config": m.config,
    }


def _score_batch(rows: list[CustomerFeatures]) -> list[PredictResponse]:
    if not _STATE.get("loaded"):
        raise HTTPException(status_code=503, detail="No model loaded")

    df = pd.DataFrame([r.model_dump() for r in rows])
    for c in ["ltv_was_missing", "tenure_months_was_missing", "sessions_30d_was_missing"]:
        df[c] = 0

    fb = _STATE["feature_builder"]
    model = _STATE["cate_model"]
    X = fb.transform(df)
    started = time.perf_counter()
    cate = model.predict(X)
    latency_ms = (time.perf_counter() - started) * 1000

    out: list[PredictResponse] = []
    for i, r in enumerate(rows):
        decision = decide_for_customer(
            customer_id=r.customer_id,
            predicted_ite=float(cate[i]),
            baseline_p=r.baseline_probability,
            ite_se=None,
            customer_value=r.ltv,
            cfg=DEFAULT_CONFIG,
            model_version=_STATE["version"],
        )
        out.append(PredictResponse(latency_ms=latency_ms / max(len(rows), 1), **decision.as_dict()))
    return out


@app.post("/predict-treatment", response_model=PredictResponse)
def predict_treatment(customer: CustomerFeatures):
    return _score_batch([customer])[0]


@app.post("/predict-treatment/batch", response_model=list[PredictResponse])
def predict_treatment_batch(req: PredictBatchRequest):
    if not req.customers:
        raise HTTPException(status_code=400, detail="customers must be non-empty")
    if len(req.customers) > 1000:
        raise HTTPException(status_code=413, detail="max 1000 customers per batch")
    return _score_batch(req.customers)
