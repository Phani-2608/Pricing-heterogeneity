"""Predictive-model baselines for conversion propensity.

Compares Logistic Regression, Random Forest, XGBoost, and LightGBM head-to-head
on classification metrics + calibration. These are conversion-propensity models
(P(Y=1 | X)), NOT causal models — they are the natural baseline that gets
compared against the causal, treatment-effect-aware approach downstream.

XGBoost and LightGBM are optional: the module degrades gracefully if either
isn't installed so CI can still run in a minimal environment.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import Config

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    name: str
    metrics: dict[str, float]
    calibration: pd.DataFrame
    y_pred_proba: np.ndarray
    model: Any


def _try_xgboost(cfg: Config):
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=cfg.random_state, n_jobs=-1, verbosity=0,
            use_label_encoder=False, eval_metric="logloss",
        )
    except ImportError:
        logger.warning("xgboost not installed; skipping")
        return None


def _try_lightgbm(cfg: Config):
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=200, max_depth=-1, learning_rate=0.1,
            random_state=cfg.random_state, n_jobs=-1, verbose=-1,
        )
    except ImportError:
        logger.warning("lightgbm not installed; skipping")
        return None


def build_model_zoo(cfg: Config) -> dict[str, Any]:
    zoo = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=cfg.random_state, n_jobs=-1)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=25, random_state=cfg.random_state, n_jobs=-1,
        ),
    }
    if (xgb := _try_xgboost(cfg)) is not None:
        zoo["XGBoost"] = xgb
    if (lgb := _try_lightgbm(cfg)) is not None:
        zoo["LightGBM"] = lgb
    return zoo


def evaluate_classification(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def calibration_frame(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac_pos})


def compare_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    cfg: Config,
) -> tuple[pd.DataFrame, dict[str, ModelResult]]:
    """Fit every model in the zoo, return a metrics table plus per-model artifacts."""
    zoo = build_model_zoo(cfg)
    results: dict[str, ModelResult] = {}
    rows = []
    for name, model in zoo.items():
        logger.info("Training %s", name)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_classification(y_test, proba)
        cal = calibration_frame(y_test, proba)
        results[name] = ModelResult(name, metrics, cal, proba, model)
        rows.append({"model": name, **metrics})
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False), results
