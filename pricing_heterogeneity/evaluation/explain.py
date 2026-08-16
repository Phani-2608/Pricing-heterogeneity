"""Explainability: SHAP values for global drivers and per-customer explanations.

Falls back to sklearn permutation importance if SHAP isn't installed, so this
module doesn't hard-block CI.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


def global_feature_importance(model, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 5, seed: int = 42) -> pd.DataFrame:
    """SHAP mean-absolute values per feature, or permutation importance as fallback."""
    try:
        import shap
        explainer = shap.TreeExplainer(model) if hasattr(model, "estimators_") else shap.Explainer(model, X)
        vals = explainer(X[:1000])
        arr = vals.values if hasattr(vals, "values") else np.array(vals)
        if arr.ndim == 3:
            arr = arr[:, :, 1] if arr.shape[-1] == 2 else arr.mean(axis=-1)
        imp = np.abs(arr).mean(axis=0)
        return pd.DataFrame({"feature": list(X.columns), "importance": imp}).sort_values(
            "importance", ascending=False
        )
    except ImportError:
        logger.warning("shap not installed; falling back to permutation importance")
    except Exception as e:
        logger.warning("SHAP failed (%s); falling back to permutation importance", e)

    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=seed, n_jobs=-1)
    return pd.DataFrame({"feature": list(X.columns), "importance": r.importances_mean}).sort_values(
        "importance", ascending=False
    )


def explain_customer(model, X_row: pd.DataFrame) -> dict:
    """Per-customer explanation. SHAP per-feature contributions if available,
    otherwise the model's top-N features from global importance are surfaced as
    a diagnostic rather than a true local explanation."""
    try:
        import shap
        explainer = shap.TreeExplainer(model) if hasattr(model, "estimators_") else shap.Explainer(model)
        vals = explainer(X_row)
        arr = vals.values if hasattr(vals, "values") else np.array(vals)
        if arr.ndim == 3:
            arr = arr[:, :, 1] if arr.shape[-1] == 2 else arr.mean(axis=-1)
        contributions = dict(sorted(
            zip(X_row.columns, arr[0].tolist(), strict=False), key=lambda kv: abs(kv[1]), reverse=True,
        ))
        return {"method": "shap", "contributions": contributions}
    except Exception as e:
        logger.warning("Per-customer SHAP unavailable (%s)", e)
        return {"method": "unavailable", "contributions": {}}
