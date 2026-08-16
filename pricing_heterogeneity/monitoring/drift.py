"""Data drift, prediction drift, and treatment-effect drift monitoring.

Compares a live batch against a baseline reference using Population Stability
Index (PSI) for feature/prediction drift, Kolmogorov–Smirnov for continuous
CATE distributions, and simple lift-vs-reference for realized policy value.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index. > 0.25 typically means significant drift."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = np.quantile(ref, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, edges)
    cur_counts, _ = np.histogram(cur, edges)
    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame, threshold: float = 0.25) -> pd.DataFrame:
    common = [c for c in reference.columns if c in current.columns and pd.api.types.is_numeric_dtype(reference[c])]
    rows = []
    for c in common:
        p = psi(reference[c].values, current[c].values)
        rows.append({"feature": c, "psi": p, "drift_flag": p > threshold})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def prediction_drift(reference_pred: np.ndarray, current_pred: np.ndarray) -> dict:
    psi_val = psi(reference_pred, current_pred)
    ks_stat, ks_p = stats.ks_2samp(reference_pred, current_pred)
    return {
        "psi": psi_val,
        "ks_stat": float(ks_stat),
        "ks_p_value": float(ks_p),
        "drift_flag": bool(psi_val > 0.25 or ks_p < 0.01),
    }


def treatment_effect_drift(reference_cate: np.ndarray, current_cate: np.ndarray) -> dict:
    """Cohort-over-cohort ITE distribution drift with a plain-English label."""
    d = prediction_drift(reference_cate, current_cate)
    d["reference_mean"] = float(np.mean(reference_cate))
    d["current_mean"] = float(np.mean(current_cate))
    d["mean_shift"] = d["current_mean"] - d["reference_mean"]
    return d


def calibration_drift(reference_gates: pd.DataFrame, current_gates: pd.DataFrame) -> dict:
    """Compare current vs reference GATES calibration slope."""
    ref_slope, *_ = stats.linregress(reference_gates["predicted_cate"], reference_gates["realized_dr_effect"])
    cur_slope, *_ = stats.linregress(current_gates["predicted_cate"], current_gates["realized_dr_effect"])
    return {
        "reference_slope": float(ref_slope),
        "current_slope": float(cur_slope),
        "slope_delta": float(cur_slope - ref_slope),
        "calibration_degraded": bool(abs(cur_slope - 1) > abs(ref_slope - 1) + 0.15),
    }


@dataclass
class MonitorReport:
    feature_drift: pd.DataFrame
    prediction_drift: dict
    treatment_effect_drift: dict
    calibration_drift: dict | None

    def summary(self) -> dict:
        return {
            "n_features_drifting": int(self.feature_drift["drift_flag"].sum()),
            "prediction_drift_flag": self.prediction_drift["drift_flag"],
            "treatment_effect_mean_shift": self.treatment_effect_drift["mean_shift"],
            "calibration_degraded": self.calibration_drift["calibration_degraded"] if self.calibration_drift else None,
        }


def run_monitor(
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    reference_pred: np.ndarray,
    current_pred: np.ndarray,
    reference_cate: np.ndarray,
    current_cate: np.ndarray,
    reference_gates: pd.DataFrame | None = None,
    current_gates: pd.DataFrame | None = None,
) -> MonitorReport:
    fd = feature_drift(reference_features, current_features)
    pd_ = prediction_drift(reference_pred, current_pred)
    td = treatment_effect_drift(reference_cate, current_cate)
    cd = calibration_drift(reference_gates, current_gates) if reference_gates is not None and current_gates is not None else None
    return MonitorReport(fd, pd_, td, cd)
