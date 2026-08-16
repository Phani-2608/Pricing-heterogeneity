"""Data quality: missingness, outliers, temporal structure, and validation
checks for duplicates, schema, treatment imbalance, and leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EXPECTED_SCHEMA = {
    "age": "float", "ltv": "float", "purchase_freq": "float",
    "tenure_months": "float", "prior_discounts": "float", "sessions_30d": "float",
    "category": "object", "region": "object", "cohort_month": "int",
    "treated": "int", "converted": "int",
}


def inject_missingness(
    df: pd.DataFrame, cols: list[str], rate: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict]:
    """MCAR-inject missing values, then impute with a kept missing-indicator."""
    df = df.copy()
    n = len(df)
    for col in cols:
        idx = rng.choice(n, size=int(rate * n), replace=False)
        df.loc[idx, col] = np.nan
    missing_before = df[cols].isna().sum().to_dict()

    for col in cols:
        df[f"{col}_was_missing"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(df[col].median())

    return df, {"missing_rate_injected": rate, "missing_counts_before": missing_before}


def _iqr_bounds(s: pd.Series, k: float = 3.0) -> tuple[float, float]:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def inject_and_winsorize_outliers(
    df: pd.DataFrame, col: str, rng: np.random.Generator, share: float = 0.015, k: float = 3.0
) -> tuple[pd.DataFrame, dict]:
    """Inject extreme values, then winsorize with an IQR rule."""
    df = df.copy()
    n = len(df)
    outlier_idx = rng.choice(n, size=int(share * n), replace=False)
    df.loc[outlier_idx, col] *= rng.uniform(8, 25, len(outlier_idx))

    lo, hi = _iqr_bounds(df[col], k=k)
    n_flagged = int(((df[col] < lo) | (df[col] > hi)).sum())
    df[col] = df[col].clip(lo, hi)

    return df, {
        "outliers_injected": len(outlier_idx),
        "outliers_winsorized": n_flagged,
        "winsorize_bounds": (float(lo), float(hi)),
    }


def monthly_lift_range(df: pd.DataFrame) -> tuple[pd.Series, float]:
    """Naive per-cohort lift, showing the effect is not time-stationary."""
    monthly = df.groupby("cohort_month").apply(
        lambda g: g.loc[g.treated == 1, "converted"].mean()
        - g.loc[g.treated == 0, "converted"].mean()
    )
    return monthly, float(monthly.max() - monthly.min())


def validate_schema(df: pd.DataFrame, expected: dict[str, str] = EXPECTED_SCHEMA) -> list[str]:
    """Return a list of schema problems; empty list means the schema is clean."""
    issues = []
    for col, kind in expected.items():
        if col not in df.columns:
            issues.append(f"missing column: {col}")
            continue
        dtype = str(df[col].dtype)
        if kind == "float" and "float" not in dtype:
            issues.append(f"{col}: expected float, got {dtype}")
        elif kind == "int" and not any(t in dtype for t in ("int", "float")):
            issues.append(f"{col}: expected int-like, got {dtype}")
        elif kind == "object" and dtype not in ("object", "str", "string", "category"):
            issues.append(f"{col}: expected object/category, got {dtype}")
    return issues


def check_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> int:
    return int(df.duplicated(subset=subset).sum())


def check_treatment_imbalance(df: pd.DataFrame, min_share: float = 0.10) -> dict:
    """Flag if either arm is smaller than `min_share` of the population."""
    share = df["treated"].mean()
    return {
        "treated_share": float(share),
        "control_share": float(1 - share),
        "imbalanced": bool(share < min_share or (1 - share) < min_share),
    }


def check_temporal_leakage(df: pd.DataFrame, time_col: str = "cohort_month") -> dict:
    """A crude temporal-leakage guard: features must not encode post-outcome info.

    Verifies the pre-treatment outcome column (if present) does not correlate
    suspiciously highly with the post-treatment outcome, which would suggest
    the "pre" column is not actually pre-treatment.
    """
    if "pre_period_spend" not in df.columns or "converted" not in df.columns:
        return {"checked": False}
    corr = float(np.corrcoef(df["pre_period_spend"], df["converted"])[0, 1])
    return {"checked": True, "pre_vs_post_corr": corr, "suspiciously_high": abs(corr) > 0.9}


def check_feature_leakage(feature_cols: list[str], forbidden: tuple[str, ...] = ("converted", "cate", "propensity", "_true_cate", "_baseline_p", "_true_propensity")) -> list[str]:
    """Flag any outcome- or estimate-derived columns that leaked into features."""
    return [c for c in feature_cols if c in forbidden]
