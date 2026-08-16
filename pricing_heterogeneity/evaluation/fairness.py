"""Fairness audit: four-fifths (disparate-impact) check on a targeting policy."""
from __future__ import annotations

import numpy as np
import pandas as pd

FOUR_FIFTHS_THRESHOLD = 0.80


def fairness_audit(df: pd.DataFrame, group_col: str, would_treat: pd.Series) -> tuple[pd.DataFrame, float]:
    work = df.copy()
    work["_would_treat"] = would_treat.values

    rows = []
    for g, sub_g in work.groupby(group_col, observed=True):
        rows.append({
            "group": str(g),
            "n": len(sub_g),
            "treat_rate": float(sub_g["_would_treat"].mean()),
            "mean_cate": float(sub_g["cate"].mean()) if "cate" in sub_g else float("nan"),
        })
    fair_df = pd.DataFrame(rows)
    rates = fair_df["treat_rate"].replace(0, np.nan)
    disparity = float(rates.min() / rates.max()) if rates.max() > 0 else float("nan")
    return fair_df, disparity


def passes_four_fifths(disparity_ratio: float) -> bool:
    return disparity_ratio >= FOUR_FIFTHS_THRESHOLD
