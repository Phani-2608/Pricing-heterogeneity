"""Translate individual CATE/ITE estimates into interpretable, stable segments."""
from __future__ import annotations

import numpy as np
import pandas as pd


def segment_customers(
    df: pd.DataFrame,
    cate: np.ndarray,
    psi: np.ndarray,
    true_cate: np.ndarray,
    q: int = 3,
    labels: tuple[str, ...] = ("LOW", "MEDIUM", "HIGH"),
) -> tuple[pd.Series, pd.DataFrame]:
    """Quantile-bucket customers by estimated CATE into named segments."""
    segment = pd.qcut(pd.Series(cate), q=q, labels=list(labels), duplicates="drop")

    rows = []
    for s in labels:
        m = (segment == s).values
        if not m.any():
            continue
        rows.append(
            {
                "segment": s,
                "n": int(m.sum()),
                "share": float(m.mean()),
                "est_cate": float(cate[m].mean()),
                "realized_dr_effect": float(psi[m].mean()),
                "true_cate": float(true_cate[m].mean()),
                "avg_ltv": float(df.loc[m, "ltv"].mean()),
                "avg_age": float(df.loc[m, "age"].mean()),
            }
        )
    seg_df = pd.DataFrame(rows)
    return segment, seg_df


def subgroup_stability(
    df: pd.DataFrame, segment: pd.Series, group_col: str, min_share_diff: float = 0.15
) -> pd.DataFrame:
    """Check whether segment composition is stable across a subgroup (e.g. region).

    Flags a subgroup if its share of any segment deviates a lot from that
    segment's overall share, which would suggest the segmentation is really
    picking up the subgroup rather than treatment response.
    """
    overall_share = segment.value_counts(normalize=True)
    rows = []
    for g, idx in df.groupby(group_col, observed=True).groups.items():
        sub_share = segment.loc[idx].value_counts(normalize=True)
        for seg_label in overall_share.index:
            s = float(sub_share.get(seg_label, 0.0))
            o = float(overall_share.get(seg_label, 0.0))
            rows.append(
                {
                    "group": str(g),
                    "segment": str(seg_label),
                    "group_share": s,
                    "overall_share": o,
                    "deviation": s - o,
                    "unstable": abs(s - o) > min_share_diff,
                }
            )
    return pd.DataFrame(rows)
