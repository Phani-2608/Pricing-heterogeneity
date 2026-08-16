"""CATE/ITE estimator validation: GATES calibration and Qini ranking quality."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def gates_table(
    cate: np.ndarray, psi: np.ndarray, true_cate: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Sorted group-average treatment effects (GATES), by predicted-CATE decile."""
    decile = pd.qcut(pd.Series(cate), n_bins, labels=False, duplicates="drop")
    rows = []
    for d in sorted(decile.unique()):
        m = (decile == d).values
        rows.append(
            {
                "decile": int(d) + 1,
                "n": int(m.sum()),
                "predicted_cate": float(cate[m].mean()),
                "realized_dr_effect": float(psi[m].mean()),
                "true_cate": float(true_cate[m].mean()),
            }
        )
    return pd.DataFrame(rows)


def calibration_stats(gates_df: pd.DataFrame) -> dict:
    slope, intercept, r, _, _ = stats.linregress(
        gates_df["predicted_cate"], gates_df["realized_dr_effect"]
    )
    rank_corr = float(
        stats.spearmanr(gates_df["predicted_cate"], gates_df["realized_dr_effect"]).statistic
    )
    return {
        "calibration_slope": float(slope),
        "calibration_r2": float(r ** 2),
        "gates_rank_correlation": rank_corr,
    }


def qini_coefficient(cate: np.ndarray, psi: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Qini curve and coefficient: value of ranking customers by estimated CATE."""
    n = len(cate)
    order = np.argsort(-cate)
    psi_sorted = psi[order]
    cum_gain = np.cumsum(psi_sorted)
    frac = np.arange(1, n + 1) / n
    random_gain = frac * psi.sum()
    trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2.0 renamed this
    qini = float(trapz(cum_gain - random_gain, frac) / n)
    return qini, frac, cum_gain, random_gain
