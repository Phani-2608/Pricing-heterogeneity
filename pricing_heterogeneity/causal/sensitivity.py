"""Sensitivity to unmeasured confounding.

Simulates an omitted confounder U that shifts both assignment and outcome,
and asks how strong it must be to drive the estimate to zero. This is the
Rosenbaum-bounds question posed as a simulation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from .estimation import fit_dr_scores


def sensitivity_grid(
    X: pd.DataFrame,
    true_propensity: np.ndarray,
    baseline_p: np.ndarray,
    true_cate: np.ndarray,
    ate_dr: float,
    cfg: Config,
    rng: np.random.Generator,
    gammas: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30),
) -> tuple[pd.DataFrame, float]:
    n = len(true_propensity)
    rows = []
    for gamma in gammas:
        U = rng.normal(0, 1, n)
        W_c = rng.binomial(1, np.clip(true_propensity + gamma * (U - U.mean()) / 4, 0.05, 0.95))
        Y_c = rng.binomial(1, np.clip(baseline_p + true_cate * W_c + gamma * U * 0.5, 0.001, 0.999))
        psi_c, *_ = fit_dr_scores(X, W_c, Y_c, cfg, n_folds=3, n_estimators=120)
        a = float(psi_c.mean())
        rows.append({
            "confounder_strength": gamma,
            "estimated_ate": a,
            "pct_of_baseline": a / ate_dr if ate_dr else float("nan"),
        })
    sens_df = pd.DataFrame(rows)
    killed = sens_df[sens_df["estimated_ate"] <= 0]
    breakpoint_gamma = float(killed["confounder_strength"].iloc[0]) if len(killed) else float("inf")
    return sens_df, breakpoint_gamma
