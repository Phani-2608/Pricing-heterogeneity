"""Falsification tests: the estimator should find nothing where nothing exists."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from .estimation import fit_dr_scores


def placebo_treatment_test(
    X: pd.DataFrame, W: np.ndarray, Y: np.ndarray, cfg: Config, rng: np.random.Generator
) -> dict:
    W_placebo = rng.binomial(1, W.mean(), len(W))
    psi_p, *_ = fit_dr_scores(X, W_placebo, Y, cfg, n_folds=3, n_estimators=150)
    ate_p = float(psi_p.mean())
    se_p = float(psi_p.std(ddof=1) / np.sqrt(len(psi_p)))
    t_p = ate_p / se_p if se_p else float("nan")
    return {"placebo_treatment_ate": ate_p, "placebo_treatment_t": t_p, "passed": abs(t_p) < 2.0}


def placebo_outcome_test(X: pd.DataFrame, W: np.ndarray, Y_pre: np.ndarray, cfg: Config) -> dict:
    psi_pre, *_ = fit_dr_scores(X, W, Y_pre, cfg, n_folds=3, n_estimators=150)
    ate_pre = float(psi_pre.mean())
    se_pre = float(psi_pre.std(ddof=1) / np.sqrt(len(psi_pre)))
    t_pre = ate_pre / se_pre if se_pre else float("nan")
    std_effect = ate_pre / Y_pre.std()
    return {
        "placebo_outcome_ate": ate_pre,
        "placebo_outcome_t": t_pre,
        "placebo_outcome_std_effect": float(std_effect),
        "passed": abs(std_effect) < 0.10,
    }
