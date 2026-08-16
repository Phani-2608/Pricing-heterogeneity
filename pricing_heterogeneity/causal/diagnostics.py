"""Design diagnostics: covariate balance, propensity estimation, overlap."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from ..config import Config


def standardized_mean_diff(x: np.ndarray, w: np.ndarray) -> float:
    x1, x0 = x[w == 1], x[w == 0]
    pooled = np.sqrt((x1.var(ddof=1) + x0.var(ddof=1)) / 2)
    return 0.0 if pooled == 0 else float((x1.mean() - x0.mean()) / pooled)


def weighted_smd(x: np.ndarray, w: np.ndarray, wt: np.ndarray) -> float:
    m1 = np.average(x[w == 1], weights=wt[w == 1])
    m0 = np.average(x[w == 0], weights=wt[w == 0])
    v1 = np.average((x[w == 1] - m1) ** 2, weights=wt[w == 1])
    v0 = np.average((x[w == 0] - m0) ** 2, weights=wt[w == 0])
    pooled = np.sqrt((v1 + v0) / 2)
    return 0.0 if pooled == 0 else float((m1 - m0) / pooled)


def covariate_balance(X: pd.DataFrame, W: np.ndarray, threshold: float = 0.10) -> tuple[pd.Series, int]:
    """Standardized mean differences per covariate, sorted by |SMD|."""
    smd = {c: standardized_mean_diff(X[c].values, W) for c in X.columns}
    smd_s = pd.Series(smd).sort_values(key=np.abs, ascending=False)
    n_imbalanced = int((np.abs(smd_s) > threshold).sum())
    return smd_s, n_imbalanced


def estimate_propensity(
    X: pd.DataFrame, W: np.ndarray, cfg: Config, n_estimators: int = 300
) -> np.ndarray:
    """Cross-fitted propensity scores, clipped to avoid extreme weights."""
    n = len(W)
    ps = np.zeros(n)
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_state)
    for tr, te in skf.split(X, W):
        m = RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=25,
            random_state=cfg.random_state, n_jobs=-1,
        )
        m.fit(X.iloc[tr], W[tr])
        ps[te] = m.predict_proba(X.iloc[te])[:, 1]
    return np.clip(ps, *cfg.propensity_clip)


def positivity_violations(ps: np.ndarray, lo: float = 0.10, hi: float = 0.90) -> int:
    return int(((ps < lo) | (ps > hi)).sum())


def ipw_weights(W: np.ndarray, ps: np.ndarray) -> np.ndarray:
    return np.where(W == 1, 1 / ps, 1 / (1 - ps))
