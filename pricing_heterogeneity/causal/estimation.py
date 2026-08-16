"""Double Machine Learning: the causal centerpiece of this project.

Estimates ATE, CATE, and per-customer ITE via a cross-fitted doubly-robust
learner (Chernozhukov et al.):
  1. K-fold split. Nuisances trained out-of-fold to avoid own-observation bias.
  2. Fit e(X) = P(W=1|X), mu1(X) = E[Y|X,W=1], mu0(X) = E[Y|X,W=0].
  3. Doubly-robust pseudo-outcome:
        psi = mu1 - mu0 + W(Y-mu1)/e - (1-W)(Y-mu0)/(1-e)
     Consistent if EITHER the outcome model OR the propensity model is correct.
  4. Regress psi on X (again cross-fitted) to obtain the CATE function -- this
     is also each customer's individual treatment effect (ITE) estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold

from ..config import Config


def fit_dr_scores(
    X: pd.DataFrame,
    W: np.ndarray,
    Y: np.ndarray,
    cfg: Config,
    n_folds: int | None = None,
    n_estimators: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cross-fitted doubly-robust pseudo-outcomes. Returns (psi, mu1, mu0, e)."""
    n_folds = n_folds or cfg.n_folds
    nn = len(Y)
    mu1 = np.zeros(nn)
    mu0 = np.zeros(nn)
    e = np.zeros(nn)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.random_state)
    for tr, te in skf.split(X, W):
        Xtr, Wtr, Ytr = X.iloc[tr], W[tr], Y[tr]

        pm = RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=25,
            random_state=cfg.random_state, n_jobs=-1,
        )
        pm.fit(Xtr, Wtr)
        e[te] = pm.predict_proba(X.iloc[te])[:, 1]

        m1 = RandomForestRegressor(
            n_estimators=n_estimators, min_samples_leaf=15,
            random_state=cfg.random_state, n_jobs=-1,
        )
        m1.fit(Xtr[Wtr == 1], Ytr[Wtr == 1])
        mu1[te] = m1.predict(X.iloc[te])

        m0 = RandomForestRegressor(
            n_estimators=n_estimators, min_samples_leaf=15,
            random_state=cfg.random_state, n_jobs=-1,
        )
        m0.fit(Xtr[Wtr == 0], Ytr[Wtr == 0])
        mu0[te] = m0.predict(X.iloc[te])

    e = np.clip(e, *cfg.propensity_clip)
    psi = (mu1 - mu0) + W * (Y - mu1) / e - (1 - W) * (Y - mu0) / (1 - e)
    return psi, mu1, mu0, e


def fit_cate(
    X: pd.DataFrame, psi: np.ndarray, cfg: Config, n_folds: int | None = None, n_estimators: int = 400
) -> tuple[np.ndarray, RandomForestRegressor]:
    """Cross-fitted CATE/ITE regression on the doubly-robust pseudo-outcome.

    Returns the out-of-fold CATE array plus a model refit on the full data,
    which is what gets saved and served for scoring new customers.
    """
    n_folds = n_folds or cfg.n_folds
    cate = np.zeros(len(psi))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=cfg.random_state)
    for tr, te in kf.split(X):
        m = RandomForestRegressor(
            n_estimators=n_estimators, min_samples_leaf=40, max_features=0.5,
            random_state=cfg.random_state, n_jobs=-1,
        )
        m.fit(X.iloc[tr], psi[tr])
        cate[te] = m.predict(X.iloc[te])

    final_model = RandomForestRegressor(
        n_estimators=n_estimators, min_samples_leaf=40, max_features=0.5,
        random_state=cfg.random_state, n_jobs=-1,
    )
    final_model.fit(X, psi)
    return cate, final_model


def dr_ate_with_ci(psi: np.ndarray) -> tuple[float, float, tuple[float, float]]:
    """Doubly-robust ATE with an influence-function standard error and 95% CI."""
    n = len(psi)
    ate = float(psi.mean())
    se = float(psi.std(ddof=1) / np.sqrt(n))
    ci = (ate - 1.96 * se, ate + 1.96 * se)
    return ate, se, ci


def bootstrap_ate_ci(
    psi: np.ndarray, n_boot: int = 500, seed: int = 42, alpha: float = 0.05
) -> tuple[float, float]:
    """Nonparametric bootstrap CI on the ATE, as a robustness check on the
    analytic influence-function CI above."""
    rng = np.random.default_rng(seed)
    n = len(psi)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means[b] = psi[idx].mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def t_learner_baseline(
    X: pd.DataFrame, W: np.ndarray, Y: np.ndarray, cfg: Config, n_estimators: int = 300
) -> np.ndarray:
    """Naive T-learner: separate outcome models per arm, no cross-fitting.

    Used only as a baseline to show what the simpler approach gets wrong.
    """
    mt = RandomForestRegressor(n_estimators=n_estimators, random_state=cfg.random_state, n_jobs=-1)
    mt.fit(X[W == 1], Y[W == 1])
    mc = RandomForestRegressor(n_estimators=n_estimators, random_state=cfg.random_state, n_jobs=-1)
    mc.fit(X[W == 0], Y[W == 0])
    return mt.predict(X) - mc.predict(X)
