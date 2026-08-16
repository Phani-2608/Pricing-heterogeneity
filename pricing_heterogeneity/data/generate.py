"""Synthetic customer, transaction, and pricing-experiment tables with a known
ground-truth treatment effect.

Ground truth is generated so estimator accuracy can be scored directly,
something that is never possible on real data without a held-out experiment.
Treatment assignment is deliberately confounded, not randomized, to mirror
how real marketing/pricing data actually looks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Config


def generate_dataset(cfg: Config, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Generate customers, a confounded treatment assignment, and outcomes.

    Returns
    -------
    df : DataFrame with covariates, treated, converted, and oracle columns
         (_true_cate, _baseline_p) kept ONLY for validation, never for fitting.
    summary : dict of headline generation statistics.
    """
    n = cfg.n_customers

    age = rng.normal(38, 12, n).clip(18, 78)
    ltv = rng.lognormal(5.4, 0.9, n)
    purchase_freq = rng.poisson(5, n).astype(float)
    tenure_months = rng.exponential(22, n).clip(0, 120)
    prior_discounts = rng.binomial(6, 0.45, n).astype(float)
    sessions_30d = rng.poisson(8, n).astype(float)
    category = rng.choice(["Electronics", "Apparel", "Home", "Beauty"], n)
    region = rng.choice(["North", "South", "East", "West"], n)
    cohort_month = rng.integers(1, 13, n)

    # Pre-treatment outcome, used later as a placebo outcome: treatment cannot
    # have caused something that happened before it.
    pre_period_spend = (
        45 + 0.55 * ltv / 10 + 3.0 * purchase_freq + rng.normal(0, 18, n)
    ).clip(0, None)

    df = pd.DataFrame(
        {
            "age": age,
            "ltv": ltv,
            "purchase_freq": purchase_freq,
            "tenure_months": tenure_months,
            "prior_discounts": prior_discounts,
            "sessions_30d": sessions_30d,
            "category": category,
            "region": region,
            "cohort_month": cohort_month,
            "pre_period_spend": pre_period_spend,
        }
    )

    # --- Ground-truth heterogeneous treatment effect -----------------------
    ltv_pct = stats.rankdata(df["ltv"]) / n
    age_pct = stats.rankdata(df["age"]) / n
    disc_pct = stats.rankdata(df["prior_discounts"]) / n

    true_cate = (
        0.34 * (1 - ltv_pct)
        + 0.10 * (1 - age_pct)
        + 0.08 * disc_pct
        - 0.20
        + rng.normal(0, 0.02, n)
    )

    seasonal = 1.0 + 0.22 * np.cos(2 * np.pi * (df["cohort_month"] - 1) / 12)
    true_cate = true_cate * seasonal

    baseline_p = (
        0.10 + 0.16 * ltv_pct + 0.010 * df["purchase_freq"] + rng.normal(0, 0.01, n)
    ).clip(0.03, 0.60)

    # --- Confounded assignment ----------------------------------------------
    propensity_logit = (
        -0.35
        + 0.9 * (disc_pct - 0.5)
        + 0.7 * (stats.rankdata(df["tenure_months"]) / n - 0.5)
        - 0.6 * (ltv_pct - 0.5)
    )
    true_propensity = 1 / (1 + np.exp(-propensity_logit))
    treated = rng.binomial(1, true_propensity)

    p_outcome = (baseline_p + true_cate * treated).clip(0.001, 0.999)
    converted = rng.binomial(1, p_outcome)

    df["treated"] = treated
    df["converted"] = converted
    df["_true_cate"] = true_cate
    df["_baseline_p"] = baseline_p
    df["_true_propensity"] = true_propensity

    true_ate = float(true_cate.mean())
    naive_ate = float(
        df.loc[treated == 1, "converted"].mean() - df.loc[treated == 0, "converted"].mean()
    )

    summary = {
        "n_customers": n,
        "treated_share": float(treated.mean()),
        "overall_conversion": float(converted.mean()),
        "true_ate": true_ate,
        "naive_difference_in_means": naive_ate,
        "naive_bias": naive_ate - true_ate,
    }
    return df, summary
