"""Compare targeting strategies on business KPIs.

Strategies compared:
    * treat_none       — no intervention
    * treat_all        — blanket discounting
    * random_targeting — random treated share matching ITE-policy
    * ite_targeted     — treat only where ITE clears the break-even threshold
    * segment_high     — treat only the top segment

KPIs: incremental conversions, revenue, intervention cost, ROI, policy value,
plus uplift and Qini curves for ranking-quality visualization.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from .policy_engine import break_even_ite


def _profit_arrays(baseline_p: np.ndarray, tau: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Per-customer profit if untreated vs treated."""
    m, d, aov = cfg.gross_margin, cfg.discount_depth, cfg.avg_order_value
    profit_untreated = baseline_p * aov * m
    profit_treated = (baseline_p + tau).clip(0, 1) * aov * (m - d)
    return profit_untreated, profit_treated


def compare_strategies(
    df: pd.DataFrame,
    cate_dr: np.ndarray,
    baseline_p: np.ndarray,
    true_cate: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Compare targeting strategies on profit / conversions / ROI.

    `true_cate` is the oracle used to *realize* outcomes, exactly as it is in
    an offline evaluation. Policies themselves may only act on `cate_dr`, the
    estimate, mirroring what a real deployment would have.
    """
    n = len(df)
    profit_un, profit_tr = _profit_arrays(baseline_p, true_cate, cfg)

    thresholds = np.array([break_even_ite(p, cfg) for p in baseline_p])
    ite_targeted = (cate_dr > thresholds).astype(int)
    target_share = float(ite_targeted.mean())
    random_targeted = rng.binomial(1, target_share, n)

    strategies = {
        "treat_none": np.zeros(n, dtype=int),
        "treat_all": np.ones(n, dtype=int),
        "random_targeting": random_targeted,
        "segment_high": (df["segment"] == "HIGH").astype(int).values,
        "ite_targeted": ite_targeted,
    }

    base_conversions = baseline_p.sum()
    base_profit = profit_un.sum()

    rows = []
    for name, pol in strategies.items():
        realized_conv = np.where(pol == 1, (baseline_p + true_cate).clip(0, 1), baseline_p).sum()
        profit = np.where(pol == 1, profit_tr, profit_un).sum()
        intervention_cost = pol.sum() * cfg.avg_order_value * cfg.discount_depth
        incremental_conv = realized_conv - base_conversions
        incremental_profit = profit - base_profit
        roi = (incremental_profit / intervention_cost) if intervention_cost > 0 else float("nan")
        rows.append({
            "strategy": name,
            "treated_share": float(pol.mean()),
            "total_profit": float(profit),
            "incremental_conversions": float(incremental_conv),
            "incremental_profit_vs_treat_none": float(incremental_profit),
            "intervention_cost": float(intervention_cost),
            "roi": float(roi) if np.isfinite(roi) else None,
            "lift_vs_treat_none": float(incremental_profit / base_profit) if base_profit else 0.0,
        })

    strat_df = pd.DataFrame(rows)
    treat_all_profit = strat_df.loc[strat_df.strategy == "treat_all", "total_profit"].iloc[0]
    strat_df["lift_vs_treat_all"] = strat_df["total_profit"] / treat_all_profit - 1

    best_row = strat_df.iloc[strat_df["total_profit"].idxmax()]
    summary = {
        "treat_all_lift_vs_none": float(strat_df.loc[strat_df.strategy == "treat_all", "lift_vs_treat_none"].iloc[0]),
        "best_strategy": str(best_row["strategy"]),
        "best_strategy_lift_vs_treat_all": float(best_row["lift_vs_treat_all"]),
        "best_strategy_lift_vs_none": float(best_row["lift_vs_treat_none"]),
        "best_strategy_roi": best_row["roi"],
    }
    return strat_df, summary


def uplift_curve(cate_dr: np.ndarray, psi: np.ndarray, n_points: int = 100) -> pd.DataFrame:
    """Uplift curve: cumulative incremental conversions as targeting depth grows."""
    order = np.argsort(-cate_dr)
    psi_sorted = psi[order]
    n = len(cate_dr)
    fracs = np.linspace(1 / n_points, 1.0, n_points)
    rows = []
    for f in fracs:
        k = max(1, int(f * n))
        rows.append({
            "targeting_depth": float(f),
            "cumulative_uplift": float(psi_sorted[:k].sum()),
            "random_baseline": float(f * psi.sum()),
        })
    return pd.DataFrame(rows)
