"""Pricing-policy engine.

Turns each customer's estimated ITE into a concrete pricing recommendation
by combining predicted treatment effect, expected conversion lift, customer
value, discount cost, and expected incremental revenue. This is the piece
that makes the causal estimates actually actionable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from ..config import Config

Action = Literal["OFFER_DISCOUNT", "NO_ACTION"]


@dataclass
class PricingDecision:
    customer_id: int | str
    predicted_ite: float
    ite_lower: float
    ite_upper: float
    expected_conv_lift: float
    customer_value: float
    intervention_cost: float
    expected_incremental_revenue: float
    action: Action
    confidence: str
    reason: str
    model_version: str

    def as_dict(self) -> dict:
        return asdict(self)


def _confidence_label(lower: float, upper: float, threshold: float) -> str:
    """Rate confidence by whether the CI clears the break-even threshold."""
    if lower > threshold:
        return "HIGH"      # definitely worth treating
    if upper < threshold:
        return "HIGH"      # definitely not worth treating
    if lower > 0 or upper < 0:
        return "MEDIUM"    # sign is clear but magnitude is uncertain
    return "LOW"           # sign itself is uncertain


def break_even_ite(baseline_p: float, cfg: Config) -> float:
    """Smallest ITE such that the discounted-margin outcome beats the un-discounted one.

    Derived from:
        untreated_profit = p0 * AOV * margin
        treated_profit   = (p0 + tau) * AOV * (margin - discount)
    Solve for tau where treated_profit > untreated_profit.
    """
    m, d = cfg.gross_margin, cfg.discount_depth
    return (baseline_p * m - baseline_p * (m - d)) / (m - d)


def decide_for_customer(
    *,
    customer_id,
    predicted_ite: float,
    baseline_p: float,
    ite_se: float | None,
    customer_value: float,
    cfg: Config,
    model_version: str,
) -> PricingDecision:
    """Score one customer. `ite_se` may be None if uncertainty isn't tracked."""
    threshold = break_even_ite(baseline_p, cfg)
    intervention_cost = customer_value * cfg.discount_depth
    expected_incremental_revenue = predicted_ite * customer_value * (cfg.gross_margin - cfg.discount_depth)

    if ite_se is None or not np.isfinite(ite_se):
        lower, upper = predicted_ite, predicted_ite
    else:
        lower, upper = predicted_ite - 1.96 * ite_se, predicted_ite + 1.96 * ite_se

    action: Action = "OFFER_DISCOUNT" if predicted_ite > threshold else "NO_ACTION"
    confidence = _confidence_label(lower, upper, threshold)

    reason = (
        f"ITE {predicted_ite:+.4f} vs break-even {threshold:+.4f}; "
        f"expected incremental revenue ${expected_incremental_revenue:+.2f}"
    )

    return PricingDecision(
        customer_id=customer_id,
        predicted_ite=float(predicted_ite),
        ite_lower=float(lower),
        ite_upper=float(upper),
        expected_conv_lift=float(predicted_ite),
        customer_value=float(customer_value),
        intervention_cost=float(intervention_cost),
        expected_incremental_revenue=float(expected_incremental_revenue),
        action=action,
        confidence=confidence,
        reason=reason,
        model_version=model_version,
    )


def score_population(
    df: pd.DataFrame,
    cate: np.ndarray,
    baseline_p: np.ndarray,
    cfg: Config,
    model_version: str,
    ite_se: np.ndarray | None = None,
    customer_value_col: str = "ltv",
) -> pd.DataFrame:
    """Score every customer with the pricing policy engine."""
    ids = df.index if "customer_id" not in df.columns else df["customer_id"].values
    values = df[customer_value_col].values
    rows = []
    for i, cid in enumerate(ids):
        d = decide_for_customer(
            customer_id=cid,
            predicted_ite=float(cate[i]),
            baseline_p=float(baseline_p[i]),
            ite_se=float(ite_se[i]) if ite_se is not None else None,
            customer_value=float(values[i]),
            cfg=cfg,
            model_version=model_version,
        )
        rows.append(d.as_dict())
    return pd.DataFrame(rows)
