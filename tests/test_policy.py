import numpy as np
import pandas as pd

from pricing_heterogeneity.config import Config
from pricing_heterogeneity.optimization.policy_engine import (
    break_even_ite,
    decide_for_customer,
    score_population,
)
from pricing_heterogeneity.optimization.strategies import compare_strategies, uplift_curve


def test_break_even_is_positive_and_scales_with_baseline():
    cfg = Config()
    lo = break_even_ite(0.10, cfg)
    hi = break_even_ite(0.50, cfg)
    assert lo > 0 and hi > lo


def test_decide_for_customer_high_ite_offers_discount():
    """A large positive ITE that clearly beats the break-even threshold triggers OFFER_DISCOUNT."""
    cfg = Config()
    # For baseline_p=0.20 with margin=0.40 and discount=0.15, break-even ITE is ~0.12.
    d = decide_for_customer(
        customer_id=1, predicted_ite=0.25, baseline_p=0.20,
        ite_se=0.01, customer_value=100.0, cfg=cfg, model_version="test",
    )
    assert d.action == "OFFER_DISCOUNT"
    assert d.confidence in {"HIGH", "MEDIUM"}


def test_decide_for_customer_negative_ite_declines():
    cfg = Config()
    d = decide_for_customer(
        customer_id=2, predicted_ite=-0.05, baseline_p=0.20,
        ite_se=0.01, customer_value=100.0, cfg=cfg, model_version="test",
    )
    assert d.action == "NO_ACTION"


def test_score_population_returns_row_per_customer():
    cfg = Config()
    n = 50
    df = pd.DataFrame({
        "customer_id": range(n), "ltv": np.linspace(50, 500, n),
    })
    cate = np.linspace(-0.1, 0.3, n)
    baseline = np.full(n, 0.2)
    out = score_population(df, cate, baseline, cfg, model_version="v-test")
    assert len(out) == n
    assert (out["model_version"] == "v-test").all()


def test_compare_strategies_yields_all_named_strategies():
    cfg = Config()
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "ltv": rng.uniform(50, 500, n),
        "segment": pd.Categorical(rng.choice(["LOW", "MEDIUM", "HIGH"], n)),
    })
    cate = rng.normal(0.05, 0.1, n)
    baseline_p = rng.uniform(0.05, 0.4, n)
    true_cate = cate + rng.normal(0, 0.02, n)
    strat_df, summary = compare_strategies(df, cate, baseline_p, true_cate, cfg, rng)
    assert set(strat_df["strategy"]) == {
        "treat_none", "treat_all", "random_targeting", "segment_high", "ite_targeted"
    }
    assert "best_strategy" in summary


def test_uplift_curve_ends_at_total_uplift():
    rng = np.random.default_rng(1)
    n = 300
    cate = rng.normal(0.05, 0.1, n)
    psi = cate + rng.normal(0, 0.05, n)
    upl = uplift_curve(cate, psi, n_points=20)

    assert np.isclose(
        upl["cumulative_uplift"].iloc[-1],
        psi.sum(),
        rtol=1e-12,
        atol=1e-12,
    )
