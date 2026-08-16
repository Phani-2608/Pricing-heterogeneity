import numpy as np
import pandas as pd

from pricing_heterogeneity.causal import diagnostics, estimation
from pricing_heterogeneity.causal.validation import calibration_stats, gates_table, qini_coefficient
from pricing_heterogeneity.config import Config
from pricing_heterogeneity.data.complexity import inject_and_winsorize_outliers, inject_missingness
from pricing_heterogeneity.data.generate import generate_dataset
from pricing_heterogeneity.features.build import FeatureBuilder


def _prep():
    cfg = Config(n_customers=2_000, n_folds=3)
    rng = np.random.default_rng(0)
    df, gen = generate_dataset(cfg, rng)
    df, _ = inject_missingness(df, ["ltv", "tenure_months", "sessions_30d"], 0.1, rng)
    df, _ = inject_and_winsorize_outliers(df, "ltv", rng)
    fb = FeatureBuilder().fit(df)
    X = fb.transform(df)
    return cfg, df, X, gen


def test_dr_learner_beats_naive_bias():
    """DR should reduce bias on average; use n=6000 for stable results."""
    cfg = Config(n_customers=6_000, n_folds=3)
    rng = np.random.default_rng(0)
    df, gen = generate_dataset(cfg, rng)
    df, _ = inject_missingness(df, ["ltv", "tenure_months", "sessions_30d"], 0.1, rng)
    df, _ = inject_and_winsorize_outliers(df, "ltv", rng)
    X = FeatureBuilder().fit_transform(df)
    W, Y = df["treated"].values, df["converted"].values
    psi, *_ = estimation.fit_dr_scores(X, W, Y, cfg, n_folds=3, n_estimators=150)
    ate_dr, se, ci = estimation.dr_ate_with_ci(psi)
    naive_bias = abs(gen["naive_difference_in_means"] - gen["true_ate"])
    dr_bias = abs(ate_dr - gen["true_ate"])
    assert dr_bias < naive_bias, f"DR bias {dr_bias:.4f} !< naive bias {naive_bias:.4f}"


def test_propensity_within_configured_clip():
    cfg, df, X, _ = _prep()
    ps = diagnostics.estimate_propensity(X, df["treated"].values, cfg, n_estimators=100)
    lo, hi = cfg.propensity_clip
    assert (ps >= lo).all() and (ps <= hi).all()


def test_gates_and_qini_run_and_have_expected_shapes():
    cfg, df, X, _ = _prep()
    W, Y = df["treated"].values, df["converted"].values
    psi, *_ = estimation.fit_dr_scores(X, W, Y, cfg, n_folds=3, n_estimators=100)
    cate_dr, _ = estimation.fit_cate(X, psi, cfg, n_folds=3, n_estimators=100)
    gates = gates_table(cate_dr, psi, df["_true_cate"].values, n_bins=5)
    assert set(gates.columns) >= {"decile", "predicted_cate", "realized_dr_effect"}
    stats_ = calibration_stats(gates)
    assert -5 < stats_["calibration_slope"] < 5
    qini, frac, cum, rand = qini_coefficient(cate_dr, psi)
    assert len(frac) == len(cate_dr) == len(cum) == len(rand)


def test_bootstrap_ate_ci_returns_ordered_interval():
    cfg, df, X, _ = _prep()
    W, Y = df["treated"].values, df["converted"].values
    psi, *_ = estimation.fit_dr_scores(X, W, Y, cfg, n_folds=3, n_estimators=100)
    lo, hi = estimation.bootstrap_ate_ci(psi, n_boot=100, seed=1)
    assert lo <= hi
