import numpy as np
import pandas as pd

from pricing_heterogeneity.evaluation.fairness import fairness_audit, passes_four_fifths
from pricing_heterogeneity.evaluation.stats import (
    minimum_detectable_effect,
    power_table,
    sample_size_two_proportions,
    segment_test_sample_size,
    two_proportion_z_test,
)
from pricing_heterogeneity.monitoring.drift import (
    feature_drift,
    prediction_drift,
    psi,
    treatment_effect_drift,
)


def test_z_test_detects_real_difference():
    z, p = two_proportion_z_test(300, 1000, 200, 1000)
    assert p < 0.01 and z > 0


def test_z_test_no_difference():
    z, p = two_proportion_z_test(200, 1000, 205, 1000)
    assert p > 0.05


def test_sample_size_grows_as_mde_shrinks():
    n_small = sample_size_two_proportions(0.2, 0.22)
    n_big = sample_size_two_proportions(0.2, 0.30)
    assert n_small > n_big > 0


def test_mde_grows_as_n_shrinks():
    m_lo = minimum_detectable_effect(0.2, n_per_arm=1_000)
    m_hi = minimum_detectable_effect(0.2, n_per_arm=100)
    assert m_hi > m_lo > 0


def test_power_table_shape_and_bonferroni_bigger():
    tbl = power_table(0.2)
    assert all("mde_absolute" in row for row in tbl)
    normal = sample_size_two_proportions(0.2, 0.23)
    bonf = segment_test_sample_size(0.2, mde=0.03, n_segments=3)
    assert bonf > normal


def test_fairness_audit_returns_ratio():
    df = pd.DataFrame({
        "group": ["A"] * 100 + ["B"] * 100,
        "cate": np.concatenate([np.full(100, 0.1), np.full(100, 0.05)]),
    })
    would_treat = pd.Series([1] * 80 + [0] * 20 + [1] * 30 + [0] * 70)
    fair_df, ratio = fairness_audit(df, "group", would_treat)
    assert 0 < ratio < 1
    assert not passes_four_fifths(0.4)
    assert passes_four_fifths(0.85)


def test_psi_identical_distributions_is_near_zero():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    p = psi(ref, ref)
    assert p < 0.01


def test_psi_flags_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    shifted = rng.normal(1.5, 1, 5000)
    assert psi(ref, shifted) > 0.25


def test_feature_drift_flags_shifted_column():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 2000), "y": rng.normal(0, 1, 2000)})
    cur = pd.DataFrame({"x": rng.normal(2, 1, 2000), "y": rng.normal(0, 1, 2000)})
    fd = feature_drift(ref, cur)
    assert fd.set_index("feature").loc["x", "drift_flag"]
    assert not fd.set_index("feature").loc["y", "drift_flag"]


def test_prediction_and_treatment_effect_drift_return_expected_keys():
    rng = np.random.default_rng(2)
    ref = rng.normal(0.05, 0.1, 3000)
    cur = rng.normal(0.20, 0.1, 3000)
    pd_ = prediction_drift(ref, cur)
    td = treatment_effect_drift(ref, cur)
    assert set(pd_) >= {"psi", "ks_stat", "ks_p_value", "drift_flag"}
    assert td["mean_shift"] > 0.10
