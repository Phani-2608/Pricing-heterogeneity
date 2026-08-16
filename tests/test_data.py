import numpy as np

from pricing_heterogeneity.config import Config
from pricing_heterogeneity.data.complexity import (
    check_duplicates,
    check_feature_leakage,
    check_treatment_imbalance,
    inject_and_winsorize_outliers,
    inject_missingness,
    monthly_lift_range,
    validate_schema,
)
from pricing_heterogeneity.data.generate import generate_dataset


def _small_cfg():
    return Config(n_customers=1_500, n_folds=3)


def test_generate_dataset_has_expected_columns():
    df, summary = generate_dataset(_small_cfg(), np.random.default_rng(0))
    for col in ("age", "ltv", "treated", "converted", "_true_cate", "_baseline_p", "_true_propensity"):
        assert col in df.columns
    assert len(df) == 1_500


def test_generate_dataset_treatment_is_confounded():
    """Confounding is the whole point: naive != true."""
    df, summary = generate_dataset(_small_cfg(), np.random.default_rng(0))
    assert abs(summary["naive_difference_in_means"] - summary["true_ate"]) > 1e-3


def test_missingness_injection_and_imputation():
    df, _ = generate_dataset(_small_cfg(), np.random.default_rng(0))
    df2, info = inject_missingness(df, ["ltv", "tenure_months", "sessions_30d"], 0.1, np.random.default_rng(1))
    assert info["missing_rate_injected"] == 0.1
    for c in ("ltv_was_missing", "tenure_months_was_missing", "sessions_30d_was_missing"):
        assert c in df2.columns
    # Imputation must succeed
    assert not df2[["ltv", "tenure_months", "sessions_30d"]].isna().any().any()


def test_winsorize_bounds_clip_extremes():
    df, _ = generate_dataset(_small_cfg(), np.random.default_rng(0))
    df2, info = inject_and_winsorize_outliers(df, "ltv", np.random.default_rng(2))
    assert df2["ltv"].max() <= info["winsorize_bounds"][1] + 1e-6


def test_schema_and_duplicates_and_imbalance():
    df, _ = generate_dataset(_small_cfg(), np.random.default_rng(0))
    assert validate_schema(df) == []
    assert check_duplicates(df) >= 0
    imb = check_treatment_imbalance(df)
    assert 0.0 < imb["treated_share"] < 1.0


def test_check_feature_leakage_flags_outcome_column():
    leaks = check_feature_leakage(["age", "ltv", "converted", "cate"])
    assert "converted" in leaks and "cate" in leaks


def test_monthly_lift_range_returns_positive_range():
    df, _ = generate_dataset(_small_cfg(), np.random.default_rng(0))
    monthly, rng_val = monthly_lift_range(df)
    assert rng_val >= 0.0
    assert len(monthly) == 12
