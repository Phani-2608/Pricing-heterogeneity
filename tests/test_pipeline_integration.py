"""Integration test: full pipeline runs end-to-end at a small size and
produces the expected result keys, artifacts, and a saved model."""
import json
import os

from pricing_heterogeneity.config import Config
from pricing_heterogeneity.pipeline import run_pipeline


def test_pipeline_end_to_end_small(tmp_path):
    cfg = Config(
        n_customers=2_000, n_folds=3, out_dir=str(tmp_path / "outputs"),
        model_version="test-1.0.0",
    )
    results = run_pipeline(cfg, save=True)

    for key in (
        "ate_dr", "ate_dr_ci95", "ate_dr_bootstrap_ci95",
        "qini", "calibration_slope", "gates_rank_correlation",
        "segments", "strategy_comparison", "fairness_disparity_ratio",
        "power_analysis", "validation_tests", "tests_passed", "tests_total",
    ):
        assert key in results, f"missing key: {key}"

    for f in ("results.json", "customer_level_results.csv", "segment_summary.csv",
              "gates_calibration.csv", "strategy_comparison.csv",
              "customer_decisions.csv", "uplift_curve.csv"):
        assert (tmp_path / "outputs" / f).exists(), f"missing artifact: {f}"

    assert (tmp_path / "outputs" / "registry" / "test-1.0.0" / "model.pkl").exists()
    assert (tmp_path / "outputs" / "registry" / "test-1.0.0" / "metadata.json").exists()

    with open(tmp_path / "outputs" / "results.json") as f:
        saved = json.load(f)
    assert saved["tests_total"] > 10
