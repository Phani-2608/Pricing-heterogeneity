"""End-to-end training pipeline.

Runs data generation → validation → EDA/SQL → features → propensity → DML
→ validation → segmentation → falsification → sensitivity → fairness →
policy engine → strategy comparison → checks → save.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd

from .causal import diagnostics, estimation, falsification, sensitivity
from .causal.segmentation import segment_customers, subgroup_stability
from .causal.validation import calibration_stats, gates_table, qini_coefficient
from .checks import run_validation_suite
from .config import DEFAULT_CONFIG, Config
from .data.complexity import (
    check_duplicates,
    check_feature_leakage,
    check_temporal_leakage,
    check_treatment_imbalance,
    inject_and_winsorize_outliers,
    inject_missingness,
    monthly_lift_range,
    validate_schema,
)
from .data.generate import generate_dataset
from .evaluation.explain import global_feature_importance
from .evaluation.fairness import fairness_audit
from .evaluation.stats import power_table, segment_test_sample_size
from .features.build import FeatureBuilder
from .models.registry import ModelRegistry, make_metadata
from .optimization.policy_engine import score_population
from .optimization.strategies import compare_strategies, uplift_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(cfg: Config = DEFAULT_CONFIG, save: bool = True) -> dict:
    rng = np.random.default_rng(cfg.random_state)
    results: dict = {"model_version": cfg.model_version}

    # 1. Generate synthetic data with a known ground truth
    logger.info("Generating %s customers with known ground-truth CATE", f"{cfg.n_customers:,}")
    df, gen = generate_dataset(cfg, rng)
    results.update({k: gen[k] for k in ("n_customers", "true_ate", "naive_difference_in_means", "naive_bias")})
    true_ate = gen["true_ate"]
    naive_ate = gen["naive_difference_in_means"]

    # 2. Data complexity + validation
    logger.info("Injecting realistic data issues and running data validation")
    df, miss = inject_missingness(df, ["ltv", "tenure_months", "sessions_30d"], 0.12, rng)
    df, outl = inject_and_winsorize_outliers(df, "ltv", rng)
    _monthly, monthly_range = monthly_lift_range(df)
    schema_issues = validate_schema(df)
    dupes = check_duplicates(df, subset=["age", "ltv", "cohort_month", "treated"])
    imbalance = check_treatment_imbalance(df)
    temporal = check_temporal_leakage(df)
    results.update({
        "missing_rate_injected": miss["missing_rate_injected"],
        "outliers_winsorized": outl["outliers_winsorized"],
        "monthly_lift_range": monthly_range,
        "schema_issues": schema_issues,
        "duplicate_rows": dupes,
        "treatment_imbalance": imbalance,
        "temporal_leakage_check": temporal,
    })
    if schema_issues:
        logger.warning("Schema issues detected: %s", schema_issues)

    # 3. Feature engineering (persisted for the API)
    logger.info("Building feature matrix")
    feature_builder = FeatureBuilder().fit(df)
    X = feature_builder.transform(df)
    W = df["treated"].values
    Y = df["converted"].values
    leaks = check_feature_leakage(list(X.columns))
    if leaks:
        raise RuntimeError(f"Feature leakage: {leaks}")

    # 4. Design diagnostics (causal identification)
    logger.info("Design diagnostics: covariate balance, propensity, overlap")
    smd_raw, n_imb_raw = diagnostics.covariate_balance(X, W)
    ps = diagnostics.estimate_propensity(X, W, cfg)
    df["propensity"] = ps
    w_ipw = diagnostics.ipw_weights(W, ps)
    smd_adj = pd.Series({c: diagnostics.weighted_smd(X[c].values, W, w_ipw) for c in X.columns})
    n_imb_adj = int((smd_adj.abs() > 0.10).sum())
    violations = diagnostics.positivity_violations(ps)
    results.update({
        "covariates_imbalanced_before": n_imb_raw,
        "covariates_imbalanced_after_ipw": n_imb_adj,
        "max_abs_smd_before": float(smd_raw.abs().max()),
        "max_abs_smd_after": float(smd_adj.abs().max()),
        "positivity_violation_rate": violations / cfg.n_customers,
    })

    # 5. Double ML estimation (the causal centerpiece)
    logger.info("Fitting cross-fitted doubly-robust CATE/ITE model")
    psi, mu1, mu0, e_hat = estimation.fit_dr_scores(X, W, Y, cfg)
    cate_dr, cate_model = estimation.fit_cate(X, psi, cfg)
    df["cate"] = cate_dr
    ate_dr, ate_se, ci = estimation.dr_ate_with_ci(psi)
    boot_lo, boot_hi = estimation.bootstrap_ate_ci(psi, n_boot=300, seed=cfg.random_state)
    cate_t = estimation.t_learner_baseline(X, W, Y, cfg)

    true_cate = df["_true_cate"].values
    corr_dr = float(np.corrcoef(cate_dr, true_cate)[0, 1])
    corr_t = float(np.corrcoef(cate_t, true_cate)[0, 1])
    results.update({
        "ate_dr": ate_dr, "ate_dr_se": ate_se, "ate_dr_ci95": list(ci),
        "ate_dr_bootstrap_ci95": [boot_lo, boot_hi],
        "dr_bias": ate_dr - true_ate, "naive_bias": naive_ate - true_ate,
        "cate_corr_with_truth_dr": corr_dr, "cate_corr_with_truth_tlearner": corr_t,
    })

    # 6. Estimator validation: GATES + Qini
    logger.info("Validating estimator: GATES calibration + Qini ranking")
    gates_df = gates_table(cate_dr, psi, true_cate)
    cal_stats = calibration_stats(gates_df)
    qini, frac, cum_gain, random_gain = qini_coefficient(cate_dr, psi)
    results.update(cal_stats)
    results["qini"] = qini

    # 7. Segmentation + subgroup stability
    logger.info("Segmenting customers by estimated ITE")
    segment, seg_df = segment_customers(df, cate_dr, psi, true_cate)
    df["segment"] = segment
    stability_df = subgroup_stability(df, segment, "region")
    results["segments"] = seg_df.to_dict(orient="records")
    results["segment_stability_by_region"] = stability_df.to_dict(orient="records")

    # 8. Falsification tests
    logger.info("Falsification: placebo treatment + placebo outcome")
    p_treat = falsification.placebo_treatment_test(X, W, Y, cfg, rng)
    p_out = falsification.placebo_outcome_test(X, W, df["pre_period_spend"].values, cfg)
    results.update({
        "placebo_treatment_ate": p_treat["placebo_treatment_ate"],
        "placebo_treatment_t": p_treat["placebo_treatment_t"],
        "placebo_outcome_std_effect": p_out["placebo_outcome_std_effect"],
        "falsification_passed": bool(p_treat["passed"] and p_out["passed"]),
    })

    # 9. Sensitivity analysis
    logger.info("Sensitivity analysis for unmeasured confounding")
    sens_df, sens_break = sensitivity.sensitivity_grid(
        X, df["_true_propensity"].values, df["_baseline_p"].values, true_cate, ate_dr, cfg, rng,
    )
    results["sensitivity_grid"] = sens_df.to_dict(orient="records")
    results["sensitivity_breakpoint"] = sens_break

    # 10. Explainability
    logger.info("Global feature importance for the CATE model")
    fi = global_feature_importance(cate_model, X, psi, n_repeats=3, seed=cfg.random_state)
    results["feature_importance"] = fi.head(15).to_dict(orient="records")

    # 11. Fairness audit
    logger.info("Fairness audit (four-fifths rule on age groups)")
    df["age_group"] = pd.cut(df["age"], [0, 30, 45, 60, 200], labels=["<30", "30-45", "45-60", "60+"])
    would_treat = (df["segment"] == "HIGH").astype(int)
    fair_df, disparity = fairness_audit(df, "age_group", would_treat)
    results["fairness_disparity_ratio"] = disparity
    results["fairness_by_group"] = fair_df.to_dict(orient="records")

    # 12. Policy engine + strategy comparison
    logger.info("Running policy engine and strategy comparison")
    baseline_p = df["_baseline_p"].values
    decisions = score_population(df, cate_dr, baseline_p, cfg, cfg.model_version)
    strat_df, biz = compare_strategies(df, cate_dr, baseline_p, true_cate, cfg, rng)
    upl_df = uplift_curve(cate_dr, psi)
    results["strategy_comparison"] = strat_df.to_dict(orient="records")
    results.update(biz)

    # 13. Power analysis for the confirmatory experiment
    logger.info("Sample-size design for the confirmatory experiment")
    baseline_rate = float(df.loc[df.treated == 0, "converted"].mean())
    results["baseline_conversion"] = baseline_rate
    results["power_analysis"] = power_table(baseline_rate)
    results["segment_test_n_per_arm_bonferroni"] = segment_test_sample_size(baseline_rate)

    # 14. Validation suite
    logger.info("Validation suite")
    ctx = dict(
        n=cfg.n_customers, X=X, ps=ps, violations=violations,
        n_imbalanced_raw=n_imb_raw, n_imbalanced_adj=n_imb_adj,
        ate_dr=ate_dr, true_ate=true_ate, naive_ate=naive_ate, ci=ci, corr_dr=corr_dr,
        cal_slope=cal_stats["calibration_slope"], gates_rank_corr=cal_stats["gates_rank_correlation"],
        qini=qini, placebo_pass=p_treat["passed"], t_p=p_treat["placebo_treatment_t"],
        pre_pass=p_out["passed"], std_effect=p_out["placebo_outcome_std_effect"],
        cate_dr=cate_dr, seg_df=seg_df, best_vs_all=biz["best_strategy_lift_vs_treat_all"],
        disparity=disparity,
    )
    checks = run_validation_suite(ctx)
    n_pass = sum(c.passed for c in checks)
    for c in checks:
        logger.info("  [%s] %s %s", "PASS" if c.passed else "FAIL", c.name, f"({c.detail})" if c.detail else "")
    results["validation_tests"] = [c.as_dict() for c in checks]
    results["tests_passed"] = n_pass
    results["tests_total"] = len(checks)

    # 15. Persist artifacts
    if save:
        os.makedirs(cfg.out_dir, exist_ok=True)
        os.makedirs(cfg.fig_dir, exist_ok=True)
        df_out_cols = ["age", "ltv", "purchase_freq", "tenure_months", "prior_discounts",
                       "category", "region", "cohort_month", "treated", "converted",
                       "propensity", "cate", "segment", "age_group"]
        df[df_out_cols].to_csv(os.path.join(cfg.out_dir, "customer_level_results.csv"), index=False)
        seg_df.to_csv(os.path.join(cfg.out_dir, "segment_summary.csv"), index=False)
        gates_df.to_csv(os.path.join(cfg.out_dir, "gates_calibration.csv"), index=False)
        strat_df.to_csv(os.path.join(cfg.out_dir, "strategy_comparison.csv"), index=False)
        decisions.to_csv(os.path.join(cfg.out_dir, "customer_decisions.csv"), index=False)
        upl_df.to_csv(os.path.join(cfg.out_dir, "uplift_curve.csv"), index=False)
        with open(os.path.join(cfg.out_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)

        registry = ModelRegistry(os.path.join(cfg.out_dir, "registry"))
        meta = make_metadata(
            model_version=cfg.model_version, n_train=cfg.n_customers,
            metrics={"ate_dr": ate_dr, "qini": qini, "cate_corr": corr_dr},
            config={"random_state": cfg.random_state, "n_folds": cfg.n_folds},
        )
        registry.save(cfg.model_version, {"cate_model": cate_model, "feature_builder": feature_builder}, meta)
        logger.info("All artifacts written under %s/", cfg.out_dir)

    return results


if __name__ == "__main__":
    run_pipeline()
