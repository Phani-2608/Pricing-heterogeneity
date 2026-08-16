"""End-to-end validation checks that gate a real run."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""

    def as_dict(self) -> dict:
        return {"test": self.name, "passed": bool(self.passed), "detail": self.detail}


def run_validation_suite(ctx: dict) -> list[CheckResult]:
    results: list[CheckResult] = []

    def check(name: str, condition, detail: str = "") -> None:
        results.append(CheckResult(name, bool(condition), detail))

    n = ctx["n"]
    check("Dataset size >= 10,000", n >= 10_000, f"n={n:,}")
    check("No missing values remain", not ctx["X"].isna().any().any())
    ps = ctx["ps"]
    check("Propensity strictly within (0,1)", (ps > 0).all() and (ps < 1).all(),
          f"[{ps.min():.3f}, {ps.max():.3f}]")
    check("Overlap adequate (<5% violations)", ctx["violations"] / n < 0.05,
          f"{ctx['violations']/n:.2%}")
    check("IPW improves covariate balance", ctx["n_imbalanced_adj"] < ctx["n_imbalanced_raw"],
          f"{ctx['n_imbalanced_raw']} -> {ctx['n_imbalanced_adj']}")
    check("DR less biased than naive",
          abs(ctx["ate_dr"] - ctx["true_ate"]) < abs(ctx["naive_ate"] - ctx["true_ate"]),
          f"{abs(ctx['ate_dr']-ctx['true_ate']):.4f} vs {abs(ctx['naive_ate']-ctx['true_ate']):.4f}")
    ci = ctx["ci"]
    check("True ATE inside 95% CI", ci[0] <= ctx["true_ate"] <= ci[1],
          f"[{ci[0]:+.4f}, {ci[1]:+.4f}]")
    check("CATE correlates with truth (r>0.5)", ctx["corr_dr"] > 0.5, f"r={ctx['corr_dr']:.3f}")
    check("GATES calibration slope in [0.5, 1.5]", 0.5 <= ctx["cal_slope"] <= 1.5,
          f"{ctx['cal_slope']:.3f}")
    check("GATES rank correlation > 0.85", ctx["gates_rank_corr"] > 0.85,
          f"{ctx['gates_rank_corr']:.3f}")
    check("Qini beats random", ctx["qini"] > 0, f"{ctx['qini']:.4f}")
    check("Placebo treatment shows no effect", ctx["placebo_pass"], f"t={ctx['t_p']:+.2f}")
    check("No effect on pre-treatment outcome", ctx["pre_pass"],
          f"{ctx['std_effect']:+.4f} SD")
    check("Sign heterogeneity recovered", ctx["cate_dr"].min() < 0 < ctx["cate_dr"].max())
    check("Segments ordered by effect", ctx["seg_df"]["est_cate"].is_monotonic_increasing)
    check("Targeted policy beats treat-all", ctx["best_vs_all"] > 0,
          f"{ctx['best_vs_all']:+.1%}")
    check("Fairness audit executed", np.isfinite(ctx["disparity"]), f"ratio={ctx['disparity']:.3f}")
    return results
