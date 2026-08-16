"""Hypothesis tests, sample-size and MDE calculations for A/B experiment design."""
from __future__ import annotations

import numpy as np
from scipy import stats


def two_proportion_z_test(x1: int, n1: int, x0: int, n0: int) -> tuple[float, float]:
    """Two-sample z-test on proportions. Returns (z_stat, two-sided p-value)."""
    p1, p0 = x1 / n1, x0 / n0
    p_pool = (x1 + x0) / (n1 + n0)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n0))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p0) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p_value)


def sample_size_two_proportions(
    p1: float, p2: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Sample size per arm for a two-proportion z-test."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (
        z_a * np.sqrt(2 * pbar * (1 - pbar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return int(np.ceil(num / (p1 - p2) ** 2))


def minimum_detectable_effect(
    p1: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Smallest absolute lift a fixed-n experiment could reliably detect."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    lo, hi = 1e-5, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        pbar = p1 + mid / 2
        num = (
            z_a * np.sqrt(2 * pbar * (1 - pbar))
            + z_b * np.sqrt(p1 * (1 - p1) + (p1 + mid) * (1 - p1 - mid))
        ) ** 2
        needed = num / mid ** 2
        if needed > n_per_arm:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def power_table(baseline_rate: float, mdes: tuple[float, ...] = (0.010, 0.020, 0.030, 0.050)) -> list[dict]:
    rows = []
    for mde in mdes:
        nper = sample_size_two_proportions(baseline_rate, baseline_rate + mde)
        rows.append({"mde_absolute": mde, "mde_relative": mde / baseline_rate, "n_per_arm": nper, "n_total": 2 * nper})
    return rows


def segment_test_sample_size(
    baseline_rate: float, mde: float = 0.03, n_segments: int = 3, alpha: float = 0.05
) -> int:
    """Bonferroni-corrected per-arm sample size for a segment-level test."""
    return sample_size_two_proportions(baseline_rate, baseline_rate + mde, alpha=alpha / n_segments)
