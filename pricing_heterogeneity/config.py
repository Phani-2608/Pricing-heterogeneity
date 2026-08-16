"""Central configuration for the pricing heterogeneity pipeline.

All tunable constants live here so a run can be reproduced or scaled down
(e.g. for fast tests) without touching analysis code.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    random_state: int = 42
    n_customers: int = 12_000
    n_folds: int = 5
    propensity_clip: tuple[float, float] = (0.05, 0.95)

    # Business assumptions, stated explicitly so they can be challenged.
    avg_order_value: float = 100.0
    gross_margin: float = 0.40
    discount_depth: float = 0.15

    out_dir: str = "outputs"

    # Model registry / versioning
    model_version: str = "1.0.0"

    @property
    def fig_dir(self) -> str:
        return f"{self.out_dir}/figures"


DEFAULT_CONFIG = Config()
