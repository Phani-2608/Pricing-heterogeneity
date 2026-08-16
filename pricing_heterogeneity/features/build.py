"""Reusable feature engineering pipeline.

Wrapped as a scikit-learn-compatible transformer so the exact same feature
logic runs in training, batch scoring, and the API, with no risk of
train/serve skew from re-implementing it twice.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_COLS = [
    "age", "ltv", "purchase_freq", "tenure_months", "prior_discounts",
    "sessions_30d", "cohort_month",
    "ltv_was_missing", "tenure_months_was_missing", "sessions_30d_was_missing",
]
CATEGORICAL_COLS = ["category", "region"]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Builds the numeric design matrix used by every downstream model.

    Fits the one-hot column vocabulary on training data so unseen categories
    at serve time are handled consistently (dropped, not crashed on).
    """

    def __init__(self, feature_cols: list[str] | None = None, categorical_cols: list[str] | None = None):
        self.feature_cols = feature_cols or FEATURE_COLS
        self.categorical_cols = categorical_cols or CATEGORICAL_COLS
        self.dummy_columns_: list[str] | None = None

    def fit(self, df: pd.DataFrame, y=None) -> FeatureBuilder:
        dummies = pd.get_dummies(df[self.categorical_cols], prefix=self.categorical_cols, drop_first=True)
        self.dummy_columns_ = list(dummies.columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.dummy_columns_ is None:
            raise RuntimeError("FeatureBuilder must be fit before transform.")
        base = df[[c for c in self.feature_cols if c in df.columns]].copy()
        for c in self.feature_cols:
            if c not in base.columns:
                base[c] = 0.0
        dummies = pd.get_dummies(df[self.categorical_cols], prefix=self.categorical_cols, drop_first=True)
        dummies = dummies.reindex(columns=self.dummy_columns_, fill_value=0)
        X = pd.concat([base[self.feature_cols], dummies], axis=1).astype(float)
        return X

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df).transform(df)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience one-shot call for exploratory / notebook use."""
    return FeatureBuilder().fit_transform(df)
