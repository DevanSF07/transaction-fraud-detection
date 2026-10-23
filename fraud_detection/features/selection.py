"""Multicollinearity analysis and feature pruning module.

Combines Variance Inflation Factor (VIF) and Pearson/Spearman correlation
matrices to prune redundant collinear features, mitigating tree model split
dilution and variance inflation.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)


def calculate_vif(X: pd.DataFrame, max_cols: int = 50) -> pd.DataFrame:
    """Calculate Variance Inflation Factor (VIF) for numeric feature matrix."""
    df_clean = X.select_dtypes(include=[np.number]).dropna()
    if df_clean.shape[1] > max_cols:
        df_clean = df_clean.iloc[:, :max_cols]

    vif_records = []
    values = df_clean.values
    cols = df_clean.columns

    for i, col in enumerate(cols):
        try:
            val = variance_inflation_factor(values, i)
        except Exception:
            val = np.nan
        vif_records.append({'feature': col, 'vif': val})

    vif_df = pd.DataFrame(vif_records).sort_values(by='vif', ascending=False).reset_index(drop=True)
    return vif_df


class CollinearFeatureSelector(BaseEstimator, TransformerMixin):
    """Prunes collinear features based on correlation matrix and VIF analysis."""

    def __init__(
        self,
        correlation_threshold: float = 0.85,
        vif_threshold: float = 10.0,
        enable_vif_prune: bool = False
    ):
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.enable_vif_prune = enable_vif_prune
        self.dropped_features_: List[str] = []
        self.kept_features_: List[str] = []
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CollinearFeatureSelector":
        """Identify redundant collinear features from training correlation matrix."""
        df_num = X.select_dtypes(include=[np.number])
        corr_matrix = df_num.corr().abs()

        # Find pairs exceeding correlation_threshold
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        drop_candidates = set()

        # If target y is provided, retain feature with highest correlation to target
        target_corr = {}
        if y is not None:
            for col in df_num.columns:
                target_corr[col] = abs(float(np.corrcoef(df_num[col].fillna(0), y)[0, 1]))

        for col in upper.columns:
            correlated_with = upper.index[upper[col] > self.correlation_threshold].tolist()
            for other_col in correlated_with:
                if y is not None:
                    # Drop the one with lower correlation to target
                    to_drop = other_col if target_corr.get(col, 0) >= target_corr.get(other_col, 0) else col
                else:
                    to_drop = col
                drop_candidates.add(to_drop)

        # Optional VIF analysis
        if self.enable_vif_prune and len(df_num) > 0:
            remaining_cols = [c for c in df_num.columns if c not in drop_candidates]
            vif_df = calculate_vif(df_num[remaining_cols])
            high_vif = vif_df[vif_df['vif'] > self.vif_threshold]['feature'].tolist()
            for feat in high_vif:
                drop_candidates.add(feat)

        self.dropped_features_ = sorted(list(drop_candidates))
        self.kept_features_ = [c for c in X.columns if c not in self.dropped_features_]
        self.is_fitted_ = True

        logger.info(
            f"CollinearFeatureSelector: identified {len(self.dropped_features_)} redundant features "
            f"to prune (threshold={self.correlation_threshold}): {self.dropped_features_}"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop selected collinear features."""
        if not self.is_fitted_:
            raise RuntimeError("CollinearFeatureSelector must be fitted before transform().")
        cols_to_keep = [c for c in self.kept_features_ if c in X.columns]
        return X[cols_to_keep].copy()

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
