"""Multivariate Imputation by Chained Equations (MICE) module.

Recovers missing feature attributes using an iterative series of chained
regression models over all available feature dimensions, preserving
multivariate distribution covariance without data leakage.
"""

import numpy as np
import pandas as pd
from typing import Optional, List
import logging
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class MICEImputer(BaseEstimator, TransformerMixin):
    """Production-grade MICE imputer wrapper ensuring strict fit/transform isolation."""

    def __init__(
        self,
        max_iter: int = 10,
        initial_strategy: str = "median",
        n_nearest_features: int = 10,
        random_state: int = 42
    ):
        self.max_iter = max_iter
        self.initial_strategy = initial_strategy
        self.n_nearest_features = n_nearest_features
        self.random_state = random_state
        self.imputer = None
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y=None) -> "MICEImputer":
        """Fit MICE on training feature matrix."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.feature_names_ = list(X.columns)

        missing_count = X.isna().sum().sum()
        logger.info(f"Fitting MICE on {len(X)} records ({missing_count} missing entries)...")

        estimator = BayesianRidge()
        self.imputer = IterativeImputer(
            estimator=estimator,
            max_iter=self.max_iter,
            initial_strategy=self.initial_strategy,
            n_nearest_features=self.n_nearest_features,
            random_state=self.random_state,
            verbose=0,
            skip_complete=True
        )
        self.imputer.fit(X.values)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing attributes using learned chained relationships."""
        if not self.is_fitted_:
            raise RuntimeError("MICEImputer must be fitted before calling transform().")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)

        missing_before = X.isna().sum().sum()
        if missing_before == 0:
            return X.copy()

        logger.info(f"Imputing {missing_before} missing values via MICE...")
        imputed_array = self.imputer.transform(X[self.feature_names_].values)
        df_imputed = pd.DataFrame(imputed_array, columns=self.feature_names_, index=X.index)

        # Validate zero NaNs remain
        assert df_imputed.isna().sum().sum() == 0, "MICE Imputation left unresolved NaNs!"
        return df_imputed

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
