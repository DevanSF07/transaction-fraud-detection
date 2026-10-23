"""Ensemble model blending Optuna-tuned Random Forest and XGBoost.

Soft-voting probability combination:
P(fraud | x) = w_xgb * P_xgb(fraud | x) + (1 - w_xgb) * P_rf(fraud | x)

Combines Random Forest's variance reduction on noisy features with XGBoost's
gradient-boosted non-linear interaction learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class FraudEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Ensemble of Random Forest and XGBoost with soft voting and calibrated thresholding."""

    def __init__(
        self,
        xgb_params: Optional[Dict[str, Any]] = None,
        rf_params: Optional[Dict[str, Any]] = None,
        xgb_weight: float = 0.55,
        decision_threshold: float = 0.45,
        random_state: int = 42
    ):
        self.xgb_params = xgb_params or {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 1.0,
            'random_state': random_state,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        self.rf_params = rf_params or {
            'n_estimators': 150,
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': random_state,
            'n_jobs': -1
        }
        self.xgb_weight = xgb_weight
        self.decision_threshold = decision_threshold
        self.random_state = random_state

        self.xgb_model: Optional[XGBClassifier] = None
        self.rf_model: Optional[RandomForestClassifier] = None
        self.feature_names_: List[str] = []
        self.classes_ = np.array([0, 1])
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FraudEnsembleClassifier":
        """Fit both XGBoost and Random Forest on balanced training data."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
            self.feature_names_ = [f"feat_{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y)

        logger.info(f"Training XGBoost classifier on {len(X_arr)} samples...")
        self.xgb_model = XGBClassifier(**self.xgb_params)
        self.xgb_model.fit(X_arr, y_arr)

        logger.info(f"Training Random Forest classifier on {len(X_arr)} samples...")
        self.rf_model = RandomForestClassifier(**self.rf_params)
        self.rf_model.fit(X_arr, y_arr)

        self.is_fitted_ = True
        logger.info(
            f"Fitted FraudEnsemble (XGB weight: {self.xgb_weight:.2f}, "
            f"RF weight: {1.0 - self.xgb_weight:.2f}, threshold: {self.decision_threshold:.2f})."
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate weighted soft-voting fraud probabilities."""
        if not self.is_fitted_:
            raise RuntimeError("FraudEnsembleClassifier is not fitted.")

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        p_xgb = self.xgb_model.predict_proba(X_arr)[:, 1]
        p_rf = self.rf_model.predict_proba(X_arr)[:, 1]

        p_fraud = (self.xgb_weight * p_xgb) + ((1.0 - self.xgb_weight) * p_rf)
        p_legit = 1.0 - p_fraud
        return np.column_stack([p_legit, p_fraud])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Classify transactions according to operational decision threshold."""
        prob_fraud = self.predict_proba(X)[:, 1]
        return (prob_fraud >= self.decision_threshold).astype(int)

    @property
    def feature_importances_(self) -> pd.DataFrame:
        """Combined ensemble feature importance."""
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted.")
        imp_xgb = self.xgb_model.feature_importances_
        imp_rf = self.rf_model.feature_importances_
        blended = (self.xgb_weight * imp_xgb) + ((1.0 - self.xgb_weight) * imp_rf)

        return pd.DataFrame({
            'feature': self.feature_names_,
            'xgb_importance': imp_xgb,
            'rf_importance': imp_rf,
            'blended_importance': blended
        }).sort_values(by='blended_importance', ascending=False).reset_index(drop=True)
