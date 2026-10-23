"""End-to-end pipeline orchestrator for transaction fraud detection.

Integrates:
1. MICE Multivariate Missing Data Imputation
2. Temporal Feature Engineering (lags, EMA, rolling moments, diurnal time)
3. Collinear & VIF Feature Pruning
4. Two-Stage Hybrid Resampling (RandomUnderSampler + SMOTE) on training split
5. Optuna-Tuned Random Forest + XGBoost Ensemble
6. Post-hoc Explainability via TreeSHAP and LIME
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging

from fraud_detection.config import AppConfig, get_default_config
from fraud_detection.data.imputer import MICEImputer
from fraud_detection.features.temporal import TemporalFeatureExtractor
from fraud_detection.features.selection import CollinearFeatureSelector
from fraud_detection.resampling.hybrid import TwoStageHybridResampler
from fraud_detection.models.ensemble import FraudEnsembleClassifier
from fraud_detection.models.optimization import optimize_hyperparameters
from fraud_detection.models.evaluation import evaluate_predictions
from fraud_detection.explainability.shap_explainer import TransactionSHAPExplainer
from fraud_detection.explainability.lime_explainer import TransactionLIMEExplainer

logger = logging.getLogger(__name__)


class FraudDetectionPipeline:
    """Production-grade end-to-end fraud detection and risk explainability pipeline."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_default_config()

        self.imputer = MICEImputer(
            max_iter=self.config.imputation.max_iter,
            initial_strategy=self.config.imputation.initial_strategy,
            random_state=self.config.random_state
        )
        self.temporal_extractor = TemporalFeatureExtractor(
            lag_steps=self.config.temporal.lag_steps,
            ema_alpha=self.config.temporal.ema_alpha,
            rolling_window=self.config.temporal.rolling_window,
            diurnal_period_hours=self.config.temporal.diurnal_period_hours,
            use_log_amount=self.config.temporal.use_log_amount
        )
        self.selector = CollinearFeatureSelector(
            correlation_threshold=self.config.selection.correlation_threshold,
            vif_threshold=self.config.selection.vif_threshold
        )
        self.resampler = TwoStageHybridResampler(
            rus_strategy=self.config.resampling.rus_ratio,
            smote_strategy=self.config.resampling.smote_ratio,
            smote_k_neighbors=self.config.resampling.smote_k_neighbors,
            random_state=self.config.random_state
        )
        self.model = FraudEnsembleClassifier(
            xgb_weight=self.config.model.xgb_weight,
            decision_threshold=self.config.model.decision_threshold,
            random_state=self.config.random_state
        )

        self.shap_explainer: Optional[TransactionSHAPExplainer] = None
        self.lime_explainer: Optional[TransactionLIMEExplainer] = None
        self.background_sample_: Optional[pd.DataFrame] = None
        self.feature_columns_: List[str] = []
        self.is_fitted_: bool = False

    def transform_features(self, X: pd.DataFrame, is_training: bool = False, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Sequential feature transformation without target leakage."""
        if is_training:
            X_imp = self.imputer.fit_transform(X)
            X_temp = self.temporal_extractor.fit_transform(X_imp)
            X_pruned = self.selector.fit_transform(X_temp, y=y)
            self.feature_columns_ = list(X_pruned.columns)
        else:
            X_imp = self.imputer.transform(X)
            X_temp = self.temporal_extractor.transform(X_imp)
            X_pruned = self.selector.transform(X_temp)

        return X_pruned

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, optimize: bool = False, n_trials: int = 15) -> "FraudDetectionPipeline":
        """Fit entire pipeline: transformations, two-stage resampling, and ensemble training."""
        logger.info(f"Fitting FraudDetectionPipeline on {len(X_train)} records...")

        # Feature transformations on training data
        X_trans = self.transform_features(X_train, is_training=True, y=y_train)

        # Optional Bayesian Hyperparameter Optimization
        if optimize:
            logger.info("Executing Optuna hyperparameter optimization...")
            best_params = optimize_hyperparameters(
                X=X_trans,
                y=y_train,
                n_trials=n_trials,
                cv_splits=self.config.model.cv_splits,
                random_state=self.config.random_state
            )
            self.model.xgb_weight = best_params.get('xgb_weight', self.model.xgb_weight)
            self.model.xgb_params.update({
                'n_estimators': best_params.get('xgb_n_estimators', self.model.xgb_params['n_estimators']),
                'max_depth': best_params.get('xgb_max_depth', self.model.xgb_params['max_depth']),
                'learning_rate': best_params.get('xgb_lr', self.model.xgb_params['learning_rate']),
                'subsample': best_params.get('xgb_subsample', self.model.xgb_params['subsample']),
                'colsample_bytree': best_params.get('xgb_colsample', self.model.xgb_params['colsample_bytree']),
                'scale_pos_weight': best_params.get('xgb_pos_weight', self.model.xgb_params['scale_pos_weight']),
            })
            self.model.rf_params.update({
                'n_estimators': best_params.get('rf_n_estimators', self.model.rf_params['n_estimators']),
                'max_depth': best_params.get('rf_max_depth', self.model.rf_params['max_depth']),
                'min_samples_split': best_params.get('rf_min_split', self.model.rf_params['min_samples_split']),
            })

        # Resample training data strictly inside fit (zero leakage)
        X_res, y_res = self.resampler.fit_resample(X_trans, y_train)

        # Fit tree ensemble
        self.model.fit(X_res, y_res)

        # Save compact background sample for post-hoc explainers
        self.background_sample_ = X_trans.sample(min(150, len(X_trans)), random_state=self.config.random_state)
        self._init_explainers()

        self.is_fitted_ = True
        logger.info("Pipeline fitted successfully.")
        return self

    def _init_explainers(self) -> None:
        """Initialize or restore TreeSHAP and LIME explainers."""
        if self.background_sample_ is not None:
            self.shap_explainer = TransactionSHAPExplainer(
                model=self.model,
                feature_names=self.feature_columns_,
                background_sample=self.background_sample_
            )
            self.lime_explainer = TransactionLIMEExplainer(
                training_data=self.background_sample_,
                feature_names=self.feature_columns_,
                random_state=self.config.random_state
            )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of fraud for given transactions."""
        if not self.is_fitted_:
            raise RuntimeError("Pipeline must be fitted before predict_proba().")
        X_trans = self.transform_features(X, is_training=False)
        return self.model.predict_proba(X_trans)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Classify transactions as legitimate (0) or fraud (1)."""
        if not self.is_fitted_:
            raise RuntimeError("Pipeline must be fitted before predict().")
        X_trans = self.transform_features(X, is_training=False)
        return self.model.predict(X_trans)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate fitted pipeline on held-out test transactions."""
        probs = self.predict_proba(X_test)[:, 1]
        return evaluate_predictions(
            y_true=y_test,
            y_pred_proba=probs,
            threshold=self.model.decision_threshold
        )

    def explain_transaction(self, x: pd.Series, top_n: int = 5) -> Dict[str, Any]:
        """Explain individual transaction risk using both TreeSHAP and LIME."""
        if not self.is_fitted_:
            raise RuntimeError("Pipeline must be fitted before explain_transaction().")

        if self.shap_explainer is None or self.lime_explainer is None:
            self._init_explainers()

        df_single = pd.DataFrame([x])
        x_trans = self.transform_features(df_single, is_training=False).iloc[0]
        prob_fraud = float(self.model.predict_proba(pd.DataFrame([x_trans]))[0, 1])

        shap_exp = self.shap_explainer.explain_transaction(x_trans, top_n=top_n)
        lime_exp = self.lime_explainer.explain_transaction(
            x_trans,
            predict_fn=self.model.predict_proba,
            num_features=top_n
        )

        return {
            'fraud_probability': round(prob_fraud, 4),
            'risk_level': 'HIGH RISK - ALERT' if prob_fraud >= self.model.decision_threshold else 'LOW RISK - APPROVED',
            'decision_threshold': self.model.decision_threshold,
            'shap_explanation': shap_exp,
            'lime_explanation': lime_exp
        }

    def __getstate__(self):
        """Omit unpicklable lambda closures inside LIME discretizer during serialization."""
        state = self.__dict__.copy()
        state['shap_explainer'] = None
        state['lime_explainer'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Explainers will be lazily re-initialized on demand via self._init_explainers()

    def save(self, filepath: str) -> None:
        """Serialize fitted pipeline to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved pipeline to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "FraudDetectionPipeline":
        """Deserialize fitted pipeline from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pipeline checkpoint not found at: {filepath}")
        pipeline = joblib.load(filepath)
        pipeline._init_explainers()
        logger.info(f"Loaded pipeline from {filepath}")
        return pipeline
