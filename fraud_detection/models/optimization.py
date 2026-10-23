"""Bayesian Hyperparameter Optimization using Optuna.

Directly optimizes Average Precision (PR-AUC) across Stratified 5-Fold
Cross-Validation with two-stage resampling executed cleanly inside each fold
to ensure zero validation leakage.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import optuna
from fraud_detection.resampling.hybrid import TwoStageHybridResampler

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 25,
    cv_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """Execute Optuna Bayesian optimization targeting PR-AUC under Stratified CV."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    logger.info(f"Starting Optuna search ({n_trials} trials, {cv_splits}-fold Stratified CV)...")

    def objective(trial: optuna.Trial) -> float:
        xgb_params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 80, 250),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 8),
            'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('xgb_subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample', 0.7, 1.0),
            'scale_pos_weight': trial.suggest_float('xgb_pos_weight', 1.0, 4.0),
            'random_state': random_state,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        rf_params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 80, 180),
            'max_depth': trial.suggest_int('rf_max_depth', 6, 14),
            'min_samples_split': trial.suggest_int('rf_min_split', 2, 8),
            'random_state': random_state,
            'n_jobs': -1
        }
        xgb_weight = trial.suggest_float('xgb_weight', 0.35, 0.75)

        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

            # In-fold two-stage resampling to eliminate data leakage
            sampler = TwoStageHybridResampler(rus_strategy=0.1, smote_strategy=1.0, random_state=random_state)
            X_res, y_res = sampler.fit_resample(X_tr, y_tr)

            xgb = XGBClassifier(**xgb_params).fit(X_res.values, y_res.values)
            rf = RandomForestClassifier(**rf_params).fit(X_res.values, y_res.values)

            p_xgb = xgb.predict_proba(X_va.values)[:, 1]
            p_rf = rf.predict_proba(X_va.values)[:, 1]
            p_blend = (xgb_weight * p_xgb) + ((1.0 - xgb_weight) * p_rf)

            score = average_precision_score(y_va, p_blend)
            fold_scores.append(score)

        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"Optuna completed. Best PR-AUC: {study.best_value:.4f}")
    logger.info(f"Best trial parameters: {study.best_params}")
    return study.best_params
