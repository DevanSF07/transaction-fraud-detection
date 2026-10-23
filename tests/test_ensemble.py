"""Tests for Optuna tree ensemble classifier."""

import pytest
import numpy as np
import pandas as pd
from fraud_detection.models.ensemble import FraudEnsembleClassifier


def test_ensemble_fit_and_predict(clean_synthetic_features):
    X, y = clean_synthetic_features
    model = FraudEnsembleClassifier(
        xgb_params={'n_estimators': 10, 'max_depth': 3, 'random_state': 42, 'eval_metric': 'logloss'},
        rf_params={'n_estimators': 10, 'max_depth': 4, 'random_state': 42},
        xgb_weight=0.6,
        decision_threshold=0.45
    )
    model.fit(X, y)
    assert model.is_fitted_

    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})
