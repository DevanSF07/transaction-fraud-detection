"""Tests for TreeSHAP and LIME interpretability outputs."""

import pytest
import numpy as np
import pandas as pd
from fraud_detection.models.ensemble import FraudEnsembleClassifier
from fraud_detection.explainability.shap_explainer import TransactionSHAPExplainer
from fraud_detection.explainability.lime_explainer import TransactionLIMEExplainer


def test_explainers_contracts(clean_synthetic_features):
    X, y = clean_synthetic_features
    feature_names = list(X.columns)

    model = FraudEnsembleClassifier(
        xgb_params={'n_estimators': 15, 'max_depth': 3, 'random_state': 42, 'eval_metric': 'logloss'},
        rf_params={'n_estimators': 15, 'max_depth': 4, 'random_state': 42}
    )
    model.fit(X, y)

    # 1. SHAP Test
    shap_exp = TransactionSHAPExplainer(model=model, feature_names=feature_names, background_sample=X.head(30))
    tx_explanation = shap_exp.explain_transaction(X.iloc[0], top_n=5)
    assert 'top_attributions' in tx_explanation
    assert len(tx_explanation['top_attributions']) == 5
    assert 'shap_value' in tx_explanation['top_attributions'][0]

    # 2. LIME Test
    lime_exp = TransactionLIMEExplainer(training_data=X.head(50), feature_names=feature_names, random_state=42)
    lime_result = lime_exp.explain_transaction(X.iloc[0], predict_fn=model.predict_proba, num_features=4)
    assert 'top_rules' in lime_result
    assert len(lime_result['top_rules']) == 4
