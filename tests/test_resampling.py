"""Tests for two-stage hybrid resampling (RUS + SMOTE)."""

import pytest
import numpy as np
import pandas as pd
from fraud_detection.resampling.hybrid import TwoStageHybridResampler


def test_two_stage_resampling():
    rng = np.random.default_rng(42)
    n_legit = 300
    n_fraud = 10
    total = n_legit + n_fraud

    X = pd.DataFrame(rng.normal(0, 1, size=(total, 5)), columns=[f'col_{i}' for i in range(5)])
    y = pd.Series([0] * n_legit + [1] * n_fraud, name='Class')

    # rus_strategy=0.2 (1:5 ratio), smote_strategy=1.0 (1:1 balance)
    resampler = TwoStageHybridResampler(
        rus_strategy=0.2,
        smote_strategy=1.0,
        smote_k_neighbors=3,
        random_state=42
    )
    X_res, y_res = resampler.fit_resample(X, y)

    res_counts = y_res.value_counts()
    assert res_counts[0] == res_counts[1]  # 1:1 balance after stage 2
