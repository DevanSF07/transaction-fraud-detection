"""Tests for temporal and sequential feature extraction."""

import pytest
import numpy as np
import pandas as pd
from fraud_detection.features.temporal import TemporalFeatureExtractor


def test_temporal_feature_extractor():
    df = pd.DataFrame({
        'Time': [100.0, 250.0, 400.0, 800.0, 1200.0],
        'Amount': [10.0, 50.0, 20.0, 100.0, 30.0]
    })
    extractor = TemporalFeatureExtractor(
        lag_steps=[1, 2],
        ema_alpha=0.3,
        rolling_window=3,
        diurnal_period_hours=24.0,
        use_log_amount=True
    )
    df_trans = extractor.fit_transform(df)

    expected_cols = [
        'Time', 'Amount', 'time_delta', 'amount_lag1', 'amount_lag2',
        'amount_ema', 'amount_ema_diff', 'amount_rolling_mean_3',
        'amount_rolling_std_3', 'sin_time', 'cos_time', 'amount_log'
    ]
    for col in expected_cols:
        assert col in df_trans.columns

    # Verify diurnal trigonometry ranges
    assert (df_trans['sin_time'] >= -1.0).all() and (df_trans['sin_time'] <= 1.0).all()
    assert (df_trans['cos_time'] >= -1.0).all() and (df_trans['cos_time'] <= 1.0).all()
