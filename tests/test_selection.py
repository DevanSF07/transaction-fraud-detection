"""Tests for collinearity pruning and VIF estimation."""

import pytest
import numpy as np
import pandas as pd
from fraud_detection.features.selection import CollinearFeatureSelector, calculate_vif


def test_collinear_pruning():
    rng = np.random.default_rng(42)
    base_x = rng.normal(0, 1, size=100)
    collinear_x = base_x * 0.999 + rng.normal(0, 0.001, size=100)  # r ~ 1.0

    df = pd.DataFrame({
        'feat_a': base_x,
        'feat_b': collinear_x,
        'feat_independent': rng.normal(0, 1, size=100)
    })

    selector = CollinearFeatureSelector(correlation_threshold=0.85)
    df_pruned = selector.fit_transform(df)

    assert len(selector.dropped_features_) == 1
    assert 'feat_independent' in df_pruned.columns
    assert df_pruned.shape[1] == 2
