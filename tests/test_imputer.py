"""Tests for MICE missing value imputation."""

import pytest
import numpy as np
import pandas as pd
from fraud_detection.data.imputer import MICEImputer


def test_mice_imputer_zero_nans():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(0, 1, size=(100, 5)), columns=[f'feat_{i}' for i in range(5)])

    # Inject 15 NaNs
    df.iloc[0, 1] = np.nan
    df.iloc[5, 2] = np.nan
    df.iloc[10, 3] = np.nan
    df.iloc[20, 0] = np.nan
    assert df.isna().sum().sum() == 4

    imputer = MICEImputer(max_iter=3, random_state=42)
    imputed_df = imputer.fit_transform(df)

    assert imputed_df.isna().sum().sum() == 0
    assert imputed_df.shape == df.shape
    assert list(imputed_df.columns) == list(df.columns)
