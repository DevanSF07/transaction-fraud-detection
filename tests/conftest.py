"""Pytest shared fixtures for transaction fraud test suite."""

import pytest
import numpy as np
import pandas as pd
from fraud_detection.data.generator import generate_benchmark_dataset


@pytest.fixture(scope="session")
def sample_benchmark_df():
    """Generates a compact, fast benchmark slice with 1,000 records."""
    df = generate_benchmark_dataset(
        total_samples=1000,
        fraud_samples=15,
        missing_values_to_inject=50,
        random_state=42
    )
    return df


@pytest.fixture
def clean_synthetic_features():
    """Returns a clean feature matrix and binary target with zero missing values."""
    rng = np.random.default_rng(42)
    n = 200
    cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    X = pd.DataFrame(rng.normal(0, 1, size=(n, len(cols))), columns=cols)
    X['Time'] = np.sort(rng.uniform(0, 86400, size=n))
    X['Amount'] = np.abs(rng.normal(50, 30, size=n)) + 1.0

    y = pd.Series(np.zeros(n, dtype=int), name='Class')
    y.iloc[:10] = 1  # 5% fraud rate
    return X, y
