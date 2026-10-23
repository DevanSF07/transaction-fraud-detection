"""Tests for dataset synthesis and loading routines."""

import pytest
import pandas as pd
import numpy as np
from fraud_detection.data.generator import generate_benchmark_dataset
from fraud_detection.data.loader import load_dataset, EXPECTED_COLUMNS


def test_generator_structure():
    df = generate_benchmark_dataset(
        total_samples=500,
        fraud_samples=10,
        missing_values_to_inject=20,
        random_state=123
    )
    assert len(df) == 500
    assert df['Class'].sum() == 10
    assert set(EXPECTED_COLUMNS).issubset(set(df.columns))
    assert df.isna().sum().sum() == 20
    assert (df['Time'].values == np.sort(df['Time'].values)).all()


def test_loader_sample_stratification():
    df = load_dataset(sample_size=300, random_state=42)
    assert len(df) == 300
    assert df['Class'].sum() > 0  # Guarantees stratified sampling includes frauds
