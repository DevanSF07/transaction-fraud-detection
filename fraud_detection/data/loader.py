"""Data loading, caching, and schema validation routines."""

import os
import pandas as pd
from typing import Optional, Tuple
import logging
from fraud_detection.data.generator import generate_benchmark_dataset

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']


def load_dataset(
    filepath: Optional[str] = None,
    allow_synthetic_fallback: bool = True,
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """Load transaction records from CSV or parquet, falling back to synthesis if absent."""
    df = None
    if filepath and os.path.exists(filepath):
        logger.info(f"Loading transaction records from {filepath}")
        if filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath)
    elif allow_synthetic_fallback:
        fallback_path = "data/synthetic_transactions.csv"
        sample_path = "data/sample_transactions.csv"
        if os.path.exists(fallback_path):
            logger.info(f"Loading existing benchmark dataset from {fallback_path}")
            df = pd.read_csv(fallback_path)
        elif os.path.exists(sample_path):
            logger.info(f"Loading sample benchmark dataset from {sample_path}")
            df = pd.read_csv(sample_path)
        else:
            logger.info("Raw dataset not found. Generating standardized benchmark dataset...")
            df = generate_benchmark_dataset(
                output_path=fallback_path,
                random_state=random_state
            )
    else:
        raise FileNotFoundError(f"Transaction data file not found at: {filepath}")

    # Validate Schema
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    if sample_size and sample_size < len(df):
        logger.info(f"Sampling {sample_size} records stratified by Class...")
        fraud_df = df[df['Class'] == 1]
        legit_df = df[df['Class'] == 0]
        ratio = len(fraud_df) / len(df)
        n_fraud_sample = max(1, int(sample_size * ratio))
        n_legit_sample = sample_size - n_fraud_sample

        fraud_sample = fraud_df.sample(n=min(n_fraud_sample, len(fraud_df)), random_state=random_state)
        legit_sample = legit_df.sample(n=n_legit_sample, random_state=random_state)
        df = pd.concat([fraud_sample, legit_sample]).sort_values(by='Time').reset_index(drop=True)

    logger.info(
        f"Loaded {len(df)} transactions. Frauds: {df['Class'].sum()} "
        f"({df['Class'].mean() * 100:.3f}%). Missing cells: {df.isna().sum().sum()}."
    )
    return df
