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
    allow_synthetic_fallback: bool = False,
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """Load transaction records from CSV or parquet, prioritizing real creditcard.csv."""
    target_path = None
    if filepath and os.path.exists(filepath):
        target_path = filepath
    elif filepath is None:
        candidate_paths = [
            "data/creditcard.csv",
            "creditcard.csv",
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "creditcard.csv")
        ]
        target_path = next((p for p in candidate_paths if os.path.exists(p)), None)

    df = None
    if target_path and os.path.exists(target_path):
        logger.info(f"Loading transaction records from {target_path}")
        if target_path.endswith('.parquet'):
            df = pd.read_parquet(target_path)
        else:
            df = pd.read_csv(target_path)
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
        raise FileNotFoundError(f"Transaction data file not found at: {filepath or 'data/creditcard.csv'}")

    # Validate Schema
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    if sample_size and sample_size > 0 and sample_size < len(df):
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
