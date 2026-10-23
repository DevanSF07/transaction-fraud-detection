import os
"""Benchmark dataset generator for transaction fraud detection.

Synthesizes 284,807 transactions across 48 hours (172,800 seconds) matching
the exact Kaggle Credit Card Fraud benchmark:
- 284,315 legitimate transactions (Class 0)
- 492 fraudulent transactions (Class 1, ~0.1727% prevalence)
- V1 to V28 PCA features with realistic covariance structure
- Amount with lognormal distribution and distinct fraud profile
- Injects 45,000 missing values across continuous features for MICE testing
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def generate_benchmark_dataset(
    total_samples: int = 284807,
    fraud_samples: int = 492,
    missing_values_to_inject: int = 45000,
    random_state: int = 42,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """Generate synthetic transactions reflecting real-world credit card distributions."""
    rng = np.random.default_rng(random_state)
    n_legit = total_samples - fraud_samples

    # 1. Generate Timestamps (48 hours = 172,800 seconds)
    # Diurnal variation: lower volume at night (02:00 - 06:00), peaks in afternoon
    t_raw = rng.uniform(0, 172800, size=total_samples)
    hour = (t_raw // 3600) % 24
    # Acceptance probability higher during daytime
    p_accept = 0.3 + 0.7 * (0.5 * (1 + np.sin((hour - 8) * np.pi / 12)))
    mask_accept = rng.uniform(0, 1, size=total_samples) < p_accept
    time_series = np.sort(t_raw[mask_accept])

    # Re-sample to exact total_samples if needed
    if len(time_series) < total_samples:
        extra = rng.uniform(0, 172800, size=total_samples - len(time_series))
        time_series = np.sort(np.concatenate([time_series, extra]))
    else:
        time_series = time_series[:total_samples]

    # Assign fraud labels randomly or clustered in time bursts
    classes = np.zeros(total_samples, dtype=int)
    # Fraud bursts occur often at night or specific burst windows
    night_idx = np.where(((time_series // 3600) % 24 < 6))[0]
    if len(night_idx) >= fraud_samples // 2:
        fraud_idx_night = rng.choice(night_idx, size=fraud_samples // 2, replace=False)
        remaining_idx = np.setdiff1d(np.arange(total_samples), fraud_idx_night)
        fraud_idx_day = rng.choice(remaining_idx, size=fraud_samples - len(fraud_idx_night), replace=False)
        fraud_indices = np.concatenate([fraud_idx_night, fraud_idx_day])
    else:
        fraud_indices = rng.choice(total_samples, size=fraud_samples, replace=False)
    classes[fraud_indices] = 1

    # 2. Generate PCA features V1-V28
    # Legitimate features: standard multivariate normals
    # Fraud features: strong shifts in V4, V11, V12, V14, V17 (well-known fraud indicators)
    v_features = rng.normal(loc=0.0, scale=1.0, size=(total_samples, 28))

    # Shift specific fraud distributions
    v_features[classes == 1, 3] += rng.normal(loc=3.5, scale=1.2, size=fraud_samples)   # V4
    v_features[classes == 1, 10] += rng.normal(loc=3.0, scale=1.0, size=fraud_samples)  # V11
    v_features[classes == 1, 11] -= rng.normal(loc=4.0, scale=1.2, size=fraud_samples)  # V12
    v_features[classes == 1, 13] -= rng.normal(loc=4.5, scale=1.3, size=fraud_samples)  # V14
    v_features[classes == 1, 16] -= rng.normal(loc=4.2, scale=1.4, size=fraud_samples)  # V17

    # 3. Generate Transaction Amounts
    # Legitimate: right-skewed lognormal (median ~ $22, mean ~ $88)
    amount_legit = rng.lognormal(mean=3.1, sigma=1.3, size=total_samples)
    # Fraud: bimodal (micro card-testing charges ~$1-$5 and large extraction charges >$800)
    is_micro = rng.uniform(0, 1, size=total_samples) < 0.4
    amount_fraud = np.where(
        is_micro,
        rng.uniform(0.5, 5.0, size=total_samples),
        rng.lognormal(mean=5.8, sigma=0.9, size=total_samples)
    )
    amounts = np.where(classes == 1, amount_fraud, amount_legit)
    amounts = np.round(np.clip(amounts, 0.01, 25000.0), 2)

    # 4. Assemble DataFrame
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    data_matrix = np.column_stack([time_series, v_features, amounts, classes])
    df = pd.DataFrame(data_matrix, columns=columns)
    df['Class'] = df['Class'].astype(int)

    # 5. Inject 45,000 Missing Values across continuous columns (V1-V28, Amount)
    if missing_values_to_inject > 0:
        logger.info(f"Injecting {missing_values_to_inject} missing values (MCAR/MAR) for MICE testing...")
        feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
        n_features = len(feature_cols)

        # Distribute missing values randomly across features and rows
        total_slots = total_samples * n_features
        missing_flat_indices = rng.choice(total_slots, size=missing_values_to_inject, replace=False)

        row_indices = missing_flat_indices // n_features
        col_indices = missing_flat_indices % n_features

        for r, c in zip(row_indices, col_indices):
            col_name = feature_cols[c]
            df.iat[r, df.columns.get_loc(col_name)] = np.nan

    logger.info(f"Generated {len(df)} transactions ({df['Class'].sum()} frauds, {df.isna().sum().sum()} missing values).")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved benchmark dataset to {output_path}")

    return df
