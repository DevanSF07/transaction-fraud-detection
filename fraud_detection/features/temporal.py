"""Temporal feature engineering module for credit card transactions.

Extracts dynamic sequential indicators:
- Rolling lags (t-1, t-2) and inter-transaction time delta
- Exponential Moving Average (EMA, alpha=0.3) and EMA divergence
- 10-transaction rolling mean and volatility (standard deviation)
- Diurnal cyclical encodings (sin_time, cos_time) for 24-hour spending rhythms
- Log1p amount transformation for heavy-tail normalization
"""

import numpy as np
import pandas as pd
from typing import List, Optional
import logging
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Transformer for sequential & temporal signals in transaction streams."""

    def __init__(
        self,
        lag_steps: Optional[List[int]] = None,
        ema_alpha: float = 0.3,
        rolling_window: int = 10,
        diurnal_period_hours: float = 24.0,
        use_log_amount: bool = True
    ):
        self.lag_steps = lag_steps or [1, 2]
        self.ema_alpha = ema_alpha
        self.rolling_window = rolling_window
        self.diurnal_period_hours = diurnal_period_hours
        self.use_log_amount = use_log_amount
        self.amount_median_: float = 22.0
        self.last_seen_amounts_: List[float] = []
        self.last_seen_ema_: Optional[float] = None
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y=None) -> "TemporalFeatureExtractor":
        """Learn baseline statistics from training distribution."""
        if 'Amount' in X.columns:
            self.amount_median_ = float(X['Amount'].median())
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer temporal features preserving row index."""
        df = X.copy()
        if 'Time' not in df.columns or 'Amount' not in df.columns:
            return df

        # Ensure order if multiple rows
        orig_index = df.index
        is_sorted = df['Time'].is_monotonic_increasing
        if not is_sorted and len(df) > 1:
            df = df.sort_values(by='Time')

        # 1. Inter-transaction arrival delta
        time_diff = df['Time'].diff().fillna(0.0)
        df['time_delta'] = np.clip(time_diff, 0.0, 86400.0)

        # 2. Lagged Amounts
        for lag in self.lag_steps:
            df[f'amount_lag{lag}'] = df['Amount'].shift(lag).fillna(self.amount_median_)

        # 3. Exponential Moving Average (EMA)
        df['amount_ema'] = df['Amount'].ewm(alpha=self.ema_alpha, adjust=False).mean()
        df['amount_ema_diff'] = df['Amount'] - df['amount_ema']

        # 4. Rolling Window Moments (window=10)
        rolling = df['Amount'].rolling(window=self.rolling_window, min_periods=1)
        df[f'amount_rolling_mean_{self.rolling_window}'] = rolling.mean()
        df[f'amount_rolling_std_{self.rolling_window}'] = rolling.std().fillna(0.0)

        # 5. Diurnal Cyclical Encodings (circadian 24-hr spending velocity)
        seconds_in_day = self.diurnal_period_hours * 3600.0
        day_time_sec = df['Time'] % seconds_in_day
        angle = 2.0 * np.pi * (day_time_sec / seconds_in_day)
        df['sin_time'] = np.sin(angle)
        df['cos_time'] = np.cos(angle)

        # 6. Log1p Normalized Amount
        if self.use_log_amount:
            df['amount_log'] = np.log1p(np.maximum(df['Amount'], 0.0))

        # Restore original order if it was sorted
        if not is_sorted and len(df) > 1:
            df = df.loc[orig_index]

        return df

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
