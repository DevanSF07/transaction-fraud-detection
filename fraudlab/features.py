"""Causal stream features and training-only numerical preprocessing."""
from dataclasses import dataclass, field
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from .data import INPUTS, validate


@dataclass
class StreamState:
    amounts: list = field(default_factory=list)
    ema: float | None = None
    last_time: float | None = None


def stream_features(frame, state=None, window=10, alpha=.3):
    """Observed prior stream values only. No backfill, labels, card identity, or future.

    Current Amount is known at scoring time but excluded from historical summaries.
    Missing historical amounts are ignored, not retrospectively imputed with future data.
    Tied timestamps use CSV source order; this order is an explicit modeling assumption.
    Does not mutate the supplied state, so callers control stream persistence.
    """
    if window < 2 or not 0 < alpha <= 1:
        raise ValueError("window >= 2 and 0 < alpha <= 1 required")
    x = validate(frame[INPUTS], labeled=False)
    state = StreamState(**vars(state)) if state else StreamState()
    state.amounts = list(state.amounts)
    result = []
    for t, amount in zip(x.Time.to_numpy(), x.Amount.to_numpy()):
        if state.last_time is not None and t < state.last_time:
            raise ValueError("Transactions must be processed in nondecreasing Time order")
        history = np.asarray(state.amounts[-window:], dtype=float)
        observed = history[np.isfinite(history)]
        mean = float(observed.mean()) if len(observed) else np.nan
        std = float(observed.std(ddof=1)) if len(observed) >= 2 else np.nan
        ema = state.ema if state.ema is not None else np.nan
        result.append({
            "stream_time_gap": float(t - state.last_time) if state.last_time is not None else np.nan,
            "stream_amount_lag1": history[-1] if len(history) else np.nan,
            "stream_amount_lag2": history[-2] if len(history) >= 2 else np.nan,
            "stream_mean_prior10": mean, "stream_std_prior10": std,
            "stream_ema_prior": ema, "amount_minus_prior_ema": amount - ema,
            "log_amount": np.log1p(amount),
            "relative_day_sin": np.sin(2 * np.pi * (t % 86400) / 86400),
            "relative_day_cos": np.cos(2 * np.pi * (t % 86400) / 86400),
            "amount_was_missing": float(np.isnan(amount)),
        })
        if np.isfinite(amount):
            state.ema = float(amount) if state.ema is None else alpha * float(amount) + (1-alpha) * state.ema
        state.amounts.append(float(amount))
        state.amounts = state.amounts[-window:]
        state.last_time = float(t)
    engineered = pd.DataFrame(result, index=x.index)
    return pd.concat([x, engineered], axis=1), state


class CollinearitySelector(TransformerMixin, BaseEstimator):
    """Correlation + iterative VIF pruning, fitted only on training inputs.

    Constant features are removed first. Correlation keeps the earlier column;
    VIF repeatedly removes the most redundant column. No outcome-based selection.
    VIF is a redundancy diagnostic here, not a requirement for tree validity.
    """
    def __init__(self, correlation_threshold=.85, vif_threshold=10., sample_size=5000, seed=42):
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.sample_size = sample_size
        self.seed = seed

    @staticmethod
    def vifs(x):
        if x.shape[1] <= 1:
            return np.ones(x.shape[1])
        z = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-12)
        out = []
        for j in range(z.shape[1]):
            others = np.delete(z, j, axis=1)
            coef = np.linalg.lstsq(others, z[:, j], rcond=None)[0]
            residual = z[:, j] - others @ coef
            # Centered target variance is one. Clamp exact dependencies explicitly.
            out.append(float(1 / max(np.mean(residual ** 2), 1e-12)))
        return np.asarray(out)

    def fit(self, X, y=None):
        a = np.asarray(X, dtype=float)
        if not np.isfinite(a).all():
            raise ValueError("Impute before feature selection")
        self.n_features_in_ = a.shape[1]
        rng = np.random.default_rng(self.seed)
        sample = a[rng.choice(len(a), min(len(a), self.sample_size), replace=False)]
        keep = list(np.flatnonzero(a.std(axis=0) > 1e-10))
        self.removals_ = [{"index": int(i), "reason": "constant"}
                          for i in range(a.shape[1]) if i not in keep]
        if not keep:
            raise ValueError("No nonconstant features")
        accepted = []
        for j in keep:
            correlated = any(abs(np.corrcoef(sample[:, j], sample[:, k])[0, 1]) > self.correlation_threshold
                             for k in accepted)
            if correlated:
                self.removals_.append({"index": int(j), "reason": "correlation"})
            else:
                accepted.append(j)
        keep = accepted
        while len(keep) > 1:
            v = self.vifs(sample[:, keep])
            j = int(np.argmax(v))
            if v[j] <= self.vif_threshold:
                break
            self.removals_.append({"index": int(keep[j]), "reason": "VIF", "vif": float(v[j])})
            keep.pop(j)
        self.indices_ = np.asarray(keep, dtype=int)
        self.final_vifs_ = self.vifs(sample[:, keep]).tolist()
        return self

    def transform(self, X):
        a = np.asarray(X, dtype=float)
        if a.shape[1] != self.n_features_in_:
            raise ValueError("Feature count changed")
        return a[:, self.indices_]

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features if input_features is not None else
                           [f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return names[self.indices_]


def make_preprocessor(config):
    # Single chained-equations imputation. This is not pooled multiple imputation.
    return Pipeline([
        ("mice", IterativeImputer(estimator=BayesianRidge(), max_iter=config["imputer_iterations"],
             n_nearest_features=config["imputer_neighbors"], initial_strategy="median",
             skip_complete=True, keep_empty_features=True, random_state=config["seed"])),
        ("selection", CollinearitySelector(config["correlation_threshold"], config["vif_threshold"],
                                         config["selector_sample_size"], config["seed"])),
        # Trees do not require scaling; SMOTE nearest-neighbor geometry does.
        ("scale", RobustScaler()),
    ])
