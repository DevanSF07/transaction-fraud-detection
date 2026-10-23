"""Configuration management module with dataclasses and YAML loading."""

import os
from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class PathConfig:
    raw_data: str = "data/creditcard.csv"
    synthetic_data: str = "data/synthetic_transactions.csv"
    processed_data: str = "data/processed_transactions.parquet"
    artifacts_dir: str = "artifacts"
    models_dir: str = "artifacts/models"
    figures_dir: str = "artifacts/figures"
    reports_dir: str = "reports"


@dataclass
class DataGenConfig:
    total_samples: int = 284807
    fraud_samples: int = 492
    missing_values_to_inject: int = 45000
    random_seed: int = 42


@dataclass
class ImputationConfig:
    method: str = "mice"
    max_iter: int = 10
    initial_strategy: str = "median"
    n_nearest_features: int = 10
    random_state: int = 42


@dataclass
class TemporalFeatureConfig:
    lag_steps: List[int] = field(default_factory=lambda: [1, 2])
    ema_alpha: float = 0.3
    rolling_window: int = 10
    diurnal_period_hours: float = 24.0
    use_log_amount: bool = True


@dataclass
class FeatureSelectionConfig:
    correlation_threshold: float = 0.85
    vif_threshold: float = 10.0
    mutual_info_top_k: Optional[int] = None


@dataclass
class ResamplingConfig:
    two_stage_hybrid: bool = True
    rus_ratio: float = 0.1
    smote_ratio: float = 1.0
    smote_k_neighbors: int = 5
    random_state: int = 42


@dataclass
class ModelConfig:
    cv_splits: int = 5
    optuna_trials: int = 25
    xgb_weight: float = 0.55
    rf_weight: float = 0.45
    decision_threshold: float = 0.45


@dataclass
class AppConfig:
    pipeline_name: str = "transaction_fraud_detection"
    random_state: int = 42
    paths: PathConfig = field(default_factory=PathConfig)
    data_gen: DataGenConfig = field(default_factory=DataGenConfig)
    imputation: ImputationConfig = field(default_factory=ImputationConfig)
    temporal: TemporalFeatureConfig = field(default_factory=TemporalFeatureConfig)
    selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    resampling: ResamplingConfig = field(default_factory=ResamplingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "AppConfig":
        if not os.path.exists(yaml_path):
            return cls()
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        paths_dict = cfg.get("paths", {})
        data_gen_dict = cfg.get("data_generation", {})
        imputation_dict = cfg.get("imputation", {})
        temporal_dict = cfg.get("feature_engineering", {}).get("temporal", {})
        selection_dict = cfg.get("feature_engineering", {}).get("selection", {})
        resampling_dict = cfg.get("resampling", {})
        modeling_dict = cfg.get("modeling", {})
        ensemble_dict = modeling_dict.get("ensemble", {})

        return cls(
            pipeline_name=cfg.get("pipeline", {}).get("name", "transaction_fraud_detection"),
            random_state=cfg.get("pipeline", {}).get("random_state", 42),
            paths=PathConfig(**paths_dict),
            data_gen=DataGenConfig(**data_gen_dict),
            imputation=ImputationConfig(**imputation_dict),
            temporal=TemporalFeatureConfig(**temporal_dict),
            selection=FeatureSelectionConfig(**selection_dict),
            resampling=ResamplingConfig(**resampling_dict),
            model=ModelConfig(
                cv_splits=modeling_dict.get("cv_splits", 5),
                optuna_trials=modeling_dict.get("optuna", {}).get("n_trials", 25),
                xgb_weight=ensemble_dict.get("default_xgb_weight", 0.55),
                rf_weight=ensemble_dict.get("default_rf_weight", 0.45),
                decision_threshold=ensemble_dict.get("decision_threshold", 0.45),
            ),
        )


def get_default_config(yaml_path: Optional[str] = None) -> AppConfig:
    if yaml_path and os.path.exists(yaml_path):
        return AppConfig.from_yaml(yaml_path)
    default_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "default_config.yaml"
    )
    if os.path.exists(default_path):
        return AppConfig.from_yaml(default_path)
    return AppConfig()
