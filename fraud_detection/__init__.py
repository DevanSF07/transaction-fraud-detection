"""Transaction Fraud Detection Engine.

End-to-End ML Pipeline featuring MICE Imputation, Temporal Pattern Mining,
Hybrid Resampling (RUS+SMOTE), Optuna Tree Ensembles, and XAI (TreeSHAP & LIME).
"""

__version__ = "1.0.0"
__author__ = "Devan Singh Faujdar"

from fraud_detection.pipeline import FraudDetectionPipeline
from fraud_detection.config import AppConfig, get_default_config

__all__ = ["FraudDetectionPipeline", "AppConfig", "get_default_config"]
