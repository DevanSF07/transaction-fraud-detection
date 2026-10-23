"""Pydantic schemas for REST API request/response contracts."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """Raw transaction payload conforming to standard credit card schema."""
    transaction_id: Optional[str] = Field(default=None, description="Unique transaction ID")
    Time: float = Field(..., description="Seconds elapsed since the first transaction in dataset", ge=0.0)
    Amount: float = Field(..., description="Transaction monetary amount in currency units", ge=0.0)
    V1: float = Field(0.0)
    V2: float = Field(0.0)
    V3: float = Field(0.0)
    V4: float = Field(0.0)
    V5: float = Field(0.0)
    V6: float = Field(0.0)
    V7: float = Field(0.0)
    V8: float = Field(0.0)
    V9: float = Field(0.0)
    V10: float = Field(0.0)
    V11: float = Field(0.0)
    V12: float = Field(0.0)
    V13: float = Field(0.0)
    V14: float = Field(0.0)
    V15: float = Field(0.0)
    V16: float = Field(0.0)
    V17: float = Field(0.0)
    V18: float = Field(0.0)
    V19: float = Field(0.0)
    V20: float = Field(0.0)
    V21: float = Field(0.0)
    V22: float = Field(0.0)
    V23: float = Field(0.0)
    V24: float = Field(0.0)
    V25: float = Field(0.0)
    V26: float = Field(0.0)
    V27: float = Field(0.0)
    V28: float = Field(0.0)


class PredictionResponse(BaseModel):
    """Real-time fraud scoring prediction result."""
    transaction_id: Optional[str] = None
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    is_fraud: bool
    risk_level: str
    decision_threshold: float
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionInput]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int
    frauds_flagged: int


class FeatureAttribution(BaseModel):
    feature: str
    feature_value: float
    shap_value: float
    risk_direction: str


class ExplanationResponse(BaseModel):
    """Interpretability response containing TreeSHAP and LIME diagnostics."""
    transaction_id: Optional[str] = None
    fraud_probability: float
    risk_level: str
    top_shap_factors: List[FeatureAttribution]
    lime_decision_rules: List[Dict[str, Any]]
    analyst_summary: str
