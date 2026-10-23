"""Production FastAPI application for real-time transaction risk scoring."""

import time
import os
import logging
from contextlib import asynccontextmanager
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from fraud_detection.pipeline import FraudDetectionPipeline
from fraud_detection.data.generator import generate_benchmark_dataset
from fraud_detection.serving.schemas import (
    TransactionInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ExplanationResponse,
    FeatureAttribution
)

logger = logging.getLogger(__name__)

# Global pipeline state
pipeline_instance: Optional[FraudDetectionPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to ensure pipeline warmup on startup."""
    global pipeline_instance
    model_path = "artifacts/models/fraud_pipeline.joblib"
    loaded = False
    if os.path.exists(model_path):
        try:
            logger.info(f"Loading existing model from {model_path}...")
            pipeline_instance = FraudDetectionPipeline.load(model_path)
            loaded = True
        except Exception as e:
            logger.warning(f"Failed to load existing checkpoint: {e}. Rebuilding...")

    if not loaded:
        logger.info("Artifact not found; fitting warmup pipeline on real dataset sample...")
        from fraud_detection.data.loader import load_dataset
        df_sample = load_dataset(sample_size=1500, allow_synthetic_fallback=True)
        X = df_sample.drop(columns=['Class'])
        y = df_sample['Class']
        pipeline_instance = FraudDetectionPipeline()
        pipeline_instance.fit(X, y, optimize=False)
        os.makedirs("artifacts/models", exist_ok=True)
        pipeline_instance.save(model_path)
    yield


app = FastAPI(
    title="Transaction Fraud Detection Engine API",
    description="High-Throughput Real-Time Fraud Triage & Explainability Microservice",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Service health and model readiness probe."""
    return {
        "status": "healthy",
        "pipeline_fitted": pipeline_instance is not None and pipeline_instance.is_fitted_,
        "version": "1.0.0"
    }


@app.post("/v1/predict", response_model=PredictionResponse)
def predict_transaction(tx: TransactionInput):
    """Score a single incoming credit card transaction in real time."""
    if pipeline_instance is None or not pipeline_instance.is_fitted_:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    t0 = time.perf_counter()
    data_dict = tx.model_dump(exclude={'transaction_id'})
    df_tx = pd.DataFrame([data_dict])

    probs = pipeline_instance.predict_proba(df_tx)[0]
    p_fraud = float(probs[1])
    is_fraud = bool(p_fraud >= pipeline_instance.model.decision_threshold)
    latency = (time.perf_counter() - t0) * 1000.0

    return PredictionResponse(
        transaction_id=tx.transaction_id,
        fraud_probability=round(p_fraud, 4),
        is_fraud=is_fraud,
        risk_level="HIGH RISK - ALERT" if is_fraud else "LOW RISK - APPROVED",
        decision_threshold=pipeline_instance.model.decision_threshold,
        latency_ms=round(latency, 2)
    )


@app.post("/v1/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """High-throughput micro-batch transaction risk classification."""
    if pipeline_instance is None or not pipeline_instance.is_fitted_:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    t0 = time.perf_counter()
    rows = [tx.model_dump(exclude={'transaction_id'}) for tx in request.transactions]
    df_batch = pd.DataFrame(rows)

    probs = pipeline_instance.predict_proba(df_batch)[:, 1]
    predictions = []
    frauds_count = 0

    threshold = pipeline_instance.model.decision_threshold
    elapsed_per_tx = ((time.perf_counter() - t0) * 1000.0) / max(1, len(rows))

    for tx, p in zip(request.transactions, probs):
        is_fraud = bool(p >= threshold)
        if is_fraud:
            frauds_count += 1
        predictions.append(PredictionResponse(
            transaction_id=tx.transaction_id,
            fraud_probability=round(float(p), 4),
            is_fraud=is_fraud,
            risk_level="HIGH RISK - ALERT" if is_fraud else "LOW RISK - APPROVED",
            decision_threshold=threshold,
            latency_ms=round(elapsed_per_tx, 2)
        ))

    return BatchPredictionResponse(
        predictions=predictions,
        total_processed=len(predictions),
        frauds_flagged=frauds_count
    )


@app.post("/v1/explain", response_model=ExplanationResponse)
def explain_transaction(tx: TransactionInput):
    """Generate TreeSHAP feature attributions and LIME rules for human triage."""
    if pipeline_instance is None or not pipeline_instance.is_fitted_:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    data_dict = tx.model_dump(exclude={'transaction_id'})
    series_tx = pd.Series(data_dict)
    exp_result = pipeline_instance.explain_transaction(series_tx, top_n=5)

    shap_data = exp_result['shap_explanation']['top_attributions']
    top_shap = [
        FeatureAttribution(
            feature=item['feature'],
            feature_value=item['feature_value'],
            shap_value=item['shap_value'],
            risk_direction=item['risk_direction']
        )
        for item in shap_data
    ]

    lime_rules = exp_result['lime_explanation']['top_rules']
    p_fraud = exp_result['fraud_probability']

    summary = (
        f"Transaction flagged with {p_fraud * 100:.1f}% fraud probability. "
        f"Primary driver: {top_shap[0].feature} ({top_shap[0].risk_direction}, SHAP: {top_shap[0].shap_value:+.3f})."
        if top_shap else "No significant risk drivers detected."
    )

    return ExplanationResponse(
        transaction_id=tx.transaction_id,
        fraud_probability=p_fraud,
        risk_level=exp_result['risk_level'],
        top_shap_factors=top_shap,
        lime_decision_rules=lime_rules,
        analyst_summary=summary
    )
