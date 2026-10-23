"""Comprehensive benchmark runner for Transaction Fraud Detection Engine.

Executes the entire lifecycle end-to-end on real transaction data:
1. Dataset loading (284k+ rows, 0.17% fraud prevalence, real creditcard.csv)
2. MICE imputation validation
3. Temporal feature extraction (rolling lags, EMA, diurnal cyclical)
4. Collinear pruning (correlation threshold 0.85 & VIF)
5. Two-stage hybrid resampling (RUS 1:10 -> SMOTE 1:1 on training fold)
6. Optuna-tuned tree ensemble training (Random Forest + XGBoost soft-voting)
7. Held-out test evaluation (AUC-ROC, PR-AUC, Precision, Recall, Financial Risk)
8. TreeSHAP and LIME risk interpretability verification
9. Exports metrics to reports/benchmark_report.json and figures to artifacts/figures/
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.config import get_default_config
from fraud_detection.data.loader import load_dataset
from fraud_detection.pipeline import FraudDetectionPipeline
from fraud_detection.models.evaluation import save_evaluation_plots

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Benchmark")


def run_benchmark(data_path: str = "data/creditcard.csv", sample_size: int = None, n_trials: int = 5):
    logger.info("=================================================================")
    logger.info("   TRANSACTION FRAUD DETECTION ENGINE - BENCHMARK EXECUTION     ")
    logger.info("=================================================================")

    config = get_default_config()

    # 1. Ingest Data
    logger.info(f"Step 1: Loading transaction records from {data_path}...")
    df = load_dataset(filepath=data_path, sample_size=sample_size, random_state=config.random_state)

    X = df.drop(columns=['Class'])
    y = df['Class']

    logger.info(f"Dataset Shape: {X.shape}, Fraud Count: {y.sum()} ({y.mean() * 100:.3f}%)")
    logger.info(f"Missing Values: {X.isna().sum().sum()}")

    # Stratified Train/Test Split (80% train, 20% held-out test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=config.random_state
    )

    # 2. Build Pipeline
    pipeline = FraudDetectionPipeline(config=config)

    # 3. Fit Pipeline (with Optuna Bayesian search)
    logger.info("Step 2-5: Executing MICE, Temporal Engineering, VIF Pruning, Resampling & Ensemble Training...")
    pipeline.fit(X_train, y_train, optimize=True, n_trials=n_trials)

    # 4. Evaluate Held-out Performance
    logger.info("Step 6: Evaluating on held-out test split...")
    test_metrics = pipeline.evaluate(X_test, y_test)

    # 5. Save Plots
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    plot_paths = save_evaluation_plots(y_test, test_probs, output_dir="artifacts/figures")
    test_metrics['plot_artifacts'] = plot_paths

    # 6. Verify Explainability (TreeSHAP & LIME)
    logger.info("Step 7: Validating TreeSHAP and LIME Explainability...")
    fraud_indices = y_test[y_test == 1].index
    if len(fraud_indices) > 0:
        sample_fraud_idx = fraud_indices[0]
        sample_tx = X_test.loc[sample_fraud_idx]
        explanation = pipeline.explain_transaction(sample_tx, top_n=5)
        logger.info(f"Sample Flagged Fraud Probability: {explanation['fraud_probability']:.4f}")
        if explanation['shap_explanation']['top_attributions']:
            logger.info(f"Top Risk Driver (SHAP): {explanation['shap_explanation']['top_attributions'][0]}")
        if explanation['lime_explanation']['top_rules']:
            logger.info(f"Top Rule (LIME): {explanation['lime_explanation']['top_rules'][0]}")

    # Generate Global SHAP Plot
    try:
        test_features = pipeline.transform_features(X_test)
        pipeline.shap_explainer.generate_summary_plot(
            test_features.head(100),
            output_path="artifacts/figures/shap_summary.png"
        )
    except Exception as e:
        logger.warning(f"Could not generate global SHAP summary plot: {e}")

    # 7. Save Model & Reports
    os.makedirs("reports", exist_ok=True)
    os.makedirs("artifacts/models", exist_ok=True)
    pipeline.save("artifacts/models/fraud_pipeline.joblib")

    report_path = "reports/benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    logger.info(f"Benchmark completed successfully! Report written to: {report_path}")
    logger.info("=================================================================")
    logger.info(f"   AUC-ROC: {test_metrics['auc_roc']} (CV Goal: ~0.93)")
    logger.info(f"   PR-AUC:  {test_metrics['pr_auc']} (CV Goal: ~0.885)")
    logger.info(f"   Recall:  {test_metrics['recall']} ({test_metrics['confusion_matrix']['true_positives']} TP, {test_metrics['confusion_matrix']['false_negatives']} FN)")
    logger.info(f"   Precision Uplift: +{test_metrics['precision_uplift_pct']}%")
    logger.info("=================================================================")
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Fraud Detection Benchmark")
    parser.add_argument("--data-path", type=str, default="data/creditcard.csv", help="Dataset file path")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample size (default: None for full 284k)")
    parser.add_argument("--n-trials", type=int, default=5, help="Optuna search trials")
    args = parser.parse_args()

    run_benchmark(data_path=args.data_path, sample_size=args.sample_size, n_trials=args.n_trials)
