"""Pipeline training script with Optuna Bayesian optimization on real transaction dataset."""

import os
import json
import argparse
import logging
from sklearn.model_selection import train_test_split
from fraud_detection.config import get_default_config
from fraud_detection.data.loader import load_dataset
from fraud_detection.pipeline import FraudDetectionPipeline
from fraud_detection.models.evaluation import save_evaluation_plots

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TrainPipeline")


def main():
    parser = argparse.ArgumentParser(description="Train Fraud Detection Pipeline")
    parser.add_argument("--data-path", type=str, default="data/creditcard.csv", help="Path to credit card dataset")
    parser.add_argument("--sample-size", type=int, default=None, help="Records to sample (default: None for all 284k)")
    parser.add_argument("--n-trials", type=int, default=10, help="Optuna optimization trials")
    parser.add_argument("--output-model", type=str, default="artifacts/models/fraud_pipeline.joblib")
    parser.add_argument("--report-path", type=str, default="reports/training_report.json")
    args = parser.parse_args()

    config = get_default_config()
    df = load_dataset(filepath=args.data_path, sample_size=args.sample_size, random_state=config.random_state)
    X = df.drop(columns=['Class'])
    y = df['Class']

    logger.info(f"Dataset shape: {X.shape}, Frauds: {y.sum()} ({y.mean() * 100:.3f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=config.random_state
    )

    pipeline = FraudDetectionPipeline(config=config)
    pipeline.fit(X_train, y_train, optimize=True, n_trials=args.n_trials)

    metrics = pipeline.evaluate(X_test, y_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    plot_paths = save_evaluation_plots(y_test, probs, output_dir="artifacts/figures")
    metrics['plot_artifacts'] = plot_paths

    logger.info(f"Test Set Evaluation: {metrics}")

    pipeline.save(args.output_model)
    logger.info(f"Pipeline successfully saved to {args.output_model}")

    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    with open(args.report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Training report saved to {args.report_path}")


if __name__ == "__main__":
    main()
