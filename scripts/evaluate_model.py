"""Model evaluation script on held-out transactions."""

import argparse
import json
import logging
from fraud_detection.pipeline import FraudDetectionPipeline
from fraud_detection.data.loader import load_dataset
from fraud_detection.models.evaluation import save_evaluation_plots

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EvaluateModel")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fraud Detection Pipeline")
    parser.add_argument("--model-path", type=str, default="artifacts/models/fraud_pipeline.joblib")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=10000)
    args = parser.parse_args()

    pipeline = FraudDetectionPipeline.load(args.model_path)
    df = load_dataset(filepath=args.data_path, sample_size=args.sample_size)
    X = df.drop(columns=['Class'])
    y = df['Class']

    metrics = pipeline.evaluate(X, y)
    probs = pipeline.predict_proba(X)[:, 1]
    plots = save_evaluation_plots(y, probs)
    metrics['plots'] = plots

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
