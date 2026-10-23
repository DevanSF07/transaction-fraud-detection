"""Script demonstrating TreeSHAP and LIME transaction risk diagnostics."""
import argparse
import json
import logging
from fraud_detection.pipeline import FraudDetectionPipeline
from fraud_detection.data.loader import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ExplainTransaction")

def main():
    parser = argparse.ArgumentParser(description="Explain Individual Transaction Risk")
    parser.add_argument("--model-path", type=str, default="artifacts/models/fraud_pipeline.joblib")
    args = parser.parse_args()

    pipeline = FraudDetectionPipeline.load(args.model_path)
    df = load_dataset(sample_size=3000)

    fraud_df = df[df["Class"] == 1]
    legit_df = df[df["Class"] == 0]

    if not fraud_df.empty:
        fraud_tx = fraud_df.drop(columns=["Class"]).iloc[0]
        print()
        print("=== EXPLAINING FRAUDULENT TRANSACTION ===")
        exp_fraud = pipeline.explain_transaction(fraud_tx, top_n=5)
        print(f"Predicted Fraud Probability: {exp_fraud['fraud_probability']:.4f}")
        print(f"Operational Risk Level:      {exp_fraud['risk_level']}")
        print()
        print("Top 5 TreeSHAP Risk Contributors:")
        for factor in exp_fraud["shap_explanation"]["top_attributions"]:
            feat = factor["feature"]
            val = factor["feature_value"]
            shp = factor["shap_value"]
            direction = factor["risk_direction"]
            print(f"  * {feat:<20} = {val:>8.2f} -> SHAP: {shp:>+8.4f} ({direction})")
        print()
        print("Top LIME Decision Surrogate Rules:")
        for rule in exp_fraud["lime_explanation"]["top_rules"]:
            r = rule["rule"]
            w = rule["weight"]
            imp = rule["risk_impact"]
            print(f"  * {r:<30} (Weight: {w:>+7.4f}) -> {imp}")

    if not legit_df.empty:
        legit_tx = legit_df.drop(columns=["Class"]).iloc[0]
        print()
        print("=== EXPLAINING LEGITIMATE TRANSACTION ===")
        exp_legit = pipeline.explain_transaction(legit_tx, top_n=3)
        print(f"Predicted Fraud Probability: {exp_legit['fraud_probability']:.4f}")
        print(f"Operational Risk Level:      {exp_legit['risk_level']}")

if __name__ == "__main__":
    main()
