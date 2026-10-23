"""Model evaluation and financial risk metrics module.

Computes:
- AUC-ROC and PR-AUC (Average Precision)
- Precision uplift against baseline
- Financial cost-sensitive triage matrix
- Metric plots (ROC curve, PR curve, confusion matrix)
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def evaluate_predictions(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    threshold: float = 0.45,
    baseline_precision: Optional[float] = None,
    cost_fraud: float = 500.0,
    cost_investigation: float = 15.0
) -> Dict[str, Any]:
    """Compute comprehensive fraud classification metrics."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    auc_roc = float(roc_auc_score(y_true, y_pred_proba))
    pr_auc = float(average_precision_score(y_true, y_pred_proba))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    # Financial Cost Metric
    # False Negatives miss fraud ($500 avg loss); False Positives waste analyst review ($15)
    financial_loss = (fn * cost_fraud) + (fp * cost_investigation)

    # Precision Uplift
    # If baseline precision isn't provided, compare against default uncalibrated or prevalence
    base_prec = baseline_precision if baseline_precision is not None else float(y_true.mean() * 3.5)
    precision_uplift_pct = float(((prec - base_prec) / base_prec) * 100.0) if base_prec > 0 else 0.0

    metrics = {
        'auc_roc': round(auc_roc, 4),
        'pr_auc': round(pr_auc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'decision_threshold': threshold,
        'confusion_matrix': {
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        },
        'precision_uplift_pct': round(precision_uplift_pct, 2),
        'financial_risk_cost': round(financial_loss, 2)
    }

    logger.info(
        f"Evaluation Results -> AUC-ROC: {auc_roc:.4f} | PR-AUC: {pr_auc:.4f} | "
        f"Precision: {prec:.4f} | Recall: {rec:.4f} | Uplift: +{precision_uplift_pct:.1f}%"
    )
    return metrics


def save_evaluation_plots(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    output_dir: str = "artifacts/figures"
) -> Dict[str, str]:
    """Generate and save ROC and PR curves."""
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}

    # 1. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_score = average_precision_score(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall_vals, precision_vals, color='#1f77b4', lw=2, label=f'Ensemble PR Curve (PR-AUC = {pr_score:.3f})')
    ax.axhline(y=y_true.mean(), color='r', linestyle='--', label=f'No Skill Baseline ({y_true.mean():.4f})')
    ax.set_title('Precision-Recall Curve on Held-Out Test Set', fontsize=12, fontweight='bold')
    ax.set_xlabel('Recall (Fraud Detection Rate)')
    ax.set_ylabel('Precision (True Fraud / Flagged Alerts)')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    pr_path = os.path.join(output_dir, "pr_curve.png")
    plt.tight_layout()
    fig.savefig(pr_path, dpi=200)
    plt.close(fig)
    plot_paths['pr_curve'] = pr_path

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_score = roc_auc_score(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color='#2ca02c', lw=2, label=f'Ensemble ROC Curve (AUC = {roc_score:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Chance')
    ax.set_title('ROC Curve on Held-Out Test Set', fontsize=12, fontweight='bold')
    ax.set_xlabel('False Positive Rate (Legit Marked Fraud)')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.tight_layout()
    fig.savefig(roc_path, dpi=200)
    plt.close(fig)
    plot_paths['roc_curve'] = roc_path

    return plot_paths
