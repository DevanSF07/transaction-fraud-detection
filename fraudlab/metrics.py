"""Thresholds selected on validation only; all failures/counts remain visible."""
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, roc_curve, auc, precision_score, recall_score, f1_score)


def select_threshold(y, scores, target_recall=.8):
    if not 0 < target_recall <= 1 or len(np.unique(y)) != 2:
        raise ValueError("Need both classes and 0 < target_recall <= 1")
    p, r, thresholds = precision_recall_curve(y, scores)
    eligible = np.flatnonzero(r[:-1] >= target_recall)
    # Prefer greatest precision, then greatest threshold among equal precisions.
    best = max(eligible, key=lambda i: (p[i], thresholds[i]))
    return float(thresholds[best])


def evaluate(y, scores, threshold):
    y, scores = np.asarray(y), np.asarray(scores)
    if len(np.unique(y)) != 2 or not np.isfinite(scores).all():
        raise ValueError("Evaluation needs both classes and finite scores")
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    p, r, _ = precision_recall_curve(y, scores)
    return {"rows": len(y), "frauds": int(y.sum()), "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "pr_auc_trapezoidal": float(auc(r, p)),
        "threshold": float(threshold), "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)), "alerts": int(pred.sum()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def save_curves(y, scores, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    p, r, _ = precision_recall_curve(y, scores)
    fig, ax = plt.subplots()
    ax.plot(r, p, label=f"AP = {average_precision_score(y, scores):.4f}")
    ax.axhline(np.mean(y), color="gray", linestyle="--", label="Test prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="Untouched chronological test distribution")
    ax.legend(); fig.tight_layout(); fig.savefig(destination / "precision_recall.png", dpi=160); plt.close(fig)
    fpr, tpr, _ = roc_curve(y, scores)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y, scores):.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Chronological test ROC")
    ax.legend(); fig.tight_layout(); fig.savefig(destination / "roc.png", dpi=160); plt.close(fig)
