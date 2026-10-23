"""Explain the actual weighted probability score, not one constituent model."""
from pathlib import Path
import numpy as np
from .model import probabilities
from .data import write_json


def tree_shap(pair, X, background, weight):
    import shap
    contributions, bases = [], []
    for model in pair:
        explainer = shap.TreeExplainer(model, data=background,
            feature_perturbation="interventional", model_output="probability")
        values = np.asarray(explainer.shap_values(X, check_additivity=False))
        base = np.asarray(explainer.expected_value)
        if values.ndim == 3:  # RF: rows, features, classes
            values, base = values[:, :, 1], base[1]
        contributions.append(values)
        bases.append(float(base))
    combined = (1-weight) * contributions[0] + weight * contributions[1]
    base = (1-weight) * bases[0] + weight * bases[1]
    actual = probabilities(pair, X, weight)
    residual = np.abs(base + combined.sum(axis=1) - actual)
    if residual.max() > 1e-4:
        raise RuntimeError(f"Ensemble SHAP additivity failed: {residual.max()}")
    return combined, base, residual


def explain_rows(artifact, X, destination, lime_samples=1500, seed=42):
    from lime.lime_tabular import LimeTabularExplainer
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    pair, weight = artifact["pair"], artifact["params"]["xgb_weight"]
    background, names = artifact["background"], artifact["selected_features"]
    X = np.asarray(X)
    values, base, residual = tree_shap(pair, X, background, weight)
    lime = LimeTabularExplainer(background, feature_names=names,
        class_names=["legitimate", "fraud"], discretize_continuous=True,
        random_state=seed, mode="classification")
    def predict(a):
        p = probabilities(pair, a, weight)
        return np.column_stack([1-p, p])
    output = []
    for i, row in enumerate(X):
        local = lime.explain_instance(row, predict, labels=(1,),
            num_features=min(8, len(names)), num_samples=lime_samples)
        order = np.argsort(-np.abs(values[i]))[:8]
        entry = {"row_in_explanation_batch": i, "ensemble_score": float(predict(row[None])[0, 1]),
            "shap_expected_score": base, "shap_additivity_error": float(residual[i]),
            "shap_all_contributions": {n: float(v) for n, v in zip(names, values[i])},
            "top_shap": [{"feature": names[j], "transformed_value": float(row[j]),
                          "contribution_to_score": float(values[i, j])} for j in order],
            "lime_rules": [{"rule": r, "weight": float(w)} for r, w in local.as_list(label=1)],
            "lime_local_surrogate_r2": float(local.score),
            "interpretation": "Associations with this model score, not causal explanations; "
                "inputs are selected, imputed, robust-scaled features. PCA meanings are unknown. "
                "LIME perturbations may violate transaction/temporal constraints."}
        output.append(entry)
        shap.plots.waterfall(shap.Explanation(values=values[i], base_values=base,
            data=row, feature_names=names), max_display=12, show=False)
        plt.tight_layout(); plt.savefig(destination / f"shap_waterfall_{i}.png", dpi=150, bbox_inches="tight"); plt.close("all")
        local.save_to_file(str(destination / f"lime_{i}.html"))
    write_json(destination / "explanations.json", output)
    return output
