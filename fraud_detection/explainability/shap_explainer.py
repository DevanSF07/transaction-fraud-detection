"""TreeSHAP model interpretability module.

Delivers exact local and global Shapley feature attributions (phi_i) using
TreeSHAP over tree ensemble components, enabling audit compliance and
analyst risk triage.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
import shap
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class TransactionSHAPExplainer:
    """TreeSHAP explainer for global feature impact and local transaction audit trails."""

    def __init__(self, model, feature_names: List[str], background_sample: Optional[pd.DataFrame] = None):
        self.model = model
        self.feature_names = feature_names
        self.background_sample = background_sample

        # Target the XGBoost model within the ensemble for fast exact TreeSHAP
        target_tree = getattr(model, 'xgb_model', model)
        if hasattr(target_tree, 'get_booster'):
            self.explainer = shap.TreeExplainer(target_tree)
        else:
            self.explainer = shap.TreeExplainer(target_tree, data=background_sample)

    def explain_transaction(self, x: pd.Series, top_n: int = 5) -> Dict[str, Any]:
        """Generate local Shapley attributions for a single flagged transaction."""
        x_df = pd.DataFrame([x])[self.feature_names]
        shap_values = self.explainer.shap_values(x_df)

        if isinstance(shap_values, list):
            vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif len(shap_values.shape) == 3:
            vals = shap_values[0, :, 1]
        else:
            vals = shap_values[0]

        expected_value = self.explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            base_val = float(expected_value[1] if len(expected_value) > 1 else expected_value[0])
        else:
            base_val = float(expected_value)

        attributions = []
        for feat, val, x_val in zip(self.feature_names, vals, x_df.iloc[0]):
            attributions.append({
                'feature': feat,
                'feature_value': float(x_val),
                'shap_value': float(val),
                'risk_direction': 'Increases Risk' if val > 0 else 'Decreases Risk'
            })

        # Rank by absolute Shapley contribution
        attributions.sort(key=lambda item: abs(item['shap_value']), reverse=True)

        return {
            'base_value': base_val,
            'prediction_sum': base_val + float(np.sum(vals)),
            'top_attributions': attributions[:top_n],
            'all_attributions': attributions
        }

    def generate_summary_plot(
        self,
        X_sample: pd.DataFrame,
        output_path: str = "artifacts/figures/shap_summary.png"
    ) -> str:
        """Create global TreeSHAP summary beeswarm plot across sample transactions."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        x_clean = X_sample[self.feature_names]
        shap_values = self.explainer.shap_values(x_clean)

        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif len(shap_values.shape) == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, x_clean, feature_names=self.feature_names, show=False)
        plt.title("TreeSHAP Global Feature Impact Summary", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved SHAP summary plot to {output_path}")
        return output_path
