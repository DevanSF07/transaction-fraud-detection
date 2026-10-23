"""LIME (Local Interpretable Model-agnostic Explanations) module.

Generates sparse, perturbation-based linear surrogate explanations for
individual transactions, providing plain-English rule boundaries for fraud
analyst investigation.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
from lime.lime_tabular import LimeTabularExplainer

logger = logging.getLogger(__name__)


class TransactionLIMEExplainer:
    """LIME tabular explainer for perturbation-based risk diagnostics."""

    def __init__(
        self,
        training_data: pd.DataFrame,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        random_state: int = 42
    ):
        self.feature_names = feature_names
        self.class_names = class_names or ["Legitimate", "Fraud"]
        self.training_array = training_data[feature_names].values

        self.explainer = LimeTabularExplainer(
            training_data=self.training_array,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode="classification",
            discretize_continuous=True,
            random_state=random_state
        )

    def explain_transaction(
        self,
        x: pd.Series,
        predict_fn,
        num_features: int = 6
    ) -> Dict[str, Any]:
        """Explain single transaction probability via local surrogate."""
        x_clean = x[self.feature_names].values
        exp = self.explainer.explain_instance(
            data_row=x_clean,
            predict_fn=predict_fn,
            num_features=num_features,
            labels=(1,)
        )

        rule_tuples = exp.as_list(label=1)
        parsed_rules = []
        for rule, weight in rule_tuples:
            parsed_rules.append({
                'rule': rule,
                'weight': round(float(weight), 4),
                'risk_impact': 'Elevates Fraud Risk' if weight > 0 else 'Reduces Fraud Risk'
            })

        return {
            'fraud_score_surrogate': float(exp.local_pred[0] if hasattr(exp, 'local_pred') else 0.0),
            'top_rules': parsed_rules,
            'intercept': float(exp.intercept[1]) if hasattr(exp, 'intercept') else 0.0
        }
