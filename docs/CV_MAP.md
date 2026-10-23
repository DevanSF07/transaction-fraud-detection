# Interview and CV mapping

## What is implemented

The repository implements all named CV techniques, using the original Kaggle
dataset. It does not assert the historical CV metric values were reproduced.

| Claim | Accurate description of this repository |
|---|---|
| 284K transactions | Original 284,807-row source audited; duplicates removed and split counts reported |
| Imputed 45K values | Controlled masking of 45,000 real feature cells. Source data originally complete |
| MICE | BayesianRidge IterativeImputer; one chained imputation, convergence warnings recorded |
| Rolling lags and EMA | Causal observed transaction-stream features, not per-card history |
| VIF/correlation | Fold-local redundancy pruning; needs ablation because trees tolerate collinearity |
| 0.17% fraud | Original natural prevalence; evaluation not resampled |
| RUS + SMOTE | Training-only 10:1 then 1:1; adaptive neighbor count for small folds |
| Optuna RF/XGBoost | TPE trials over expanding-time CV, joint model settings and voting weight |
| PR-AUC | Report AP and trapezoidal PR area separately |
| Precision uplift | Measured against a named RF baseline, never hard-coded or relative to random guessing |
| SHAP | Both models explained in probability space, weighted and additivity-checked |
| LIME | Actual ensemble callback, local surrogate fidelity recorded |

## Dataset and experiment are different

The real CSV has no natural missingness. Masking is an experiment, not a historical
dataset fact. Missing cells are not missing rows. Do not describe the injected MCAR
mask as observed MAR/MNAR failures. Never mask Class or give Class to the imputer.

## Why temporal splitting?

Train in the past and evaluate later to approximate the deployment question.
Keep the actual rare-class counts visible. A random stratified split answers a
different question and can be added as a clearly labeled benchmark, not mixed with
chronological scores. Randomly sampling rows destroys authentic stream neighborhoods;
verification therefore uses the complete CSV, even with a small search budget.

## Proposed ablations

Pre-register comparisons using validation, not repeated selection on test:

1. Clean data versus controlled MCAR masking.
2. Median imputation versus chained-equations imputation.
3. Original features versus added stream temporal features.
4. No pruning versus correlation/VIF pruning.
5. Class weights versus RUS, SMOTE and the two-stage method.
6. RF versus XGBoost versus soft voting.
7. Separate calibration, using another chronological subset, if probability estimates are needed.

## Follow-up engineering

For per-card features, obtain a data source with stable card identifiers; do not
invent identities for these rows. For production decisions, specify label delay,
review capacity, false-alert cost, monitoring and score calibration. This code is
a research implementation, not evidence of a deployed payment decision service.
