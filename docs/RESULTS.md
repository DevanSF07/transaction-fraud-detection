# Executed real-data runs

Both configurations used the complete original Kaggle CSV with verified SHA-256,
not generated transactions or a random 15K-row shortcut. Results are from CPU runs.

## Source and partitions

* Original: 284,807 transactions, 492 frauds, zero missing cells.
* 1,081 exact duplicate rows removed, including 19 duplicated fraud rows.
* Used: 283,726 transactions, 473 frauds.
* Chronological train: 170,235 rows / 342 frauds.
* Validation: 56,745 rows / 57 frauds.
* Test: 56,746 rows / 74 frauds.
* Deliberate MCAR masks: 27,000 training, 9,000 validation, 9,000 test cells.

The raw data is real. The missingness is explicitly simulated. Each run records
its original-data hash, exact split and missing-cell manifests, environment and configuration.

## Default configuration: 15 trials, 5 expanding-time folds

| Held-out metric | CV-style RF/XGBoost hybrid | Conventional weighted RF |
|---|---:|---:|
| ROC AUC | 0.9854 | 0.9731 |
| Average precision | 0.7775 | 0.7938 |
| Precision at validation-selected threshold | 0.5000 | 0.4800 |
| Recall at that threshold | 0.7973 | 0.8108 |
| True positives | 59 | 60 |
| False positives | 59 | 65 |
| False negatives | 15 | 14 |

Precision uplift is approximately **4.17% relative to this named comparator**,
not 27% and not an improvement in all metrics. Each threshold was selected on
validation with the same minimum validation-recall objective; test recalls differ.
The baseline has higher test average precision. Nothing here establishes that
adding every CV technique improves the simpler baseline.

The smaller verification configuration (2 trials, 3 folds) scored ROC AUC 0.9860
and AP 0.7511. It is retained as an implementation verification, not selected as
a better model based on test scores. No test-driven retuning was performed.

## Validation and cautions

Unit/integration checks and executed smoke checks are listed in
`reports/validation.json`. Weighted TreeSHAP additivity is checked against the
actual ensemble score, and LIME surrogate fidelity is recorded. The stateful CLI
reproduced original test scores across two separately scored batches.

IterativeImputer emitted convergence warnings, even with the default iteration
budget; reports preserve them. Its numerical imputations and feature relations
need evaluation before any operational use. LIME may also emit a third-party
resource warning while saving HTML; it does not change predictions.

These are historical two-day benchmark measurements with 74 held-out frauds.
They are not live deployment results, calibrated payment probabilities, evidence
of financial savings, or reproduction of the user's historical CV scores.
