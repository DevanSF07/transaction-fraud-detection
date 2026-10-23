# Transaction Fraud Detection — real Kaggle data

A complete, reproducible implementation of the CV's methods: chained-equations
imputation, causal temporal features, correlation/VIF pruning, RandomUnderSampler
followed by SMOTE, Optuna-tuned Random Forest/XGBoost soft voting, TreeSHAP and LIME.
**No generated transactions, silent data fallback, or hard-coded performance targets.**

Executed results: [docs/RESULTS.md](docs/RESULTS.md). The default full-data run
measured ROC AUC **0.9854** and average precision **0.7775**. The simpler RF
baseline had higher AP; the complete comparison and limitations are preserved.

## Quick start

Python 3.11–3.13; CPU supported, including macOS. A GPU is not required.

```bash
git clone https://github.com/DevanSF07/transaction-fraud-detection.git
cd transaction-fraud-detection
git switch kaggle-cv-complete
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
fraudlab download --output data/creditcard.csv
fraudlab audit --data data/creditcard.csv --output runs/source_audit.json
pytest -q
# Complete real-data verification, modest CPU search:
fraudlab train --data data/creditcard.csv --config configs/verification.yaml --output runs/verification
# Larger search, still CPU:
fraudlab train --data data/creditcard.csv --config configs/default.yaml --output runs/full
```

The public Kaggle download is attempted directly. If your network/Kaggle requires
authentication, download **creditcard.csv** from the publisher's page yourself and
pass its path. The training command fails if the real file is missing or benchmark
row/class counts do not match. Raw data and trained artifacts are gitignored.
Training also requires the SHA-256 of the verified, unmodified Kaggle CSV:
`76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`.
Re-saving the source CSV changes its bytes; always keep the publisher's original
file and make experimental changes in memory.

## What is implemented

| CV method | Implementation | Qualification |
|---|---|---|
| 284K transactions | Original ULB/Worldline CSV; 284,807 rows and 492 frauds audited | Exact duplicates are removed and reported by default |
| 45K missing values + MICE | Exactly 45,000 observed feature cells masked, then IterativeImputer + BayesianRidge | Deliberate MCAR stress experiment, not naturally missing source data |
| Lags, EMA, rolling features | Prior stream amount lags, prior EMA, rolling mean/std, time gap, relative-day sin/cos | Global stream, **not** individual customer behavior; no card IDs exist |
| VIF and correlation | Training-only constant/correlation pruning and iterative VIF | An ablatable redundancy heuristic, not required for trees |
| Hybrid resampling | Legitimate:fraud reduced to 10:1, then SMOTE to 1:1 | Training folds only; robust scaling before SMOTE |
| RF + XGBoost + Optuna | Joint tree hyperparameters and soft-voting weight, TPE search | Expanding-time CV on training partition; objective is average precision |
| SHAP and LIME | Actual ensemble probability-score explanations | Not causal; latent PCA semantics unknown |
| Performance/precision uplift | Actual held-out metrics and named RF baseline | No attempt to force CV scores; uplift can be negative |

`configs/verification.yaml` uses all real rows, 2 trials, 3 temporal folds, 3
imputation iterations and smaller tree counts to verify the complete path.
`configs/default.yaml` uses 15 trials, 5 temporal folds, 10 imputation iterations.
These are explicit configurations, not claimed optimal hyperparameters.

## Experiment sequence and leakage controls

1. Validate columns, numeric values, binary labels, source SHA-256, duplicates and
   conflicting labels. Retain original zero-based CSV row IDs in split manifests.
2. Sort by elapsed `Time` with original CSV order as the tie-breaker. Split
   approximately 60% train / 20% validation / 20% test, keeping tied timestamps
   together. Fail if any outer partition lacks a class.
3. Allocate and inject MCAR missingness separately in each partition. Never mask
   `Time` or `Class`; record every affected source-row/column pair. Missingness
   affects V1–V28 and Amount. This preserves the real label/feature distributions
   except for the explicitly masked observations.
4. Compute causal features from observations available at each timestamp. Lag,
   rolling and EMA summaries exclude the current amount and never backfill.
   Missing prior amounts are skipped; historical summaries are not recomputed
   retrospectively after imputation. State carries across partitions without labels.
5. Within each expanding-time training fold, fit the imputer, selector, and scaler
   **on that fold's training rows**, then resample training rows only. Tied timestamps
   at fold boundaries are purged from the training side.
6. Cache these fold-local arrays once for the tree-only hyperparameter search.
   No preprocessing is fitted on an entire partition before fold validation.
7. Fit final preprocessing and tuned models on the outer training partition.
   Train a conventional class-weighted RF comparator on the natural distribution
   using the same fitted preprocessing, without resampling.
8. Choose each model's threshold independently on the validation set: maximize
   precision subject to validation recall >= 0.80. Do not refit models on validation
   afterward, and do not use test labels to choose thresholds.
9. Evaluate test data once at its original fraud prevalence. Test recall need not
   reach the validation target. Report both ranking and threshold-dependent metrics.

The temporal features are already causal before CV. They use earlier observed
amounts, which are available to an online stream, never earlier labels.

## Missingness: two valid experiments

The original standard CSV is complete. The CV-style configuration is therefore
**a controlled missing-input robustness experiment on real transactions**.

For a clean-data comparator, copy a configuration and set `missing_cells: 0`,
then run into a new output directory. Do not compare different settings by reading
the test score repeatedly and choosing whichever looks best.

Scikit-learn's IterativeImputer performs one chained-equations imputation here,
not multiple stochastic imputations with pooled uncertainty. Short verification
runs can emit convergence warnings; these are recorded in the report, not hidden.

## Outputs

Each new run directory contains:

* `data_audit.json`: source hash, raw counts, duplicates, actual original missingness.
* `splits.json` and `*_manifest.csv`: counts, prevalence, chronological boundaries,
  and exact source rows used.
* `*_mask.csv`: exact deliberate missing-cell locations.
* `cv_folds.json`, `optuna_trials.json`, `feature_selection.json`: search and fit provenance.
* `metrics.json`: held-out results, thresholds, comparator, measured timing, warnings,
  package versions and limitations.
* `test_predictions.csv`: scores, labels, decisions and source row IDs.
* `model.joblib`: fitted preprocessing, both models, threshold, background and stream state.
* `figures/`: PR and ROC curves.
* `explanations/`: JSON, ensemble TreeSHAP waterfall PNGs and LIME HTML.

Output directories must be empty/new. Interrupted experiments are not silently
overwritten. Pick another directory to rerun.

## Metrics are measurements, not resume targets

`average_precision` is the AP step-function summary of the PR curve.
`pr_auc_trapezoidal` is separately reported; the two are not interchangeable.
Precision uplift is relative to the explicitly named RF comparator, using each
model's validation-selected threshold under the same recall constraint. It is not
uplift over random guesses. The comparator is a conventional baseline, not a claim
that all competing models have been equally tuned.

RF and XGBoost scores trained after resampling are **not calibrated estimates of
real-world fraud probability**. Do not automatically block payments using them.
The CLI returns a score and flag for research triage.

Latency is a warmed batch model-scoring measurement with scope in the report.
It excludes preprocessing, network and service overhead. It is not a sub-5ms
production API SLA or evidence of deployed financial savings.

## Stateful scoring

Input CSV contains exactly `Time,V1,...,V28,Amount`, in nondecreasing Time order.
The default context is the saved history immediately before the held-out test
period. For a new independent stream explicitly use `--start-empty`; its cold-start
features differ from a warmed stream. Persist returned state between batches.

```bash
fraudlab score --model runs/full/model.joblib --input new_transactions.csv \
  --output runs/scored.csv --state-out runs/history.json --start-empty
fraudlab score --model runs/full/model.joblib --input following_transactions.csv \
  --output runs/scored_next.csv --state-in runs/history.json --state-out runs/history_next.json
```

Older-than-history transactions are rejected. Scoring is stateful and not suitable
for arbitrary row shuffling. Load joblib artifacts only from trusted sources.

Explain a chronological input batch using the actual saved ensemble:

```bash
fraudlab explain --model runs/full/model.joblib --input new_transactions.csv \
  --output runs/new_explanations --start-empty --limit 3
```

Explanation is read-only with respect to history; pass the correct `--state-in`
when explaining a continuation rather than an independent stream.

## Explanation fidelity

TreeSHAP is computed for RF and XGBoost in **probability-output space**, with the
same background, then combined using the actual ensemble weights. An additivity
check verifies `expected score + sum(contributions) = ensemble score` within tolerance.
Do not average RF probabilities and XGBoost raw-logit explanations.

LIME invokes the actual weighted ensemble on transformed selected features and
reports local surrogate R². Its perturbations may break temporal relationships.
Both explain model associations; neither establishes why real fraud occurred.

## Limitations and extensions

No card/customer IDs means no personalized velocity features. Elapsed Time does
not establish local clock hour; sin/cos are relative-period encodings. Two days
cannot establish weekly seasonality or long-term drift resilience. PCA feature
names do not reveal original business meanings. Consider class weighting vs
resampling, feature ablations, a separate probability-calibration partition,
recall uncertainty intervals, broader temporal tests and drift monitoring before
claiming operational readiness. See [docs/CV_MAP.md](docs/CV_MAP.md).

## Sources

* Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
* Resampling leakage: https://imbalanced-learn.org/stable/common_pitfalls.html
* Iterative imputation: https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
* SMOTE: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
* Optuna TPE: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html
* TreeSHAP: https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html

Code license: MIT. Dataset terms are those of its publisher; raw data is not redistributed.
