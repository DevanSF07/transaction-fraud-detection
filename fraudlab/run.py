"""Reproducible experiment: original CSV -> split -> MCAR -> causal features -> ML."""
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
import importlib.metadata
import json
import platform
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits
from .data import load, inject_missing, chronological_parts, INPUTS, write_json, VERIFIED_KAGGLE_SHA256
from .features import stream_features, make_preprocessor
from .model import prepare_folds, tune, resample, fit_pair, probabilities
from .metrics import select_threshold, evaluate, save_curves


def run(data_path, output, config):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory; previous experiments are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    started = perf_counter()
    frame, audit = load(data_path, config["deduplicate"])
    if (audit["raw_rows"] != 284807 or audit["raw_class_counts"] != {"0":284315,"1":492}
            or audit["file_sha256"] != VERIFIED_KAGGLE_SHA256):
        raise ValueError("Training requires the original verified Kaggle CSV: counts and SHA-256 must match. "
                         "Use fraudlab download or the unmodified publisher CSV; do not substitute generated data.")
    write_json(output / "data_audit.json", audit)
    original_parts = chronological_parts(frame, config["train_fraction"], config["validation_fraction"])
    parts, masks = inject_missing(original_parts, config["missing_cells"], config["seed"])
    split_report, features, state = {}, [], None
    train_state = validation_state = None
    for name, original, part, mask in zip(["train", "validation", "test"], original_parts, parts, masks):
        engineered, state = stream_features(part[INPUTS], state, config["rolling_window"], config["ema_alpha"])
        features.append(engineered)
        if name == "train": train_state = state
        if name == "validation": validation_state = state
        mask.to_csv(output / f"{name}_mask.csv", index=False)
        # Exact input/reference rows used are reproducible without redistributing the CSV.
        pd.DataFrame({"source_row": part.source_row, "Time": part.Time, "Class": part.Class}).to_csv(
            output / f"{name}_manifest.csv", index=False)
        split_report[name] = {"rows": len(part), "frauds": int(part.Class.sum()),
            "prevalence": float(part.Class.mean()), "time_min": float(part.Time.min()),
            "time_max": float(part.Time.max()), "injected_cells": len(mask),
            "input_missing_cells_after_mask": int(part[INPUTS].isna().sum().sum())}
    write_json(output / "splits.json", split_report)
    print(json.dumps({"audit":audit,"splits":split_report}), flush=True)
    Xtr, Xva, Xte = features
    ytr, yva, yte = [p.Class.to_numpy() for p in parts]
    stages = {}
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=config["threads"]):
        warnings.simplefilter("always")
        t = perf_counter()
        folds, fold_report = prepare_folds(Xtr, ytr, parts[0].Time.to_numpy(), config)
        stages["fold_preparation"] = perf_counter()-t
        write_json(output / "cv_folds.json", fold_report)
        t = perf_counter(); params, trials = tune(folds, config)
        stages["optuna_search"] = perf_counter()-t
        del folds
        write_json(output / "optuna_trials.json", trials)
        print("Fitting final training-only preprocessing", flush=True)
        t = perf_counter(); prep = make_preprocessor(config)
        a = prep.fit_transform(Xtr); b = prep.transform(Xva); c = prep.transform(Xte)
        stages["final_preprocessing"] = perf_counter()-t
        print("Final preprocessing fitted; training resampled ensemble", flush=True)
        selected = prep.named_steps["selection"]
        names = selected.get_feature_names_out(Xtr.columns).tolist()
        write_json(output / "feature_selection.json", {"input_features": Xtr.columns.tolist(),
            "selected_features": names, "removals": selected.removals_,
            "final_vifs": selected.final_vifs_, "vif_estimated_on_training_rows_only": True})
        t = perf_counter(); balanced, target, sampling = resample(a, ytr, config)
        pair = fit_pair(balanced, target, params, config)
        stages["final_hybrid_fit"] = perf_counter()-t
        print("Training named RF baseline on the natural training distribution", flush=True)
        # Named conventional comparator, same preprocessing but natural training distribution.
        t = perf_counter()
        baseline = RandomForestClassifier(n_estimators=config["rf_trees"], max_depth=10,
            min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=config["threads"], random_state=config["seed"])
        baseline.fit(a, ytr)
        stages["baseline_fit"] = perf_counter()-t
        print("Selecting validation thresholds and evaluating held-out test", flush=True)
        validation_scores = probabilities(pair, b, params["xgb_weight"])
        baseline_validation = baseline.predict_proba(b)[:, 1]
        threshold = select_threshold(yva, validation_scores, config["target_validation_recall"])
        baseline_threshold = select_threshold(yva, baseline_validation, config["target_validation_recall"])
        scores = probabilities(pair, c, params["xgb_weight"])
        baseline_scores = baseline.predict_proba(c)[:, 1]
        test = evaluate(yte, scores, threshold)
        baseline_test = evaluate(yte, baseline_scores, baseline_threshold)
        rng = np.random.default_rng(config["seed"])
        background = a[rng.choice(len(a), min(200, len(a)), replace=False)]
        artifact = {"format_version": 1, "preprocessor": prep, "pair": pair,
            "baseline": baseline, "params": params, "threshold": threshold,
            "baseline_threshold": baseline_threshold, "config": config,
            "feature_columns": Xtr.columns.tolist(), "selected_features": names,
            "background": background, "state_before_test": validation_state,
            "state_after_test": state, "source_sha256": audit["file_sha256"],
            "score_note": "Uncalibrated score. Resampling changes class priors; do not interpret as calibrated payment probability."}
        joblib.dump(artifact, output / "model.joblib", compress=3)
        pd.DataFrame({"source_row": parts[2].source_row, "label": yte, "score": scores,
            "prediction": (scores>=threshold).astype(int), "baseline_score":baseline_scores}).to_csv(
                output / "test_predictions.csv", index=False)
        save_curves(yte, scores, output / "figures")
        t = perf_counter()
        # Warmed batched MODEL scoring, not an online end-to-end latency claim.
        probabilities(pair, c[:100], params["xgb_weight"])
        times=[]
        for _ in range(5):
            q=perf_counter(); probabilities(pair,c,params["xgb_weight"]);times.append(perf_counter()-q)
        latency={"scope":"pretransformed batch model scoring only; excludes preprocessing, I/O and service overhead",
            "batch_rows":len(c), "median_batch_seconds":float(np.median(times)),
            "amortized_ms_per_row":float(np.median(times)*1000/len(c)), "repetitions":5}
        explanations=[]
        if config["explain_samples"]:
            print("Generating and checking ensemble TreeSHAP and LIME explanations", flush=True)
            from .explain import explain_rows
            positives=np.flatnonzero(yte==1); negatives=np.flatnonzero(yte==0)
            chosen=np.r_[positives[:1],negatives[:1],np.arange(len(c))]
            chosen=list(dict.fromkeys(chosen.tolist()))[:config["explain_samples"]]
            explanations=explain_rows(artifact,c[chosen],output/"explanations",config["lime_samples"],config["seed"])
            write_json(output/"explanations"/"source_rows.json",parts[2].source_row.iloc[chosen].tolist())
        stages["plots_latency_explanations"] = perf_counter()-t
        warning_messages = sorted(set(f"{w.category.__name__}: {w.message}" for w in caught))
    base_precision=baseline_test["precision"]
    report = {"created_utc":datetime.now(timezone.utc).isoformat(), "dataset":audit,
        "experiment":"original Kaggle data with controlled MCAR missingness" if config["missing_cells"] else "original complete Kaggle data",
        "injected_missing_cells":config["missing_cells"], "split_policy":"chronological 60/20/20 by default, tied outer timestamps kept together",
        "splits":split_report,"best_params":params, "resampling":sampling,
        "validation":evaluate(yva,validation_scores,threshold),
        "baseline_validation":evaluate(yva,baseline_validation,baseline_threshold),
        "test":test,"baseline_test":baseline_test,
        "precision_uplift_pct_vs_named_baseline":100*(test["precision"]/base_precision-1) if base_precision else None,
        "baseline_definition":"RF without resampling, class_weight=balanced_subsample, same training-only preprocessing; each threshold chosen on validation under the same recall target",
        "threshold_policy":f"maximize validation precision subject to validation recall >= {config['target_validation_recall']}; test recall is not constrained",
        "latency":latency,"stage_seconds":stages,"total_seconds":perf_counter()-started,
        "warnings":warning_messages,
        "environment":{"python":platform.python_version(),"platform":platform.platform(),
            "packages":{p:importlib.metadata.version(p) for p in ["numpy","pandas","scikit-learn","imbalanced-learn","xgboost","optuna","shap","lime"]}},
        "limitations":["Temporal features are global-stream features, not per-customer behavior",
            "Missingness is deliberately injected, not a property of the original dataset",
            "Single chained imputation, not pooled multiple-imputation uncertainty",
            "Scores not calibrated after resampling", "Two-day dataset cannot demonstrate production drift robustness",
            "SHAP/LIME describe model associations; PCA business semantics remain unknown",
            "Historical CV results are not targets or asserted reproduced values"],
        "explanation_rows":len(explanations)}
    write_json(output/"metrics.json",report)
    print(json.dumps({"test":test,"baseline_test":baseline_test,"output":str(output),"seconds":report["total_seconds"]}),flush=True)
    return report
