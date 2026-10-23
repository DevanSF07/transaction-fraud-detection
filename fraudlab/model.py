"""Fold-local preprocessing/resampling, Optuna, and the RF/XGBoost mixture."""
from time import perf_counter
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import optuna
from .features import make_preprocessor


def resample(X, y, config):
    y = np.asarray(y)
    counts = np.bincount(y, minlength=2)
    if counts.min() < 2 or counts[1] >= counts[0]:
        raise ValueError("Hybrid training needs >=2 frauds and a legitimate majority")
    original = {"legitimate": int(counts[0]), "fraud": int(counts[1])}
    if counts[1] / counts[0] < config["rus_ratio"]:
        X, y = RandomUnderSampler(sampling_strategy=config["rus_ratio"],
                                 random_state=config["seed"]).fit_resample(X, y)
    after_rus = np.bincount(y, minlength=2)
    k = min(config["smote_neighbors"], int(after_rus[1]) - 1)
    if after_rus[1] / after_rus[0] < config["smote_ratio"]:
        X, y = SMOTE(sampling_strategy=config["smote_ratio"], k_neighbors=k,
                     random_state=config["seed"]).fit_resample(X, y)
    after = np.bincount(y, minlength=2)
    return X, y, {"original": original,
        "after_rus": {"legitimate": int(after_rus[0]), "fraud": int(after_rus[1])},
        "after_smote": {"legitimate": int(after[0]), "fraud": int(after[1])}, "smote_neighbors": k}


def fit_pair(X, y, params, config):
    rf = RandomForestClassifier(n_estimators=config["rf_trees"],
        max_depth=params["rf_depth"], min_samples_leaf=params["rf_leaf"],
        max_features=params["rf_features"], random_state=config["seed"], n_jobs=config["threads"])
    xgb = XGBClassifier(n_estimators=config["xgb_trees"], max_depth=params["xgb_depth"],
        learning_rate=params["learning_rate"], min_child_weight=params["min_child_weight"],
        subsample=params["subsample"], colsample_bytree=params["colsample"],
        reg_lambda=params["reg_lambda"], tree_method="hist", eval_metric="logloss",
        scale_pos_weight=1., base_score=.5, random_state=config["seed"], n_jobs=config["threads"])
    rf.fit(X, y)
    xgb.fit(X, y)
    return rf, xgb


def probabilities(pair, X, xgb_weight):
    return ((1-xgb_weight) * pair[0].predict_proba(X)[:, 1]
            + xgb_weight * pair[1].predict_proba(X)[:, 1])


def prepare_folds(X, y, times, config):
    """Cache independently fitted folds once, never a full-data imputer/selector.

    Features are already causal. Tied timestamps at each fold boundary are purged
    from the fold's training side. Validation remains at natural prevalence.
    """
    folds, metadata = [], []
    for number, (tr, va) in enumerate(TimeSeriesSplit(n_splits=config["cv_folds"]).split(X)):
        tr = tr[np.asarray(times)[tr] < np.asarray(times)[va[0]]]
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2 or sum(y[tr]) < 2:
            raise ValueError(f"Time fold {number} has insufficient fraud cases")
        prep = make_preprocessor(config)
        t = perf_counter()
        a = prep.fit_transform(X.iloc[tr])
        b = prep.transform(X.iloc[va])
        a, target, sampling = resample(a, y[tr], config)
        folds.append((a, target, b, y[va]))
        metadata.append({"fold": number, "train_rows": len(tr), "validation_rows": len(va),
            "train_frauds": int(sum(y[tr])), "validation_frauds": int(sum(y[va])),
            "train_end_time": float(np.asarray(times)[tr[-1]]),
            "validation_start_time": float(np.asarray(times)[va[0]]),
            "preprocessing_and_sampling_seconds": perf_counter()-t, "resampling": sampling})
        print(f"Prepared temporal CV fold {number + 1}/{config['cv_folds']}", flush=True)
    return folds, metadata


def tune(folds, config):
    def objective(trial):
        params = {
            "rf_depth": trial.suggest_int("rf_depth", 5, 16),
            "rf_leaf": trial.suggest_int("rf_leaf", 1, 5),
            "rf_features": trial.suggest_categorical("rf_features", ["sqrt", .7]),
            "xgb_depth": trial.suggest_int("xgb_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", .025, .15, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "subsample": trial.suggest_float("subsample", .7, 1.),
            "colsample": trial.suggest_float("colsample", .6, 1.),
            "reg_lambda": trial.suggest_float("reg_lambda", .5, 8., log=True),
            "xgb_weight": trial.suggest_float("xgb_weight", .2, .8),
        }
        scores = []
        for a, target, b, truth in folds:
            pair = fit_pair(a, target, params, config)
            scores.append(float(average_precision_score(truth, probabilities(pair, b, params["xgb_weight"]))))
        trial.set_user_attr("fold_average_precision", scores)
        return float(np.mean(scores))
    study = optuna.create_study(direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config["seed"],
            n_startup_trials=min(5, max(1, config["optuna_trials"] // 3))))
    study.optimize(objective, n_trials=config["optuna_trials"], n_jobs=1)
    return study.best_params, [{"number": t.number, "value": t.value,
        "params": t.params, "attributes": t.user_attrs} for t in study.trials]
