"""Phase 2 training-only nested CV engine, independent of holdout data and UI."""
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from joblib import Memory, parallel_config
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from datasets import make_preprocessor, make_cv
from feature_selection import CachedRankingSelector, RankedSubset

METHODS = ("Baseline", "Correlation", "Mutual Information", "RFE", "RFECV", "LASSO")

@dataclass(frozen=True)
class ExperimentMode:
    name: str
    outer_folds: int
    inner_folds: int
    all_k: bool
    alpha_fractions: tuple[float, ...]
    seed: int = 42
    gap: int = 0

MODES = {
    "FAST": ExperimentMode("FAST", 2, 2, False, (0.01, 0.1)),
    "FULL": ExperimentMode("FULL", 5, 4, True, (0.0001, 0.001, 0.01, 0.1, 1.0)),
}


def make_pipeline(config, method, *, memory=None, seed=42):
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}")
    steps = [("preprocessor", make_preprocessor(config))]
    if method != "Baseline":
        steps += [("ranking", CachedRankingSelector(method="RFE" if method == "RFECV" else method, random_state=seed)),
                  ("selector", RankedSubset())]
    return Pipeline(steps + [("regressor", LinearRegression())], memory=memory)


def feature_grid(config, X_train, mode):
    # Bounds are derived ONLY from this search's training partition. No fitted
    # preprocessing or category mapping is reused in its validation folds.
    width = len(config.numeric_columns) + sum(max(1, X_train[c].nunique(dropna=True)) for c in config.categorical_columns)
    if mode.all_k:
        return list(range(1, width+1))
    return [k for k in (1, 5, 10, 20) if k < width] + ["all"]


def make_search(config, method, X_train, mode, *, memory=None, jobs=1):
    if jobs not in (1, 2):
        raise ValueError("Use jobs=1 or jobs=2; outer loops and BLAS are not parallelized")
    grid = {} if method == "Baseline" else {"selector__k": feature_grid(config, X_train, mode)}
    if method == "LASSO":
        grid = {"ranking__alpha_fraction": list(mode.alpha_fractions)}
    return GridSearchCV(make_pipeline(config, method, memory=memory, seed=mode.seed), grid,
                        scoring="neg_root_mean_squared_error",
                        cv=make_cv(config, mode.inner_folds, gap=mode.gap, random_state=mode.seed),
                        refit=True, n_jobs=jobs, pre_dispatch=jobs, error_score="raise",
                        return_train_score=False)


def feature_details(fitted, config):
    pre = fitted["preprocessor"]
    names = np.asarray(pre.get_feature_names_out(), dtype=object)
    origins = list(config.numeric_columns)
    if config.categorical_columns:
        encoder = pre.named_transformers_["categorical"]["encoder"]
        for column, categories in zip(config.categorical_columns, encoder.categories_):
            origins.extend([column]*len(categories))
    if len(names) != len(origins):
        raise ValueError("Encoded-to-original name mapping does not match preprocessing")
    indices = np.arange(len(names))
    if "ranking" in fitted.named_steps:
        indices = indices[fitted["ranking"].selected_indices_]
        indices = indices[fitted["selector"].selected_indices_]
    coefficients = fitted["regressor"].coef_
    return [{"position": i, "encoded_name": str(names[j]), "original_name": origins[j],
             "coefficient": float(coefficients[i])} for i,j in enumerate(indices)]


def regression_metrics(y, predicted, selected_count):
    y = np.asarray(y, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if not np.isfinite(predicted).all():
        raise ValueError("Model emitted nonfinite predictions")
    mse = float(mean_squared_error(y, predicted))
    r2 = float(r2_score(y, predicted))
    adjusted = 1-(1-r2)*(len(y)-1)/(len(y)-selected_count-1) if len(y)>selected_count+1 else None
    return {"mae": float(mean_absolute_error(y, predicted)), "mse": mse,
            "rmse": float(np.sqrt(mse)), "r2": r2, "adjusted_r2": adjusted}


def _plain(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k:_plain(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(v) for v in value]
    return value


def fit_training(config, X_train, y_train, *, mode=MODES["FAST"], cache_dir=None,
                 jobs=1, progress=None):
    """API deliberately cannot access X_test/y_test. All choices happen here.

    Inner CV tunes each method; each outer fold also selects its method using
    inner scores only, giving a nested estimate of the entire selection rule.
    Final method selection repeats that same inner-CV rule on all training rows.
    """
    if not X_train.index.equals(y_train.index):
        raise ValueError("X_train and y_train must have identical row indices")
    if config.split_strategy == "chronological" and not X_train.index.is_monotonic_increasing:
        raise ValueError("Chronological training rows must retain source order")
    started = perf_counter()
    memory = Memory(location=str(cache_dir) if cache_dir is not None else None, verbose=0)
    outer = make_cv(config, mode.outer_folds, gap=mode.gap, random_state=mode.seed)
    fold_records, workflow_folds = [], []
    method_seconds = {m:0.0 for m in METHODS}
    outer_splits = list(outer.split(X_train, y_train))
    fitted_models, method_records, final_searches = {}, {}, {}
    # At most two loky workers, each with one native thread. Never parallelize
    # outer folds as well; joblib cache is shared only for identical train rows.
    with threadpool_limits(limits=1), parallel_config(backend="loky", inner_max_num_threads=1):
        for fold, (train, validation) in enumerate(outer_splits):
            inner_candidates = []
            for method in METHODS:
                method_started = perf_counter()
                search = make_search(config, method, X_train.iloc[train], mode, memory=memory, jobs=jobs)
                search.fit(X_train.iloc[train], y_train.iloc[train])
                elapsed = perf_counter()-method_started
                method_seconds[method] += elapsed
                fitted = search.best_estimator_
                details = feature_details(fitted, config)
                score = regression_metrics(y_train.iloc[validation], fitted.predict(X_train.iloc[validation]), len(details))
                inner_rmse = float(-search.best_score_)
                record = {"outer_fold": fold, "method": method, "inner_cv_rmse": inner_rmse,
                          "outer_rmse": score["rmse"], "selected_encoded_count": len(details),
                          "best_params": _plain(search.best_params_), "runtime_seconds": elapsed,
                          "train_row_indices": X_train.iloc[train].index.tolist(),
                          "validation_row_indices": X_train.iloc[validation].index.tolist()}
                fold_records.append(record)
                inner_candidates.append(record)
                if progress: progress(f"outer {fold+1}/{mode.outer_folds} {method}: RMSE={score['rmse']:.4f}")
            chosen = min(inner_candidates, key=lambda r: r["inner_cv_rmse"])
            workflow_folds.append({"outer_fold": fold, "chosen_method": chosen["method"],
                                   "rmse": chosen["outer_rmse"], "selection_inner_rmse": chosen["inner_cv_rmse"]})
        for method in METHODS:
            method_started = perf_counter()
            search = make_search(config, method, X_train, mode, memory=memory, jobs=jobs)
            search.fit(X_train, y_train)
            elapsed = perf_counter()-method_started
            method_seconds[method] += elapsed
            fitted = search.best_estimator_
            # Saved predictions must never depend on an absolute training cache path.
            fitted.memory = None
            fitted_models[method] = fitted
            final_searches[method] = search
            details = feature_details(fitted, config)
            encoded_width = len(fitted["preprocessor"].get_feature_names_out())
            outer_scores = [r["outer_rmse"] for r in fold_records if r["method"] == method]
            selected_original = list(dict.fromkeys(d["original_name"] for d in details))
            method_records[method] = {"method": method, "original_input_count": config.original_feature_count,
                                      "encoded_input_count": encoded_width, "selected_encoded_count": len(details),
                                      "selected_original_count": len(selected_original),
                                      "selected_original_names": selected_original,
                                      "selected_encoded_names": [d["encoded_name"] for d in details],
                                      "features": details, "intercept": float(fitted["regressor"].intercept_),
                                      "best_params": _plain(search.best_params_),
                                      "best_cv_rmse": float(-search.best_score_),
                                      "cv_std": float(search.cv_results_["std_test_score"][search.best_index_]),
                                      "nested_cv_rmse": float(np.mean(outer_scores)),
                                      "nested_cv_std": float(np.std(outer_scores)),
                                      "runtime_seconds": method_seconds[method],
                                      "train_metrics": regression_metrics(y_train, fitted.predict(X_train), len(details))}
            if progress: progress(f"final {method}: k={len(details)}, inner CV RMSE={-search.best_score_:.4f}")
    best_method = min(METHODS, key=lambda m: method_records[m]["best_cv_rmse"])
    workflow_scores = [r["rmse"] for r in workflow_folds]
    return {"models": fitted_models, "methods": method_records, "best_method": best_method,
            "searches": final_searches, "outer_folds": fold_records, "workflow_folds": workflow_folds,
            "workflow_nested_cv_rmse": float(np.mean(workflow_scores)),
            "workflow_nested_cv_std": float(np.std(workflow_scores)),
            "mode": asdict(mode), "training_runtime_seconds": perf_counter()-started}
