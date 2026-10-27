"""Phase 2 CLI: backend-only FAST/FULL experiments, no dashboard writes."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
import sklearn
from datasets import OFFICIAL_DATASETS, load_dataset, holdout_indices
from regression_pipeline import MODES, METHODS, fit_training, regression_metrics

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")


def evaluate_holdout(training, X_test, y_test):
    """Evaluation after method/parameter selection; cannot change the decision."""
    return {method: regression_metrics(y_test, model.predict(X_test),
                                       training["methods"][method]["selected_encoded_count"],
                                       fitted_rank=training["methods"][method].get("fitted_design_rank"))
            for method, model in training["models"].items()}


def cache_namespace():
    """Invalidate cached fits when local estimator code or package versions change."""
    digest = hashlib.sha256()
    for name in ("feature_selection.py", "regression_pipeline.py", "phase2_experiment.py",
                 "datasets/config.py", "datasets/preprocessing.py"):
        digest.update((ROOT/name).read_bytes())
    digest.update((sklearn.__version__+np.__version__+joblib.__version__).encode())
    return digest.hexdigest()[:16]


def run_dataset(config, *, mode=MODES["FAST"], output_root=None, jobs=2, use_cache=True, progress=True):
    started = perf_counter()
    data = load_dataset(config)
    train, test = holdout_indices(data, test_size=0.2, random_state=mode.seed, gap=mode.gap)
    X_train, X_test = data.X.iloc[train], data.X.iloc[test]
    y_train, y_test = data.y.iloc[train], data.y.iloc[test]
    output = Path(output_root or ROOT/"results") / config.dataset_id / mode.name
    output.mkdir(parents=True, exist_ok=True)
    namespace = cache_namespace()
    cache_dir = ROOT/".cache/phase2"/config.dataset_id/mode.name/namespace if use_cache else None
    namespace_preexisting = bool(cache_dir is not None and cache_dir.exists())
    logger = (lambda message: print(f"[{config.dataset_id} {mode.name}] {message}", flush=True)) if progress else None
    training = fit_training(config, X_train, y_train, mode=mode, cache_dir=cache_dir, jobs=jobs, progress=logger)
    # The selection decision is already immutable before test data is passed here.
    chosen_method = training["best_method"]
    test_metrics = evaluate_holdout(training, X_test, y_test)
    records = []
    for method in METHODS:
        row = training["methods"][method]
        row["test_metrics"] = test_metrics[method]
        records.append(row)
    ordered = sorted(records, key=lambda record: record["best_cv_rmse"])
    best_info = {"method": chosen_method, "selection_criterion": "minimum final training-only inner CV RMSE",
                 "selection_score": training["methods"][chosen_method]["best_cv_rmse"],
                 "params": training["methods"][chosen_method]["best_params"],
                 "model_path": "best_model.joblib"}
    report = {"schema_version": 2, "phase": 2, "dataset_id": config.dataset_id,
              "display_name": config.display_name, "mode": mode.name,
              "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "target": config.target, "target_unit": config.target_unit,
              "source_sha256": hashlib.sha256(config.path.read_bytes()).hexdigest(),
              "original_input_count": config.original_feature_count,
              "original_feature_names": list(config.feature_names),
              "train_size": len(train), "test_size": len(test), "split_strategy": config.split_strategy,
              "train_row_indices": train.tolist(), "test_row_indices": test.tolist(),
              "configuration": training["mode"], "jobs": jobs, "native_threads_per_worker": 1,
              "selection_cv_note": "Best inner CV scores are tuning scores; compare nested and held-out scores separately.",
              "nested_workflow_note": "Each outer fold selects method and hyperparameters by inner CV on its training portion only.",
              "workflow_nested_cv_rmse": training["workflow_nested_cv_rmse"],
              "workflow_nested_cv_std": training["workflow_nested_cv_std"],
              "best_model": best_info, "methods": ordered,
              "training_runtime_seconds": training["training_runtime_seconds"],
              "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                           "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__}}
    if not data.temporal_metadata.empty:
        dates = data.temporal_metadata["date"]
        report["time_ranges"] = {"train_start": str(dates.iloc[train[0]]), "train_end": str(dates.iloc[train[-1]]),
                                 "test_start": str(dates.iloc[test[0]]), "test_end": str(dates.iloc[test[-1]])}
    table, coefficients = [], []
    for rank, row in enumerate(ordered, 1):
        table.append({"Rank": rank, "Method": row["method"], "Original Inputs": row["original_input_count"],
                      "Encoded Inputs": row["encoded_input_count"], "Selected Encoded": row["selected_encoded_count"],
                      "Selected Original": row["selected_original_count"],
                      "Best CV RMSE": row["best_cv_rmse"], "CV Std": row["cv_std"],
                      "Nested CV RMSE": row["nested_cv_rmse"], "Nested CV Std": row["nested_cv_std"],
                      "Runtime Seconds": row["runtime_seconds"],
                      **{f"Train {k.upper()}": v for k,v in row["train_metrics"].items()},
                      **{f"Test {k.upper()}": v for k,v in row["test_metrics"].items()}})
        coefficients.append({"method": row["method"], "position": -1, "encoded_name": "Intercept",
                             "original_name": None, "coefficient": row["intercept"]})
        coefficients.extend({"method": row["method"], **f} for f in row["features"])
        model_path = output/"models"/(row["method"].lower().replace(" ", "_")+".joblib")
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(training["models"][row["method"]], model_path, compress=3)
    joblib.dump(training["models"][chosen_method], output/"best_model.joblib", compress=3)
    pd.DataFrame(table).to_csv(output/"model_comparison.csv", index=False)
    pd.DataFrame(coefficients).to_csv(output/"coefficients.csv", index=False)
    write_json(output/"selected_features.json", {r["method"]: {"encoded": r["selected_encoded_names"],
               "original": r["selected_original_names"], "features": r["features"]} for r in records})
    for part, indices, X, y in (("train", train, X_train, y_train), ("test", test, X_test, y_test)):
        frame = pd.DataFrame({"source_row_index": data.X.index[indices], "actual": y.to_numpy()})
        if not data.temporal_metadata.empty: frame.insert(1, "date", data.temporal_metadata.iloc[indices].date.to_numpy())
        for method in METHODS: frame[method] = training["models"][method].predict(X)
        frame.to_csv(output/(part+"_predictions.csv"), index=False)
    searches = output/"cv_searches"
    searches.mkdir(exist_ok=True)
    for method, search in training["searches"].items():
        pd.DataFrame(search.cv_results_).to_csv(searches/(method.lower().replace(" ", "_")+".csv"), index=False)
    write_json(output/"outer_folds.json", {"method_folds": training["outer_folds"], "workflow_folds": training["workflow_folds"]})
    write_json(output/"best_model.json", best_info)
    report["runtime_seconds"] = perf_counter()-started
    report["runtime_note"] = "Load, nested/final CV, evaluation, model and CSV export; excludes final JSON serialization. Includes cache hits if present."
    report["cache_enabled"] = use_cache
    report["cache_namespace"] = namespace if use_cache else None
    report["cache_namespace_preexisting"] = namespace_preexisting
    write_json(output/"run.json", report)
    # Structured backend payload for the later dashboard phase. Never writes UI assets.
    write_json(output/"dashboard_payload.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *OFFICIAL_DATASETS], default="all")
    parser.add_argument("--mode", choices=list(MODES), default="FAST", type=str.upper)
    parser.add_argument("--jobs", type=int, choices=[1,2], default=2)
    parser.add_argument("--gap", type=int, default=0, help="Energy gap in rows, applied to holdout and both CV levels")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    if args.gap < 0: parser.error("gap must be nonnegative")
    total_started = perf_counter()
    reports = []
    for dataset_id, config in OFFICIAL_DATASETS.items():
        if args.dataset not in ("all", dataset_id): continue
        mode = replace(MODES[args.mode], gap=args.gap if config.split_strategy == "chronological" else 0)
        reports.append(run_dataset(config, mode=mode, jobs=args.jobs, use_cache=not args.no_cache))
    invocation_seconds = perf_counter()-total_started
    executed_ids = [r['dataset_id'] for r in reports]
    # Partial runs retain other problems' real recorded summaries.
    reports = [json.loads((ROOT/'results'/key/args.mode/'run.json').read_text(encoding='utf-8'))
               for key in OFFICIAL_DATASETS if (ROOT/'results'/key/args.mode/'run.json').is_file()]
    summary = {"mode": args.mode, "runtime_seconds": sum(r['runtime_seconds'] for r in reports),
               "latest_invocation_runtime_seconds": invocation_seconds, "latest_invocation_datasets": executed_ids,
               "runtime_note": "Sum of recorded per-dataset pipeline durations; runs may be separate invocations. Latest invocation time is recorded separately.",
               "datasets": [{"id": r["dataset_id"], "runtime_seconds": r["runtime_seconds"],
                              "best_method": r["best_model"]["method"], "workflow_nested_cv_rmse": r["workflow_nested_cv_rmse"],
                              "methods": [{"method": m["method"], "selected_encoded_count": m["selected_encoded_count"],
                                           "best_cv_rmse": m["best_cv_rmse"], "nested_cv_rmse": m["nested_cv_rmse"],
                                           "test_rmse": m["test_metrics"]["rmse"], "test_r2": m["test_metrics"]["r2"]} for m in r["methods"]]}
                             for r in reports]}
    write_json(ROOT/"results"/("phase2_"+args.mode+"_summary.json"), summary)
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
