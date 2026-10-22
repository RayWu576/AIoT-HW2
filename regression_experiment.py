"""Reproducible official experiments: raw-input CV, held-out evaluation, dashboard."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from datasets.config import LEGACY_DATASETS as OFFICIAL_DATASETS
from datasets import load_dataset, make_preprocessor, holdout_indices, make_cv
from feature_selection import OfficialSelector
from dashboard_export import export_official_dashboard

ROOT = Path(__file__).resolve().parent
METHODS = ("Correlation", "Mutual Information", "RFE", "RFECV", "LASSO")
# A predeclared cap limits variance on 205/395 observations. No test-set tuning.
MAX_SELECTED = 20
ALPHA_FRACTIONS = (0.001, 0.01, 0.1)


def make_pipeline(config, method, k=10, alpha_fraction=0.01):
    steps = [("preprocessor", make_preprocessor(config))]
    if method != "Baseline":
        steps.append(("selector", OfficialSelector(method, k, alpha_fraction)))
    return Pipeline(steps + [("regressor", LinearRegression())])


def metrics(y, predicted, p):
    r2 = float(r2_score(y, predicted))
    n = len(y)
    return {"Test MAE": float(mean_absolute_error(y, predicted)),
            "Test RMSE": float(np.sqrt(mean_squared_error(y, predicted))),
            "Test R2": r2,
            "Test Adjusted R2": float(1-(1-r2)*(n-1)/(n-p-1)) if n > p+1 else None}


def selected_features(fitted):
    names = fitted["preprocessor"].get_feature_names_out()
    return names if "selector" not in fitted.named_steps else names[fitted["selector"].selected_indices_]


def run_dataset(config, *, folds=4, max_selected=MAX_SELECTED):
    data = load_dataset(config)
    train, test = holdout_indices(data, random_state=0)
    X_train, X_test = data.X.iloc[train], data.X.iloc[test]
    y_train, y_test = data.y.iloc[train], data.y.iloc[test]
    cv = make_cv(config, folds)
    output = ROOT / "results" / config.dataset_id
    output.mkdir(parents=True, exist_ok=True)
    fitted_models, curve_models, comparisons, curves = {}, [], [], []
    baseline = make_pipeline(config, "Baseline")
    scores = cross_val_score(baseline, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
    baseline.fit(X_train, y_train)
    fitted_models["Baseline"] = baseline
    comparisons.append({"Model": "Baseline", "Features": len(selected_features(baseline)),
                        "CV RMSE": float(-scores.mean()), "CV Std": float(scores.std()),
                        **metrics(y_test, baseline.predict(X_test), len(selected_features(baseline)))})
    for method in METHODS:
        params = {"selector__k": list(range(1, max_selected+1))}
        if method == "LASSO":
            params["selector__k"].append(None)
            params["selector__alpha_fraction"] = list(ALPHA_FRACTIONS)
        search = GridSearchCV(make_pipeline(config, method), params, cv=cv,
                              scoring="neg_root_mean_squared_error", refit=True,
                              n_jobs=1, error_score="raise", return_train_score=False)
        search.fit(X_train, y_train)
        # LASSO's official automatic model uses nonzero support; curves remain fixed-k rankings.
        automatic_indices = [i for i,p in enumerate(search.cv_results_["params"])
                             if p["selector__k"] is None] if method == "LASSO" else list(range(len(search.cv_results_["params"])))
        best_index = max(automatic_indices, key=lambda i: search.cv_results_["mean_test_score"][i])
        params_best = search.cv_results_["params"][best_index]
        fitted = clone(search.estimator).set_params(**params_best).fit(X_train, y_train)
        fitted_models[method] = fitted
        count = len(selected_features(fitted))
        comparisons.append({"Model": method, "Features": count,
                            "CV RMSE": float(-search.cv_results_["mean_test_score"][best_index]),
                            "CV Std": float(search.cv_results_["std_test_score"][best_index]),
                            **metrics(y_test, fitted.predict(X_test), count)})
        # Each plotted point is fitted on training rows, with alpha selected on training CV only.
        for k in range(1, max_selected+1):
            indices = [i for i,p in enumerate(search.cv_results_["params"]) if p["selector__k"] == k]
            i = max(indices, key=lambda j: search.cv_results_["mean_test_score"][j])
            point = clone(search.estimator).set_params(**search.cv_results_["params"][i]).fit(X_train, y_train)
            curve_models.append((method, k, point, float(-search.cv_results_["mean_test_score"][i])))
        if method == "LASSO":
            # Include the genuine automatic L1 support even when it exceeds the fixed-k cap.
            curve_models = [m for m in curve_models if not (m[0] == method and m[1] == count)]
            curve_models.append((method, count, fitted, comparisons[-1]["CV RMSE"]))
        print(f"{config.dataset_id} {method}: selected={count}, train CV RMSE={comparisons[-1]['CV RMSE']:.4f}", flush=True)
        pd.DataFrame(search.cv_results_).to_csv(output / (method.replace(" ", "_") + "_cv_search.csv"), index=False)
    comparison = pd.DataFrame(comparisons).sort_values("CV RMSE", kind="stable").reset_index(drop=True)
    comparison.insert(0, "Rank", np.arange(1, len(comparison)+1))
    # Choosing the method occurs strictly by training-only CV, before final reporting.
    best_name = str(comparison.iloc[0]["Model"])
    joblib.dump({"model": fitted_models[best_name], "model_name": best_name,
                 "dataset_id": config.dataset_id, "feature_names": list(config.feature_names)}, output / "best_model.pkl")
    joblib.dump(fitted_models, output / "models.pkl")
    for method, k, point, cv_rmse in curve_models:
        curves.append({"Model": method, "Features": k, "CV RMSE": cv_rmse,
                       "Selected Features": ", ".join(selected_features(point)),
                       **metrics(y_test, point.predict(X_test), k)})
    comparison.to_csv(output / "model_comparison.csv", index=False)
    pd.DataFrame(curves).to_csv(output / "feature_count_metrics.csv", index=False)
    pd.DataFrame([{"Model": m, "Selected Features": ", ".join(selected_features(p)),
                   "Feature Count": len(selected_features(p))} for m,p in fitted_models.items()]).to_csv(output / "selected_features.csv", index=False)
    predictions = pd.DataFrame({"Sample ID": X_test.index+1, "Actual": y_test.to_numpy()})
    for method, fitted in fitted_models.items(): predictions[method] = fitted.predict(X_test)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    metadata = {"dataset_id": config.dataset_id, "original_feature_count": config.original_feature_count,
                "encoded_feature_count": len(baseline["preprocessor"].get_feature_names_out()),
                "target": config.target, "target_unit": config.target_unit,
                "train_indices": train.tolist(), "test_indices": test.tolist(), "seed": 0,
                "cv_folds": folds, "selection_metric": "training-only CV RMSE",
                "cv_note": "Tuning scores, not an unbiased nested-CV performance estimate",
                "best_method": best_name, "max_fixed_k": max_selected,
                "feature_names": list(config.feature_names), "raw_columns": list(config.raw_columns)}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    payload = export_official_dashboard(data, X_train, X_test, y_train, y_test,
                                         comparison, curve_models, fitted_models, metadata)
    return payload



def rebuild_export(config):
    """Recreate curve fits from recorded training CV choices; no new selection."""
    import ast
    data = load_dataset(config)
    output = ROOT / "results" / config.dataset_id
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    train, test = metadata["train_indices"], metadata["test_indices"]
    X_train, X_test = data.X.iloc[train], data.X.iloc[test]
    y_train, y_test = data.y.iloc[train], data.y.iloc[test]
    fitted_models = joblib.load(output / "models.pkl")
    curve_models = []
    for method in METHODS:
        results = pd.read_csv(output / (method.replace(" ", "_") + "_cv_search.csv"))
        params = [ast.literal_eval(p) for p in results.params]
        for k in range(1, metadata["max_fixed_k"]+1):
            indices = [i for i,p in enumerate(params) if p["selector__k"] == k]
            i = max(indices, key=lambda j: results.iloc[j].mean_test_score)
            fitted = make_pipeline(config, method).set_params(**params[i]).fit(X_train, y_train)
            curve_models.append((method, k, fitted, float(-results.iloc[i].mean_test_score)))
        if method == "LASSO":
            auto = fitted_models[method]
            count = len(selected_features(auto))
            curve_models = [m for m in curve_models if not (m[0] == method and m[1] == count)]
            comparison = pd.read_csv(output / "model_comparison.csv")
            cv_rmse = float(comparison[comparison.Model == method].iloc[0]["CV RMSE"])
            curve_models.append((method, count, auto, cv_rmse))
    comparison = pd.read_csv(output / "model_comparison.csv")
    # Feature order must match selector.transform, not sorted boolean support.
    pd.DataFrame([{"Model": m, "Selected Features": ", ".join(selected_features(fitted)),
                   "Feature Count": len(selected_features(fitted))} for m,fitted in fitted_models.items()]).to_csv(output / "selected_features.csv", index=False)
    pd.DataFrame([{"Model": method, "Features": k, "CV RMSE": cv_rmse,
                   "Selected Features": ", ".join(selected_features(point)),
                   **metrics(y_test, point.predict(X_test), k)}
                  for method,k,point,cv_rmse in curve_models]).to_csv(output / "feature_count_metrics.csv", index=False)
    return export_official_dashboard(data, X_train, X_test, y_train, y_test, comparison,
                                     curve_models, fitted_models, metadata)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *OFFICIAL_DATASETS], default="all")
    parser.add_argument("--export-only", action="store_true", help="Rebuild dashboard from recorded training CV choices")
    args = parser.parse_args()
    output = ROOT / "results" / "historical_student_car"
    payloads = {}
    if args.dataset != "all" and (output / "data.json").exists():
        payloads = {k:v for k,v in json.loads((output/"data.json").read_text(encoding="utf-8")).get("datasets", {}).items() if k in OFFICIAL_DATASETS}
    with threadpool_limits(limits=1):
        for dataset_id, config in OFFICIAL_DATASETS.items():
            if args.dataset in ("all", dataset_id): payloads[dataset_id] = rebuild_export(config) if args.export_only else run_dataset(config)
    from dashboard_export import write_official_bundle
    write_official_bundle(payloads)

if __name__ == "__main__":
    main()
