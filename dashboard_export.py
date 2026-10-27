"""Export real fitted models for the browser dashboard (no server required)."""
import json
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor


def export_dashboard(curves, X_train, X_test, y_train, y_test, comparison, dataset_id="startups", metadata=None):
    models = []
    for _, row in curves.iterrows():
        features = row["Selected Features"].split(", ")
        fit = LinearRegression().fit(X_train[features], y_train)
        predicted = fit.predict(X_test[features])
        design = sm.add_constant(X_train[features].astype(float), has_constant="add")
        ols = sm.OLS(y_train, design).fit()
        intervals = np.asarray(ols.conf_int())
        coefficients = []
        for i, name in enumerate(["Intercept"] + features):
            coefficients.append(dict(
                name=name, estimate=float(ols.params.iloc[i]),
                se=float(ols.bse.iloc[i]), t=float(ols.tvalues.iloc[i]),
                p=float(ols.pvalues.iloc[i]), low=float(intervals[i, 0]),
                high=float(intervals[i, 1]),
                vif=None if i == 0 else float(variance_inflation_factor(design.to_numpy(), i))
            ))
        k, n = len(features), len(y_test)
        residuals = np.asarray(y_test) - predicted
        models.append(dict(
            method=row["Model"], k=k, features=features,
            rmse=float(row["Test RMSE"]), r2=float(row["Test R2"]),
            adjustedR2=float(1 - (1 - row["Test R2"]) * (n - 1) / (n - k - 1)),
            mae=float(mean_absolute_error(y_test, predicted)),
            f=float(ols.fvalue), fP=float(ols.f_pvalue),
            intercept=float(fit.intercept_), weights=[float(v) for v in fit.coef_],
            coefficients=coefficients,
            residuals=[dict(id=int(index) + 1, actual=float(actual), predicted=float(pred),
                            residual=float(error), percent=float(abs(error / actual) * 100))
                       for index, actual, pred, error in zip(y_test.index, y_test, predicted, residuals)]
        ))
    payload = dict(
        generatedAt=datetime.now(timezone.utc).isoformat(), trainSize=len(y_train),
        testSize=len(y_test), seed=0, models=models,
        comparison=comparison.to_dict(orient="records"),
        ranges={name: dict(min=float(X_train[name].min()), max=float(X_train[name].max()))
                for name in X_train.columns}
    )
    output = Path(__file__).resolve().parent / "results" / "historical_boston_startups"
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata or {
        "id": "startups", "title": "50 Startups", "problem": "50 Startups：公司利潤預測",
        "target": "Profit", "targetLabel": "公司利潤", "unit": "美元", "unitPrefix": "$",
        "rawSize": 50, "sampleLabel": "Startup", "source": "50_Startups.csv",
        "description": "從研發、行銷、行政投入與所在地，探索新創公司的利潤預測。",
        "missingPolicy": "移除重複與缺失資料列，再進行固定 80% / 20% 分割。",
        "targetNote": "Profit 單位為美元。",
        "fieldLabels": {"R&D Spend": "研發投入", "Marketing Spend": "行銷投入", "Administration": "行政支出"}
    })
    metadata["cleanSize"] = len(y_train) + len(y_test)
    metadata["maxFeatures"] = X_train.shape[1]
    inputs = []
    for name in X_train.columns:
        if name.startswith("State_"):
            continue
        col = X_train[name]
        span = float(col.max() - col.min())
        binary = col.nunique() <= 2 and set(col.unique()).issubset({0., 1.})
        integer = bool(np.allclose(col, np.round(col)))
        step = 1. if binary or integer else float(10 ** np.floor(np.log10(max(span, 1e-9) / 1000)))
        inputs.append(dict(name=name, label=metadata.get("fieldLabels", {}).get(name, name),
                           default=float(col.median()), min=float(col.min()), max=float(col.max()),
                           step=step, binary=binary))
    recommended = next(row for row in comparison.to_dict(orient="records") if row["Model"] != "Baseline")
    payload.update(metadata=metadata, inputs=inputs,
                   defaultMethod=recommended["Model"], defaultK=int(recommended["Features"]))
    existing = json.loads((output / "data.json").read_text(encoding="utf-8")) if (output / "data.json").exists() else {}
    datasets = dict(existing.get("datasets", {}))
    datasets[dataset_id] = payload
    startup = datasets.get("startups", payload)
    bundle = {**startup, "schemaVersion": 3, "datasets": datasets}
    encoded = json.dumps(bundle, ensure_ascii=False, allow_nan=False)
    (output / "data.json").write_text(encoded, encoding="utf-8")
    (output / "data.js").write_text("window.DASHBOARD_DATA = " + encoded + ";\n", encoding="utf-8")
    refresh_asset_versions(output)
    print("Interactive dashboard data exported:", output)


def refresh_asset_versions(output=None):
    """Match asset URLs to their content so browsers cannot reuse old JS/data."""
    output = Path(output) if output is not None else Path(__file__).resolve().parent / "dashboard" / "dist"
    index = output / "index.html"
    if not index.exists():
        return
    versions = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()[:12]
                for name in ("app.js", "prediction.js", "data.js", "styles.css")}
    text = index.read_text(encoding="utf-8")
    text = re.sub(
        r'((?:src|href)=")(app\.js|prediction\.js|data\.js|styles\.css)(?:\?v=[^"]*)?(")',
        lambda match: f'{match[1]}{match[2]}?v={versions[match[2]]}{match[3]}', text
    )
    index.write_text(text, encoding="utf-8")



def export_official_dashboard(data, X_train, X_test, y_train, y_test, comparison,
                              curve_models, fitted_models, run_metadata):
    """Export existing fitted pipelines, never refit against evaluation data."""
    from datasets.preprocessing import MANUFACTURER_ALIASES
    config = data.config
    pre = fitted_models["Baseline"]["preprocessor"]
    numeric = pre.named_transformers_["numeric"]
    descriptor = {
        "numericColumns": list(config.numeric_columns),
        "numericMedian": numeric["imputer"].statistics_.tolist(),
        "numericMean": numeric["scaler"].mean_.tolist(),
        "numericScale": numeric["scaler"].scale_.tolist(),
        "categoricalColumns": list(config.categorical_columns),
        "manufacturerColumn": "CarName" if config.dataset_id == "car_price" else None,
        "manufacturerAliases": MANUFACTURER_ALIASES if config.dataset_id == "car_price" else {},
        "featureNames": pre.get_feature_names_out().tolist(),
    }
    cat = pre.named_transformers_["categorical"]
    descriptor["categoricalFill"] = cat["imputer"].statistics_.tolist()
    descriptor["categories"] = [c.tolist() for c in cat["encoder"].categories_]
    transformed_train = pre.transform(X_train)
    ranges = {n: {"min": float(transformed_train[:,i].min()), "max": float(transformed_train[:,i].max())}
              for i,n in enumerate(descriptor["featureNames"])}
    inputs = []
    for name in config.feature_names:
        if name in config.numeric_columns:
            col = X_train[name]
            i = list(config.numeric_columns).index(name)
            inputs.append({"name": name, "label": name, "kind": "numeric", "default": float(descriptor["numericMedian"][i]),
                           "min": float(col.min()), "max": float(col.max()), "step": 1 if np.allclose(col.dropna(), col.dropna().round()) else 0.01,
                           "binary": False})
        else:
            i = list(config.categorical_columns).index(name)
            inputs.append({"name": name, "label": "Manufacturer (from CarName)" if name == "CarName" else name,
                           "kind": "categorical", "default": str(descriptor["categoricalFill"][i]),
                           "options": descriptor["categories"][i]})
    models = []
    from regression_experiment import selected_features
    for method, k, fitted, cv_rmse in sorted(curve_models, key=lambda item: (item[0], item[1])):
        features = selected_features(fitted).tolist()
        transformed = fitted["preprocessor"].transform(X_train)
        selected = fitted["selector"].transform(transformed)
        design = sm.add_constant(selected, has_constant="add")
        # Full dummy sets can be rank-deficient. Their individual OLS tests
        # are not identifiable; retain predictions but report diagnostics as null.
        identifiable = bool(np.linalg.matrix_rank(design) == design.shape[1] and np.linalg.cond(design) < 1e10)
        ols = sm.OLS(y_train.to_numpy(), design).fit() if identifiable else None
        interval = np.asarray(ols.conf_int()) if ols is not None else None
        reg = fitted["regressor"]
        coefficients = []
        for i, name in enumerate(["Intercept", *features]):
            vif = None
            if i and identifiable:
                other = np.delete(design, i, axis=1)
                residual_i = design[:,i] - other @ np.linalg.lstsq(other, design[:,i], rcond=None)[0]
                total_i = np.sum((design[:,i] - design[:,i].mean())**2)
                error_i = np.sum(residual_i**2)
                vif = float(total_i/error_i) if error_i > 1e-12 else None
            coefficients.append(dict(name=name, estimate=float(reg.intercept_ if i == 0 else reg.coef_[i-1]),
                                     se=float(ols.bse[i]) if ols is not None else None,
                                     t=float(ols.tvalues[i]) if ols is not None else None,
                                     p=float(ols.pvalues[i]) if ols is not None else None,
                                     low=float(interval[i,0]) if interval is not None else None,
                                     high=float(interval[i,1]) if interval is not None else None, vif=vif))
        pred = fitted.predict(X_test)
        residual = y_test.to_numpy() - pred
        from regression_experiment import metrics
        score = metrics(y_test, pred, k)
        models.append(dict(method=method, k=k, features=features, cvRmse=cv_rmse,
                           rmse=score["Test RMSE"], r2=score["Test R2"], adjustedR2=score["Test Adjusted R2"], mae=score["Test MAE"],
                           f=float(ols.fvalue) if ols is not None else None, fP=float(ols.f_pvalue) if ols is not None else None,
                           diagnosticsAvailable=identifiable, intercept=float(reg.intercept_), weights=reg.coef_.tolist(), coefficients=coefficients,
                           residuals=[dict(id=int(index)+1, actual=float(actual), predicted=float(p), residual=float(r),
                                           percent=float(abs(r/actual)*100) if actual else None)
                                      for index,actual,p,r in zip(X_test.index,y_test,pred,residual)]))
    candidates = comparison[comparison.Model != "Baseline"]
    recommended = candidates.iloc[0]
    car = config.dataset_id == "car_price"
    metadata = dict(id=config.dataset_id, title=config.display_name, problem=config.display_name,
                    target=config.target, targetLabel="Car price" if car else "Final mathematics grade", unit=config.target_unit,
                    unitPrefix="$" if car else "", rawSize=config.expected_rows, cleanSize=len(data.y),
                    originalFeatureCount=config.original_feature_count, encodedFeatureCount=len(descriptor["featureNames"]),
                    maxFeatures=max(m["k"] for m in models), sampleLabel="Car" if car else "Student",
                    source="https://www.kaggle.com/datasets/hellbuoy/car-price-prediction" if car else "https://archive.ics.uci.edu/dataset/320/student+performance",
                    description="Predict historical car prices from vehicle specifications and manufacturer." if car else "Predict Mathematics G3 using 30 background and study inputs; G1/G2 are excluded.",
                    missingPolicy="Per-fold median/mode imputation, numeric scaling and OneHotEncoder; seeded 80%/20% split.",
                    targetNote="Only 205 historical observations; generalization is uncertain. Price unit: USD." if car else "G3 is a 0-20 grade. Historical G1/G2 grades are excluded from the official early prediction model.",
                    fieldLabels={f["name"]: f["label"] for f in inputs})
    payload = dict(generatedAt=datetime.now(timezone.utc).isoformat(), seed=0,
                   trainSize=len(y_train), testSize=len(y_test), metadata=metadata, preprocessing=descriptor,
                   models=models, comparison=comparison.to_dict(orient="records"), ranges=ranges, inputs=inputs,
                   baseline=dict(rmse=float(comparison[comparison.Model == "Baseline"].iloc[0]["Test RMSE"])),
                   defaultMethod=str(recommended.Model), defaultK=int(recommended.Features), runMetadata=run_metadata)
    # json forbids NaN/Infinity; undefined diagnostics are explicitly null.
    def clean(value):
        if isinstance(value, np.generic): return clean(value.item())
        if isinstance(value, dict): return {k:clean(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)): return [clean(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value): return None
        return value
    return clean(payload)


def write_official_bundle(payloads):
    """Historical schema-4 writer; never overwrite the Phase 4 dashboard."""
    from datasets.config import LEGACY_DATASETS as OFFICIAL_DATASETS
    if set(payloads) - set(OFFICIAL_DATASETS):
        raise ValueError("Unexpected official dashboard dataset")
    output = Path(__file__).resolve().parent / "results" / "historical_student_car"
    output.mkdir(parents=True, exist_ok=True)
    bundle = {"schemaVersion": 4, "datasets": {k:payloads[k] for k in OFFICIAL_DATASETS if k in payloads}}
    encoded = json.dumps(bundle, ensure_ascii=False, allow_nan=False)
    (output / "data.json").write_text(encoded, encoding="utf-8")
    (output / "data.js").write_text("window.DASHBOARD_DATA = " + encoded + ";\n", encoding="utf-8")
    refresh_asset_versions(output)


def export_current_dashboard(mode='FAST'):
    """Phase 4 entry point; legacy export functions remain historical references."""
    from dashboard_integration import export_dashboard
    return export_dashboard(mode)


def main():
    import argparse
    parser=argparse.ArgumentParser(description='Export the official Student/Energy/Car static dashboard from real model artifacts')
    parser.add_argument('--mode',choices=['FAST','FULL'],default='FAST',type=str.upper)
    args=parser.parse_args()
    export_current_dashboard(args.mode)


if __name__=='__main__':
    main()
