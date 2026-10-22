"""Unfitted, cloneable preprocessing to fit only inside training folds."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Fixed lexical corrections, independent of targets or dataset frequencies.
MANUFACTURER_ALIASES = {"maxda": "mazda", "porcshce": "porsche", "toyouta": "toyota",
                        "vokswagen": "volkswagen", "vw": "volkswagen"}


def manufacturer_name(value):
    if pd.isna(value) or not str(value).strip():
        return np.nan
    token = str(value).strip().lower().split()[0]
    return MANUFACTURER_ALIASES.get(token, token)


class ManufacturerTransformer(TransformerMixin, BaseEstimator):
    """Deterministic name -> brand mapping; no fitted vocabulary or target access."""
    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=object).copy()
        values[:, 0] = [manufacturer_name(v) for v in values[:, 0]]
        return values

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else ["CarName"], dtype=object)


def make_preprocessor(config):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    transformers = [("numeric", numeric, list(config.numeric_columns))]
    if config.categorical_columns:
        steps = []
        if config.dataset_id == "car_price":
            # CarName is explicitly first in the categorical schema.
            steps.append(("manufacturer", ManufacturerTransformer()))
        categorical = Pipeline(steps + [
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical, list(config.categorical_columns)))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def feature_counts(config, fitted_preprocessor):
    """Encoded width is learned per training fold; never treat it as raw width."""
    return {"original": config.original_feature_count,
            "encoded": len(fitted_preprocessor.get_feature_names_out())}
