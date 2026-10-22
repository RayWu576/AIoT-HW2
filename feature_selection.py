"""Shared training-only feature selectors for startup and housing experiments."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, RFECV, RFE
from sklearn.pipeline import Pipeline
RANDOM_STATE = 0
SCORING = "neg_root_mean_squared_error"

def subset_columns(X, indices):
    """Select columns by integer positions."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, indices]

    return np.asarray(X)[:, indices]


class BaseSubsetSelector(
    BaseEstimator,
    TransformerMixin
):
    selected_indices_: np.ndarray
    n_features_in_: int

    def transform(self, X):
        return subset_columns(
            X,
            self.selected_indices_
        )

    def get_support(self):
        mask = np.zeros(
            self.n_features_in_,
            dtype=bool
        )
        mask[self.selected_indices_] = True
        return mask


# ============================================================
# FIXED-K SELECTION: CORRELATION, MUTUAL INFORMATION, RFE, LASSO
# ============================================================

class FixedCountSelector(BaseSubsetSelector):
    """Fit a feature selection method on training data for a specified k.

    LASSO fixed-k curves rank absolute standardized LassoCV coefficients.
    Zero-coefficient ties retain input column order. RFECV curves use the
    fixed-k RFE elimination path; RFECV itself selects k by cross-validation.
    """

    def __init__(self, method="Correlation", k=1, cv=4):
        self.method = method
        self.k = k
        self.cv = cv

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        if not 1 <= self.k <= self.n_features_in_:
            raise ValueError("k must be between 1 and the number of features.")
        values = np.asarray(X, dtype=float)
        if self.k == self.n_features_in_:
            self.selected_indices_ = np.arange(self.n_features_in_)
            return self

        if self.method == "Correlation":
            centered = values - values.mean(axis=0)
            target = np.asarray(y) - np.mean(y)
            denominator = np.sqrt((centered ** 2).sum(axis=0) * (target ** 2).sum())
            scores = np.divide(
                np.abs(centered.T @ target), denominator,
                out=np.zeros(self.n_features_in_), where=denominator > 0
            )
        elif self.method == "Mutual Information":
            # The two one-hot State columns are discrete features.
            discrete = np.array([np.unique(values[:, i]).size <= 2 for i in range(values.shape[1])])
            scores = mutual_info_regression(
                values, y, discrete_features=discrete, random_state=RANDOM_STATE
            )
        elif self.method in ("RFE", "RFECV"):
            scaled = StandardScaler().fit_transform(values)
            selector = RFE(LinearRegression(), n_features_to_select=self.k, step=1)
            selector.fit(scaled, y)
            self.selected_indices_ = np.flatnonzero(selector.support_)
            return self
        elif self.method == "LASSO":
            scaled = StandardScaler().fit_transform(values)
            lasso = LassoCV(
                cv=KFold(n_splits=self.cv, shuffle=True, random_state=RANDOM_STATE),
                max_iter=100000, random_state=RANDOM_STATE
            ).fit(scaled, y)
            scores = np.abs(lasso.coef_)
        else:
            raise ValueError(f"Unknown selection method: {self.method}")

        self.selected_indices_ = np.argsort(-scores, kind="stable")[:self.k]
        return self


class TunedCountSelector(BaseSubsetSelector):
    """Choose k using training CV, refitting selection inside every fold."""

    def __init__(self, method="Correlation", cv=4):
        self.method = method
        self.cv = cv

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        cv = KFold(n_splits=self.cv, shuffle=True, random_state=RANDOM_STATE)
        best_score = -np.inf
        for k in range(1, self.n_features_in_ + 1):
            candidate = Pipeline([
                ("selector", FixedCountSelector(self.method, k, self.cv)),
                ("regressor", LinearRegression())
            ])
            score = cross_val_score(candidate, X, y, cv=cv, scoring=SCORING).mean()
            if score > best_score:
                best_score = score
                self.best_k_ = k
        fitted = FixedCountSelector(self.method, self.best_k_, self.cv).fit(X, y)
        self.selected_indices_ = fitted.selected_indices_
        return self


# ============================================================
# M4: RECURSIVE FEATURE ELIMINATION WITH CV
# ============================================================

class RFECVSelector(BaseSubsetSelector):

    def __init__(self, cv=4):
        self.cv = cv

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]

        cv = KFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        # Standardization makes coefficient
        # magnitudes comparable during RFE.
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        selector = RFECV(
            estimator=LinearRegression(),
            step=1,
            min_features_to_select=1,
            cv=cv,
            scoring=SCORING
        )

        selector.fit(X_scaled, y)

        self.selected_indices_ = np.where(
            selector.support_
        )[0]

        return self


# ============================================================
# M5: LASSO FEATURE SELECTION
# ============================================================

class LassoSelector(BaseSubsetSelector):

    def __init__(self, cv=4):
        self.cv = cv

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]

        self.scaler_ = StandardScaler()

        X_scaled = self.scaler_.fit_transform(X)

        lasso = LassoCV(
            cv=KFold(
                n_splits=self.cv,
                shuffle=True,
                random_state=RANDOM_STATE
            ),
            max_iter=100000,
            random_state=RANDOM_STATE
        )

        lasso.fit(X_scaled, y)

        coefficients = np.abs(lasso.coef_)

        indices = np.where(
            coefficients > 1e-7
        )[0]

        # Ensure at least one feature is selected.
        if len(indices) == 0:
            indices = np.array([
                np.argmax(coefficients)
            ])

        self.selected_indices_ = indices

        return self




class OfficialSelector(BaseSubsetSelector):
    """Fold-local adapters preserving the five historical selection algorithms.

    k=None uses LASSO's nonzero coefficients. Fixed-k LASSO curves use its
    coefficient ranking as in the historical dashboard. RFECV's k search is
    performed by GridSearchCV around the *whole raw-input pipeline*, so its
    inner validation rows never fit imputation, scaling, encoding or RFE.
    """
    def __init__(self, method="Correlation", k=10, alpha_fraction=0.01):
        self.method = method
        self.k = k
        self.alpha_fraction = alpha_fraction

    def fit(self, X, y):
        from sklearn.linear_model import Lasso
        self.n_features_in_ = X.shape[1]
        k = min(self.k, self.n_features_in_) if self.k is not None else None
        if self.method != "LASSO":
            if k is None:
                raise ValueError("Only LASSO supports automatic nonzero selection")
            fitted = FixedCountSelector(self.method, k).fit(X, y)
            self.selected_indices_ = fitted.selected_indices_
            return self
        self.scaler_ = StandardScaler().fit(X)
        values = self.scaler_.transform(X)
        self.lasso_ = Lasso(alpha=self.alpha_fraction * max(float(np.std(y)), 1e-12),
                            max_iter=100000, tol=1e-5, selection="cyclic").fit(values, y)
        scores = np.abs(self.lasso_.coef_)
        if k is None:
            selected = np.flatnonzero(scores > 1e-7)
            self.selected_indices_ = selected if len(selected) else np.array([np.argmax(scores)])
        else:
            self.selected_indices_ = np.argsort(-scores, kind="stable")[:k]
        return self


class CachedRankingSelector(BaseSubsetSelector):
    """Fold-local ranking independent of k, cached by Pipeline across k values.

    RFE/RFECV compute one complete elimination path per training fold. LASSO
    returns only nonzero support; the downstream predictor is LinearRegression.
    There is no internal CV here: raw-input Pipeline search owns every split.
    """
    def __init__(self, method="Correlation", alpha_fraction=0.01, random_state=42):
        self.method = method
        self.alpha_fraction = alpha_fraction
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.linear_model import Lasso
        values = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)
        self.n_features_in_ = values.shape[1]
        if self.method == "Correlation":
            centered = values - values.mean(axis=0)
            centered_y = target-target.mean()
            denominator = np.sqrt((centered**2).sum(axis=0)*(centered_y**2).sum())
            scores = np.divide(np.abs(centered.T @ centered_y), denominator,
                               out=np.zeros(self.n_features_in_), where=denominator > 0)
            indices = np.argsort(-scores, kind="stable")
        elif self.method == "Mutual Information":
            discrete = np.array([np.unique(values[:,i]).size <= 2 for i in range(values.shape[1])])
            scores = mutual_info_regression(values, target, discrete_features=discrete,
                                            random_state=self.random_state)
            indices = np.argsort(-scores, kind="stable")
        elif self.method in ("RFE", "RFECV"):
            self.scaler_ = StandardScaler().fit(values)
            elimination = RFE(LinearRegression(), n_features_to_select=1, step=1)
            elimination.fit(self.scaler_.transform(values), target)
            self.ranking_ = elimination.ranking_
            indices = np.argsort(self.ranking_, kind="stable")
        elif self.method == "LASSO":
            self.scaler_ = StandardScaler().fit(values)
            self.lasso_ = Lasso(alpha=self.alpha_fraction*max(float(target.std()), 1e-12),
                                max_iter=100000, tol=1e-5, random_state=self.random_state)
            self.lasso_.fit(self.scaler_.transform(values), target)
            scores = np.abs(self.lasso_.coef_)
            indices = np.flatnonzero(scores > 1e-7)
            if len(indices) == 0:
                indices = np.array([int(np.argmax(scores))])
        else:
            raise ValueError(f"Unknown feature selection method: {self.method}")
        self.selected_indices_ = np.asarray(indices, dtype=int)
        return self

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features if input_features is not None else
                           [f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return names[self.selected_indices_]


class RankedSubset(BaseSubsetSelector):
    """Cheap prefix selection; k='all' adapts to training-fold encoded width."""
    def __init__(self, k="all"):
        self.k = k

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        if self.k != "all" and (not isinstance(self.k, (int, np.integer)) or self.k < 1):
            raise ValueError("k must be a positive integer or 'all'")
        self.effective_k_ = self.n_features_in_ if self.k == "all" else min(int(self.k), self.n_features_in_)
        self.selected_indices_ = np.arange(self.effective_k_)
        return self

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features if input_features is not None else
                           [f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return names[self.selected_indices_]
