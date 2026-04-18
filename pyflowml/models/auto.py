"""
AutoML — Smart AutoClassifier, AutoRegressor, AutoClusterer.

Features:
- Dataset-size-aware model pruning
- Parallel training via joblib
- Time budget enforcement via iterative random hyper-param search
- Early stopping for boosting models
- Leaderboard with training time + metric scores
- Human-readable model recommendation
"""

import time
import warnings
import random
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error,
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

from pyflowml.core.optimizer import ModelOptimizer
from pyflowml.monitoring.logger import get_logger
from pyflowml.monitoring.tracker import StepTracker

logger = get_logger("AutoML")
warnings.filterwarnings("ignore")


# ─── Hyper-parameter search spaces ───────────────────────────────────────────

_REGRESSION_PARAM_SPACE = {
    "Ridge": lambda: {"alpha": random.choice([0.01, 0.1, 1.0, 10.0, 100.0])},
    "Lasso": lambda: {"alpha": random.choice([0.001, 0.01, 0.1, 1.0, 10.0])},
    "RandomForest": lambda: {
        "n_estimators": random.choice([50, 100, 200, 300]),
        "max_depth": random.choice([None, 5, 10, 20, 30]),
        "min_samples_split": random.choice([2, 5, 10]),
        "max_features": random.choice(["sqrt", "log2", 0.5, 0.8]),
    },
    "GradientBoosting": lambda: {
        "n_estimators": random.choice([100, 200, 300]),
        "learning_rate": random.choice([0.01, 0.05, 0.1, 0.2]),
        "max_depth": random.choice([3, 5, 7]),
        "subsample": random.choice([0.7, 0.8, 1.0]),
    },
    "SVR": lambda: {
        "C": random.choice([0.1, 1.0, 10.0, 100.0]),
        "epsilon": random.choice([0.01, 0.1, 0.5]),
        "kernel": random.choice(["rbf", "linear"]),
    },
    "KNN": lambda: {
        "n_neighbors": random.choice([3, 5, 7, 11, 15]),
        "weights": random.choice(["uniform", "distance"]),
    },
    "XGBoost": lambda: {
        "n_estimators": random.choice([100, 200, 300, 500]),
        "learning_rate": random.choice([0.01, 0.05, 0.1, 0.2]),
        "max_depth": random.choice([3, 5, 6, 8]),
        "subsample": random.choice([0.7, 0.8, 1.0]),
        "colsample_bytree": random.choice([0.6, 0.8, 1.0]),
        "reg_alpha": random.choice([0, 0.1, 0.5]),
        "reg_lambda": random.choice([1, 2, 5]),
    },
    "LightGBM": lambda: {
        "n_estimators": random.choice([100, 200, 300, 500]),
        "learning_rate": random.choice([0.01, 0.05, 0.1, 0.2]),
        "num_leaves": random.choice([31, 63, 127]),
        "subsample": random.choice([0.7, 0.8, 1.0]),
    },
}

_CLASSIFICATION_PARAM_SPACE = {
    "LogisticRegression": lambda: {
        "C": random.choice([0.01, 0.1, 1.0, 10.0, 100.0]),
        "solver": random.choice(["lbfgs", "saga"]),
    },
    "RandomForest": lambda: {
        "n_estimators": random.choice([50, 100, 200, 300]),
        "max_depth": random.choice([None, 5, 10, 20]),
        "min_samples_split": random.choice([2, 5, 10]),
        "max_features": random.choice(["sqrt", "log2"]),
    },
    "GradientBoosting": lambda: {
        "n_estimators": random.choice([100, 200, 300]),
        "learning_rate": random.choice([0.01, 0.05, 0.1, 0.2]),
        "max_depth": random.choice([3, 5, 7]),
        "subsample": random.choice([0.7, 0.8, 1.0]),
    },
    "SVM": lambda: {
        "C": random.choice([0.1, 1.0, 10.0, 100.0]),
        "kernel": random.choice(["rbf", "linear", "poly"]),
        "gamma": random.choice(["scale", "auto"]),
    },
    "KNN": lambda: {
        "n_neighbors": random.choice([3, 5, 7, 11, 15]),
        "weights": random.choice(["uniform", "distance"]),
    },
    "XGBoost": lambda: {
        "n_estimators": random.choice([100, 200, 300, 500]),
        "learning_rate": random.choice([0.01, 0.05, 0.1, 0.2]),
        "max_depth": random.choice([3, 5, 6, 8]),
        "subsample": random.choice([0.7, 0.8, 1.0]),
        "colsample_bytree": random.choice([0.6, 0.8, 1.0]),
    },
    "LightGBM": lambda: {
        "n_estimators": random.choice([100, 200, 300, 500]),
        "learning_rate": random.choice([0.01, 0.05, 0.1, 0.2]),
        "num_leaves": random.choice([31, 63, 127]),
    },
}


def _print_progress(elapsed: float, budget: float, iteration: int, best_score: float, metric: str):
    """Print a single-line progress bar that updates in place."""
    pct = min(elapsed / budget, 1.0)
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    remaining = max(0, budget - elapsed)
    score_str = f"{metric}={best_score:.4f}" if best_score > -np.inf else "no result yet"
    line = ((f"  ⏱  [{bar}] {elapsed:.0f}s/{budget:.0f}s  "
            f"iter={iteration}  best={score_str}  remaining={remaining:.0f}s").ljust(100))
    sys.stdout.write(f"\r{line}")
    sys.stdout.flush()


def _train_and_score(name, model, X_train, y_train, X_val, y_val,
                     metric: str, deadline: float):
    """Train a single model and return (name, model, score, time). Thread-safe."""
    if time.time() > deadline:
        return name, None, -np.inf, 0.0
    t0 = time.time()
    try:
        # Fit with early stopping for XGB/LGBM if val set provided
        if hasattr(model, "early_stopping_rounds") and X_val is not None:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)

        preds = model.predict(X_val)
        score = _compute_metric(metric, y_val, preds, model, X_val)
        elapsed = time.time() - t0
        return name, model, score, elapsed
    except Exception as e:
        logger.warning(f"  Model '{name}' failed: {e}")
        return name, None, -np.inf, 0.0


def _rebuild_model(base_model, override_params: dict):
    """Safely create a new model instance with updated hyper-params.
    Strips constructor-incompatible keys (e.g. XGBoost early_stopping_rounds).
    """
    # Keys that are valid fit() args but not __init__ params for some estimators
    _STRIP_KEYS = {"early_stopping_rounds", "eval_metric", "eval_set",
                   "use_label_encoder", "verbose"}
    base_params = {k: v for k, v in base_model.get_params().items()
                   if k not in _STRIP_KEYS}
    merged = {**base_params, **override_params}
    # Remove keys that are not valid constructor params
    import inspect
    valid = set(inspect.signature(base_model.__class__.__init__).parameters.keys()) - {"self"}
    merged = {k: v for k, v in merged.items() if k in valid}
    return base_model.__class__(**merged)


def _resolve_avg(y_true) -> str:
    """Use 'binary' only for integer {0,1} labels; otherwise 'weighted'."""
    unique = set(np.unique(y_true))
    if len(unique) == 2 and unique in ({0, 1}, {True, False}, {0.0, 1.0}):
        return "binary"
    return "weighted"


def _compute_metric(metric: str, y_true, y_pred, model, X_val):
    if metric == "f1":
        return f1_score(y_true, y_pred, average=_resolve_avg(y_true), zero_division=0)
    elif metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif metric == "roc_auc":
        try:
            from sklearn.preprocessing import LabelBinarizer
            lb = LabelBinarizer()
            y_bin = lb.fit_transform(y_true).ravel()
            proba = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_bin, proba)
        except Exception:
            return accuracy_score(y_true, y_pred)
    elif metric == "rmse":
        return -np.sqrt(mean_squared_error(y_true, y_pred))  # negative for maximization
    elif metric == "r2":
        return r2_score(y_true, y_pred)
    elif metric == "mae":
        return -mean_absolute_error(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")


class AutoClassifier:
    """
    Automatically trains multiple classifiers and selects the best one.

    Parameters
    ----------
    metric      : 'f1', 'accuracy', 'roc_auc' (default: 'f1')
    time_limit  : Maximum seconds for training (default: 120)
    dataset_tier: 'small'|'medium'|'large' — controls which models are tried
                  (auto-detected from data if not set)
    cv          : Cross-validation folds (0 = use val split instead)

    Example
    -------
    >>> clf = AutoClassifier(metric="f1", time_limit=60)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    >>> clf.leaderboard()
    """

    def __init__(self, metric: str = "f1", time_limit: int = 120,
                 dataset_tier: str = "auto", n_jobs: int = -1):
        self.metric = metric
        self.time_limit = time_limit
        self.dataset_tier = dataset_tier
        self.n_jobs = n_jobs
        self.best_model_ = None
        self.best_model_name_ = None
        self.best_score_ = -np.inf
        self._leaderboard = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train models iteratively using random search until the time budget is exhausted."""
        n_samples = len(X_train)
        tier = self.dataset_tier
        if tier == "auto":
            if n_samples < 10_000:
                tier = "small"
            elif n_samples < 500_000:
                tier = "medium"
            else:
                tier = "large"

        logger.info(f"AutoClassifier | metric={self.metric} | tier={tier} | budget={self.time_limit}s")

        # Encode target to numeric (required for XGBoost/LightGBM)
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder_ = LabelEncoder()
        y_encoded = pd.Series(self.label_encoder_.fit_transform(y_train), index=getattr(y_train, 'index', None))
        
        # Auto val split if not provided
        if X_val is None:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train_enc, y_val_enc = train_test_split(
                X_train, y_encoded, test_size=0.2, random_state=42,
                stratify=y_encoded if y_encoded.nunique() <= 20 else None
            )
        else:
            y_train_enc = y_encoded
            y_val_enc = pd.Series(self.label_encoder_.transform(y_val), index=getattr(y_val, 'index', None))

        base_candidates = ModelOptimizer("classification", tier).get_candidate_models()
        start_time = time.time()
        deadline = start_time + self.time_limit
        param_space = _CLASSIFICATION_PARAM_SPACE

        best_per_model: dict = {}   # name -> (model, score)
        iteration = 0

        logger.info(f"  Searching {len(base_candidates)} models via random hyper-param search (budget={self.time_limit}s)...")
        print(f"  Running random hyper-parameter search for {self.time_limit}s — please wait…")

        while time.time() < deadline:
            iteration += 1
            elapsed = time.time() - start_time
            _print_progress(elapsed, self.time_limit, iteration, self.best_score_, self.metric)

            # Build fresh candidates with random hyper-params
            iter_candidates = {}
            for name, base_model in base_candidates.items():
                try:
                    params = param_space.get(name, lambda: {})()
                    iter_candidates[name] = _rebuild_model(base_model, params)
                except Exception:
                    iter_candidates[name] = base_model

            results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_train_and_score)(
                    name, model, X_train, y_train_enc, X_val, y_val_enc, self.metric, deadline
                )
                for name, model in iter_candidates.items()
            )

            for name, model, score, elapsed_t in results:
                if model is not None:
                    prev_score, _ = best_per_model.get(name, (-np.inf, None))
                    if score > prev_score:
                        best_per_model[name] = (score, model)
                    if score > self.best_score_:
                        self.best_score_ = score
                        self.best_model_ = model
                        self.best_model_name_ = name

        sys.stdout.write("\n")  # end progress line

        # Build final leaderboard from best seen per model
        self._leaderboard = [
            {"model": name, "score": round(score, 4), "time_s": round(self.time_limit, 1)}
            for name, (score, _) in sorted(best_per_model.items(), key=lambda x: x[1][0], reverse=True)
        ]

        reason = ModelOptimizer.recommend_reason(self.best_model_name_, {})
        logger.info(f"\n  ✅ Best: {self.best_model_name_} | {self.metric}={self.best_score_:.4f}")
        logger.info(f"  Reason: {reason}")
        return self

    def predict(self, X):
        self._check_fitted()
        preds = self.best_model_.predict(X)
        if hasattr(self, 'label_encoder_'):
            return self.label_encoder_.inverse_transform(preds)
        return preds

    def predict_proba(self, X):
        self._check_fitted()
        if hasattr(self.best_model_, "predict_proba"):
            return self.best_model_.predict_proba(X)
        raise AttributeError(f"{self.best_model_name_} does not support predict_proba")

    def score(self, X, y):
        self._check_fitted()
        return _compute_metric(self.metric, y, self.predict(X), self.best_model_, X)

    def leaderboard(self) -> pd.DataFrame:
        """Return sorted leaderboard of all trained models."""
        df = pd.DataFrame(self._leaderboard)
        print("\n  🏆 AutoClassifier Leaderboard")
        print("  " + "─" * 50)
        for i, row in df.iterrows():
            marker = " ✅ BEST" if i == 0 else ""
            print(f"  {row['model']:<22} {self.metric}={row['score']:.4f}  {row['time_s']:.1f}s{marker}")
        print("  " + "─" * 50 + "\n")
        return df

    def _check_fitted(self):
        if self.best_model_ is None:
            raise RuntimeError("Call .fit() first.")


class AutoRegressor:
    """
    Automatically trains multiple regressors and selects the best one.

    Parameters
    ----------
    metric      : 'rmse', 'r2', 'mae' (default: 'rmse')
    time_limit  : Maximum seconds for training
    dataset_tier: 'small'|'medium'|'large'

    Example
    -------
    >>> reg = AutoRegressor(metric="rmse", time_limit=60)
    >>> reg.fit(X_train, y_train)
    >>> predictions = reg.predict(X_test)
    """

    def __init__(self, metric: str = "rmse", time_limit: int = 120,
                 dataset_tier: str = "auto", n_jobs: int = -1):
        self.metric = metric
        self.time_limit = time_limit
        self.dataset_tier = dataset_tier
        self.n_jobs = n_jobs
        self.best_model_ = None
        self.best_model_name_ = None
        self.best_score_ = -np.inf
        self._leaderboard = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train models iteratively using random search until the time budget is exhausted."""
        n_samples = len(X_train)
        tier = self.dataset_tier
        if tier == "auto":
            tier = "small" if n_samples < 10_000 else ("medium" if n_samples < 500_000 else "large")

        logger.info(f"AutoRegressor | metric={self.metric} | tier={tier} | budget={self.time_limit}s")

        if X_val is None:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )

        base_candidates = ModelOptimizer("regression", tier).get_candidate_models()
        start_time = time.time()
        deadline = start_time + self.time_limit
        param_space = _REGRESSION_PARAM_SPACE

        best_per_model: dict = {}   # name -> (score, model)
        iteration = 0

        logger.info(f"  Searching {len(base_candidates)} models via random hyper-param search (budget={self.time_limit}s)...")
        print(f"  Running random hyper-parameter search for {self.time_limit}s — please wait…")

        while time.time() < deadline:
            iteration += 1
            elapsed = time.time() - start_time
            _print_progress(elapsed, self.time_limit, iteration, self.best_score_, self.metric)

            # Build fresh candidates with random hyper-params
            iter_candidates = {}
            for name, base_model in base_candidates.items():
                try:
                    params = param_space.get(name, lambda: {})()
                    model_copy = base_model.__class__(**{**base_model.get_params(), **params})
                    iter_candidates[name] = model_copy
                except Exception:
                    iter_candidates[name] = base_model

            results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_train_and_score)(
                    name, model, X_train, y_train, X_val, y_val, self.metric, deadline
                )
                for name, model in iter_candidates.items()
            )

            for name, model, score, elapsed_t in results:
                if model is not None:
                    prev_score, _ = best_per_model.get(name, (-np.inf, None))
                    if score > prev_score:
                        best_per_model[name] = (score, model)
                    if score > self.best_score_:
                        self.best_score_ = score
                        self.best_model_ = model
                        self.best_model_name_ = name

        sys.stdout.write("\n")  # end progress line

        # Build final leaderboard from best seen per model
        self._leaderboard = [
            {"model": name, "score": round(score, 4), "time_s": round(self.time_limit, 1)}
            for name, (score, _) in sorted(best_per_model.items(), key=lambda x: x[1][0], reverse=True)
        ]

        logger.info(f"\n  ✅ Best: {self.best_model_name_} | {self.metric}={self.best_score_:.4f}")
        return self

    def predict(self, X):
        if self.best_model_ is None:
            raise RuntimeError("Call .fit() first.")
        return self.best_model_.predict(X)

    def score(self, X, y):
        return _compute_metric(self.metric, y, self.predict(X), self.best_model_, X)

    def leaderboard(self) -> pd.DataFrame:
        df = pd.DataFrame(self._leaderboard)
        print("\n  🏆 AutoRegressor Leaderboard")
        print("  " + "─" * 50)
        for i, row in df.iterrows():
            marker = " ✅ BEST" if i == 0 else ""
            print(f"  {row['model']:<22} {self.metric}={row['score']:.4f}  {row['time_s']:.1f}s{marker}")
        print("  " + "─" * 50 + "\n")
        return df


class AutoClusterer:
    """
    Automatic clustering with optimal K selection via silhouette score.

    Parameters
    ----------
    algorithm : 'kmeans' | 'dbscan'
    max_k     : Maximum clusters to try (for KMeans)

    Example
    -------
    >>> clusterer = AutoClusterer(algorithm="kmeans", max_k=10)
    >>> clusterer.fit(X)
    >>> print(clusterer.best_k)
    >>> labels = clusterer.labels_
    """

    def __init__(self, algorithm: str = "kmeans", max_k: int = 10):
        self.algorithm = algorithm
        self.max_k = max_k
        self.best_k = None
        self.best_model_ = None
        self.labels_ = None
        self._scores = {}

    def fit(self, X):
        logger.info(f"AutoClusterer | algorithm={self.algorithm} | max_k={self.max_k}")

        if self.algorithm == "kmeans":
            best_score = -np.inf
            for k in range(2, self.max_k + 1):
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(X)
                score = silhouette_score(X, labels)
                self._scores[k] = round(score, 4)
                logger.info(f"  k={k} | silhouette={score:.4f}")
                if score > best_score:
                    best_score = score
                    self.best_k = k
                    self.best_model_ = model
                    self.labels_ = labels

            logger.info(f"  ✅ Best k={self.best_k} | silhouette={best_score:.4f}")

        elif self.algorithm == "dbscan":
            model = DBSCAN(eps=0.5, min_samples=5)
            self.labels_ = model.fit_predict(X)
            self.best_model_ = model
            n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
            logger.info(f"  DBSCAN found {n_clusters} clusters")

        return self

    def predict(self, X):
        if self.best_model_ is None:
            raise RuntimeError("Call .fit() first.")
        return self.best_model_.predict(X)

    def elbow_scores(self) -> dict:
        return self._scores
