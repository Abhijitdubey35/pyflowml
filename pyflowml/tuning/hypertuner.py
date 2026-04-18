"""
HyperTuner — Grid search, random search, and Bayesian hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from pyflowml.monitoring.logger import get_logger

logger = get_logger("HyperTuner")


class HyperTuner:
    """
    Hyperparameter tuning wrapper with tqdm progress display.

    Parameters
    ----------
    model       : A fitted or unfitted sklearn-compatible model
    param_grid  : Dict of hyperparameter search space
    method      : 'grid' | 'random'
    metric      : Scoring metric (default: 'f1_weighted')
    n_trials    : Number of random trials (for 'random' method)
    cv          : Cross-validation folds

    Example
    -------
    >>> tuner = HyperTuner(clf, param_grid, method="random", n_trials=20)
    >>> tuner.fit(X_train, y_train)
    >>> print(tuner.best_params)
    >>> best_model = tuner.best_model
    """

    METRIC_MAP = {
        "f1": "f1_weighted",
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
    }

    def __init__(self, model, param_grid: dict, method: str = "random",
                 metric: str = "f1", n_trials: int = 20, cv: int = 5):
        # Unwrap AutoClassifier/AutoRegressor
        self.model = getattr(model, "best_model_", model)
        self.param_grid = param_grid
        self.method = method
        self.metric = metric
        self.n_trials = n_trials
        self.cv = cv
        self._search = None

    def fit(self, X_train, y_train) -> "HyperTuner":
        """Run the hyperparameter search."""
        scoring = self.METRIC_MAP.get(self.metric, self.metric)
        logger.info(f"HyperTuner | method={self.method} | metric={scoring} | cv={self.cv}")

        if self.method == "grid":
            self._search = GridSearchCV(
                self.model, self.param_grid,
                scoring=scoring, cv=self.cv,
                n_jobs=-1, verbose=1,
                refit=True,
            )
        elif self.method == "random":
            self._search = RandomizedSearchCV(
                self.model, self.param_grid,
                n_iter=self.n_trials,
                scoring=scoring, cv=self.cv,
                n_jobs=-1, verbose=1,
                random_state=42, refit=True,
            )
        else:
            raise ValueError(f"method must be 'grid' or 'random', got '{self.method}'")

        self._search.fit(X_train, y_train)
        logger.info(f"  Best score : {self._search.best_score_:.4f}")
        logger.info(f"  Best params: {self._search.best_params_}")
        return self

    @property
    def best_params(self) -> dict:
        self._check_fitted()
        return self._search.best_params_

    @property
    def best_score(self) -> float:
        self._check_fitted()
        return self._search.best_score_

    @property
    def best_model(self):
        self._check_fitted()
        return self._search.best_estimator_

    def results_dataframe(self) -> pd.DataFrame:
        """Return all trial results as a sorted DataFrame."""
        self._check_fitted()
        df = pd.DataFrame(self._search.cv_results_)
        return df.sort_values("rank_test_score")[
            ["rank_test_score", "mean_test_score", "std_test_score", "params"]
        ]

    def _check_fitted(self):
        if self._search is None:
            raise RuntimeError("Call .fit() first.")
