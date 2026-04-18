"""
CrossValidator — k-fold and stratified k-fold cross-validation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold
)
from pyflowml.monitoring.logger import get_logger

logger = get_logger("CrossValidator")


class CrossValidator:
    """
    Run k-fold cross-validation with statistics summary.

    Example
    -------
    >>> cv = CrossValidator(clf, cv=5, metric="f1")
    >>> results = cv.run(X, y)
    >>> results.summary()
    """

    SKLEARN_METRICS = {
        "f1": "f1_weighted",
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    }

    def __init__(self, model, cv: int = 5, metric: str = "f1",
                 stratified: bool = True, random_state: int = 42):
        self.model = model
        self.cv = cv
        self.metric = metric
        self.stratified = stratified
        self.random_state = random_state
        self._scores = None

    def run(self, X, y) -> "CrossValidator":
        """Run cross-validation."""
        scoring = self.SKLEARN_METRICS.get(self.metric, self.metric)

        if self.stratified and hasattr(y, "nunique") and y.nunique() <= 20:
            splitter = StratifiedKFold(n_splits=self.cv, shuffle=True,
                                       random_state=self.random_state)
        else:
            splitter = KFold(n_splits=self.cv, shuffle=True,
                             random_state=self.random_state)

        # If model is AutoClassifier/AutoRegressor, get inner model
        model = getattr(self.model, "best_model_", self.model)

        scores = cross_val_score(model, X, y, cv=splitter, scoring=scoring, n_jobs=-1)
        # Convert negative scores to positive
        if "neg_" in scoring:
            scores = -scores
        self._scores = scores
        logger.info(f"CV {self.cv}-fold | {self.metric}: {scores.mean():.4f} ± {scores.std():.4f}")
        return self

    def summary(self):
        """Print formatted CV summary."""
        if self._scores is None:
            raise RuntimeError("Call .run() first.")
        print("\n" + "─" * 45)
        print(f"  Cross-Validation ({self.cv}-fold)")
        print("─" * 45)
        for i, s in enumerate(self._scores, 1):
            print(f"  Fold {i}: {s:.4f}")
        print("─" * 45)
        print(f"  Mean  : {self._scores.mean():.4f}")
        print(f"  Std   : {self._scores.std():.4f}")
        print(f"  Min   : {self._scores.min():.4f}")
        print(f"  Max   : {self._scores.max():.4f}")
        print("─" * 45 + "\n")

    @property
    def scores(self):
        return self._scores
