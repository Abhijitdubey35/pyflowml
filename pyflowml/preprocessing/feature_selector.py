"""
FeatureSelector — Removes low-value features: correlated, constant, low-importance.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from pyflowml.monitoring.logger import get_logger

logger = get_logger("FeatureSelector")


class FeatureSelector:
    """
    Feature selection via multiple strategies.

    Methods
    -------
    - 'correlation'  : Drop features with Pearson correlation > threshold
    - 'constant'     : Drop zero-variance (constant) features
    - 'importance'   : Keep top_k features by Random Forest importance
    - 'all'          : Apply all three strategies in sequence

    Example
    -------
    >>> selector = FeatureSelector(method="correlation", threshold=0.9)
    >>> X_train = selector.fit_transform(X_train, y_train)
    >>> X_test  = selector.transform(X_test)
    """

    def __init__(self, method: str = "all", threshold: float = 0.9,
                 top_k: int = None, problem_type: str = "classification"):
        self.method = method
        self.threshold = threshold
        self.top_k = top_k
        self.problem_type = problem_type
        self._selected_features = None
        self._dropped = []

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureSelector":
        features = list(X.columns)

        if self.method in ("constant", "all"):
            features = self._drop_constant(X, features)

        if self.method in ("correlation", "all"):
            features = self._drop_correlated(X[features], features)

        if self.method in ("importance", "all") and y is not None:
            features = self._select_by_importance(X[features], y, features)

        self._selected_features = features
        dropped_count = len(X.columns) - len(features)
        logger.info(f"FeatureSelector: {len(X.columns)} → {len(features)} features ({dropped_count} dropped)")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            raise RuntimeError("Call .fit() before .transform()")
        cols = [c for c in self._selected_features if c in X.columns]
        return X[cols]

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _drop_constant(self, X: pd.DataFrame, features: list) -> list:
        keep = [c for c in features if X[c].nunique() > 1]
        dropped = set(features) - set(keep)
        if dropped:
            self._dropped.extend(list(dropped))
            logger.info(f"  Dropped constant features: {list(dropped)}")
        return keep

    def _drop_correlated(self, X: pd.DataFrame, features: list) -> list:
        numeric = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric) < 2:
            return features
        corr = X[numeric].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = {col for col in upper.columns if any(upper[col] > self.threshold)}
        keep = [c for c in features if c not in to_drop]
        if to_drop:
            self._dropped.extend(list(to_drop))
            logger.info(f"  Dropped high-corr features (>{self.threshold}): {list(to_drop)}")
        return keep

    def _select_by_importance(self, X: pd.DataFrame, y, features: list) -> list:
        if self.top_k is None or self.top_k >= len(features):
            return features
        numeric_X = X.select_dtypes(include=[np.number])
        if numeric_X.empty:
            return features
        try:
            if self.problem_type == "classification":
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(numeric_X, y)
            importances = pd.Series(model.feature_importances_, index=numeric_X.columns)
            top = importances.nlargest(self.top_k).index.tolist()
            # Keep non-numeric cols as well
            non_numeric = [c for c in features if c not in numeric_X.columns]
            selected = top + non_numeric
            dropped = set(features) - set(selected)
            if dropped:
                self._dropped.extend(list(dropped))
                logger.info(f"  Kept top {self.top_k} features by importance. Dropped: {list(dropped)}")
            return selected
        except Exception as e:
            logger.warning(f"Importance selection failed: {e}. Keeping all features.")
            return features

    @property
    def selected_features(self):
        return self._selected_features

    @property
    def dropped_features(self):
        return self._dropped
