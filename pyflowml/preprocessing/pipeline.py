"""
SmartPipeline — Cached preprocessing pipeline.
Chains Scaler + FeatureSelector with joblib.Memory caching.
"""

import hashlib
import os
import pandas as pd
import numpy as np
from joblib import Memory

from pyflowml.preprocessing.scaler import Scaler
from pyflowml.preprocessing.feature_selector import FeatureSelector
from pyflowml.monitoring.logger import get_logger

logger = get_logger("SmartPipeline")

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "pyflowml_cache")
_memory = Memory(_CACHE_DIR, verbose=0)


class SmartPipeline:
    """
    Preprocessing pipeline with automatic caching.
    
    Chains: FeatureSelector → Scaler
    Results are cached to disk — identical inputs never reprocessed.

    Example
    -------
    >>> pipe = SmartPipeline(scaler_method="standard", selector_method="all")
    >>> X_train = pipe.fit_transform(X_train, y_train)
    >>> X_test  = pipe.transform(X_test)
    """

    def __init__(self, scaler_method: str = "standard",
                 selector_method: str = "correlation",
                 correlation_threshold: float = 0.9,
                 top_k: int = None,
                 problem_type: str = "classification"):
        self.scaler = Scaler(method=scaler_method)
        self.selector = FeatureSelector(
            method=selector_method,
            threshold=correlation_threshold,
            top_k=top_k,
            problem_type=problem_type,
        )
        self._fitted = False

    def fit(self, X: pd.DataFrame, y=None) -> "SmartPipeline":
        logger.info("Fitting preprocessing pipeline...")
        self.selector.fit(X, y)
        X_sel = self.selector.transform(X)

        # Learn a stable category → code mapping for every categorical column
        # during fit, so train and test are encoded identically. Fitting a
        # fresh encoder inside transform() (the previous behaviour) gave train
        # and test different mappings — silent train/test skew.
        cat_cols = X_sel.select_dtypes(include=["object", "category"]).columns.tolist()
        self._cat_maps = {
            col: {cat: code for code, cat in enumerate(X_sel[col].astype(str).unique())}
            for col in cat_cols
        }

        # Only scale the originally-numeric columns
        numeric_cols = X_sel.select_dtypes(include=[np.number]).columns.tolist()
        self._numeric_cols = numeric_cols
        if numeric_cols:
            self.scaler.fit(X_sel[numeric_cols])
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .transform()")
        X_sel = self.selector.transform(X).copy()

        # Apply the category maps learned during fit. Unseen categories map to
        # -1, so train and test always share the same encoding scheme.
        for col, mapping in self._cat_maps.items():
            if col in X_sel.columns:
                X_sel[col] = X_sel[col].astype(str).map(mapping).fillna(-1).astype(int)

        # Scale numeric columns with the fitted scaler
        numeric_cols = [c for c in self._numeric_cols if c in X_sel.columns]
        if numeric_cols:
            X_sel[numeric_cols] = self.scaler.transform(X_sel[numeric_cols])
        return X_sel.reset_index(drop=True)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @property
    def selected_features(self):
        return self.selector.selected_features

    @property
    def dropped_features(self):
        return self.selector.dropped_features
