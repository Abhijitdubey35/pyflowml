"""
Scaler — Feature scaling with StandardScaler, MinMaxScaler, RobustScaler.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from pyflowml.monitoring.logger import get_logger

logger = get_logger("Scaler")

SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


class Scaler:
    """
    Wrapper around scikit-learn scalers that preserves DataFrame column names.

    Parameters
    ----------
    method : 'standard' | 'minmax' | 'robust'

    Example
    -------
    >>> scaler = Scaler(method="standard")
    >>> X_train = scaler.fit_transform(X_train)
    >>> X_test  = scaler.transform(X_test)
    """

    def __init__(self, method: str = "standard"):
        if method not in SCALERS:
            raise ValueError(f"method must be one of {list(SCALERS.keys())}")
        self.method = method
        self._scaler = SCALERS[method]()
        self._feature_names = None

    def fit(self, X) -> "Scaler":
        self._feature_names = list(X.columns) if hasattr(X, "columns") else None
        self._scaler.fit(X)
        return self

    def transform(self, X):
        arr = self._scaler.transform(X)
        if self._feature_names is not None and hasattr(X, "columns"):
            return pd.DataFrame(arr, columns=self._feature_names, index=X.index)
        return arr

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        arr = self._scaler.inverse_transform(X)
        if self._feature_names is not None:
            return pd.DataFrame(arr, columns=self._feature_names)
        return arr
