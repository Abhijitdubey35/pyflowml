"""
DataCleaner — Handles nulls, outliers, duplicates, and encoding.
Supports manual calls and automatic recommendation-based cleaning.
"""

import numpy as np
import pandas as pd
from pyflowml.monitoring.logger import get_logger

logger = get_logger("DataCleaner")


class DataCleaner:
    """
    Smart data cleaning with chainable methods.

    Example
    -------
    >>> cleaner = DataCleaner(df)
    >>> df_clean = (cleaner
    ...     .handle_nulls(strategy="median")
    ...     .remove_outliers(method="iqr")
    ...     .remove_duplicates()
    ...     .result())
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()
        self._log = []

    def handle_nulls(self, strategy: str = "auto", columns: list = None) -> "DataCleaner":
        """
        Handle missing values.

        Parameters
        ----------
        strategy : 'auto', 'median', 'mean', 'mode', 'drop_rows', 'ffill', 'bfill'
        columns  : Specific columns to apply (default: all with nulls)
        """
        cols = columns or [c for c in self._df.columns if self._df[c].isna().any()]
        for col in cols:
            null_count = self._df[col].isna().sum()
            if null_count == 0:
                continue

            if strategy == "auto":
                col_strategy = self._auto_strategy(col)
            else:
                col_strategy = strategy

            self._apply_null_strategy(col, col_strategy)
            self._log.append(f"Nulls in '{col}': {null_count} → {col_strategy}")

        return self

    def _auto_strategy(self, col: str) -> str:
        ratio = self._df[col].isna().sum() / len(self._df)
        dtype = self._df[col].dtype
        if ratio > 0.5:
            return "drop_column"
        elif pd.api.types.is_numeric_dtype(dtype):
            return "median"
        else:
            return "mode"

    def _apply_null_strategy(self, col: str, strategy: str):
        if strategy == "drop_column":
            self._df.drop(columns=[col], inplace=True)
        elif strategy == "drop_rows":
            self._df.dropna(subset=[col], inplace=True)
        elif strategy == "median":
            self._df[col].fillna(self._df[col].median(), inplace=True)
        elif strategy == "mean":
            self._df[col].fillna(self._df[col].mean(), inplace=True)
        elif strategy in ("mode", "impute_mode"):
            mode_val = self._df[col].mode()
            if len(mode_val) > 0:
                self._df[col].fillna(mode_val[0], inplace=True)
        elif strategy in ("impute_median",):
            self._df[col].fillna(self._df[col].median(), inplace=True)
        elif strategy == "ffill":
            self._df[col].fillna(method="ffill", inplace=True)
        elif strategy == "bfill":
            self._df[col].fillna(method="bfill", inplace=True)

    def remove_outliers(self, method: str = "iqr", columns: list = None,
                        multiplier: float = 1.5) -> "DataCleaner":
        """
        Remove rows with outliers using IQR method.

        Parameters
        ----------
        method     : 'iqr' (only method currently supported)
        columns    : Numeric columns to check (default: all)
        multiplier : IQR multiplier for bounds (default 1.5)
        """
        if method != "iqr":
            raise ValueError("Only 'iqr' method is currently supported.")
        numeric_cols = columns or self._df.select_dtypes(include=[np.number]).columns.tolist()
        before = len(self._df)
        for col in numeric_cols:
            Q1 = self._df[col].quantile(0.25)
            Q3 = self._df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - multiplier * IQR
            upper = Q3 + multiplier * IQR
            self._df = self._df[(self._df[col] >= lower) & (self._df[col] <= upper)]
        self._df.reset_index(drop=True, inplace=True)
        removed = before - len(self._df)
        self._log.append(f"Outlier removal: {removed} rows removed via IQR")
        return self

    def remove_duplicates(self) -> "DataCleaner":
        """Remove duplicate rows."""
        before = len(self._df)
        self._df.drop_duplicates(inplace=True)
        self._df.reset_index(drop=True, inplace=True)
        removed = before - len(self._df)
        self._log.append(f"Duplicates removed: {removed}")
        return self

    def encode_categoricals(self, method: str = "label") -> "DataCleaner":
        """
        Encode categorical columns.

        Parameters
        ----------
        method : 'label' (label encoding) or 'onehot' (one-hot encoding)
        """
        cat_cols = self._df.select_dtypes(include=["object", "category"]).columns.tolist()
        if method == "label":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            for col in cat_cols:
                self._df[col] = le.fit_transform(self._df[col].astype(str))
                self._log.append(f"Label encoded: '{col}'")
        elif method == "onehot":
            self._df = pd.get_dummies(self._df, columns=cat_cols, drop_first=True)
            self._log.append(f"One-hot encoded: {cat_cols}")
        return self

    def normalize_text(self) -> "DataCleaner":
        """Normalize string column casing to handle inconsistencies (e.g. 'PUblic' vs 'public')."""
        cat_cols = self._df.select_dtypes(include=["object", "string"]).columns.tolist()
        count = 0
        for col in cat_cols:
            # Check if there are actual strings in the column before modifying
            if self._df[col].apply(lambda x: isinstance(x, str)).any():
                self._df[col] = self._df[col].astype(str).str.title()
                count += 1
        if count > 0:
            self._log.append(f"Normalized text casing for {count} columns")
        return self

    def drop_column(self, col: str) -> "DataCleaner":
        """Drop a specific column."""
        if col in self._df.columns:
            self._df.drop(columns=[col], inplace=True)
            self._log.append(f"Dropped column: '{col}'")
        return self

    def apply_recommendations(self, recommendations: dict) -> "DataCleaner":
        """
        Apply actions from DataProfiler recommendations dict automatically.

        Parameters
        ----------
        recommendations : dict of {column: action_string}
        """
        for col, action in recommendations.items():
            if col not in self._df.columns:
                continue
            if "drop_column" in action:
                self.drop_column(col)
            elif "impute_median" in action:
                self._apply_null_strategy(col, "impute_median")
            elif "impute_mode" in action:
                self._apply_null_strategy(col, "impute_mode")
            if "log_transform" in action and col in self._df.columns:
                self._df[col] = np.log1p(self._df[col].clip(lower=0))
                self._log.append(f"Log-transformed: '{col}'")
        return self

    def result(self) -> pd.DataFrame:
        """Return the cleaned DataFrame."""
        logger.info("DataCleaner Summary:")
        for entry in self._log:
            logger.info(f"  • {entry}")
        return self._df.copy()
