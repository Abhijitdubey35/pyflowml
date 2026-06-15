"""
DataSplitter — Train/validation/test splitting with stratification support.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pyflowml.monitoring.logger import get_logger

logger = get_logger("DataSplitter")


class DataSplitter:
    """
    Flexible train/val/test splitting.

    Supports:
    - Standard train/test
    - Train/val/test (three-way)
    - Stratified splits for classification

    Example
    -------
    >>> splitter = DataSplitter(df, target="Survived")
    >>> X_train, X_test, y_train, y_test = splitter.split()

    >>> # Three-way split
    >>> X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_three_way()
    """

    def __init__(self, df: pd.DataFrame, target: str,
                 test_size: float = 0.2, val_size: float = 0.0,
                 stratify: bool = True, random_state: int = 42):
        self.df = df
        self.target = target
        self.test_size = test_size
        self.val_size = val_size
        self.stratify = stratify
        self.random_state = random_state

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

    def split(self):
        """
        Standard train/test split.

        Returns
        -------
        X_train, X_test, y_train, y_test
        """
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        strat = y if self.stratify and y.nunique() <= 20 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=strat,
            random_state=self.random_state,
        )

        logger.info(f"Split → Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    def split_three_way(self):
        """
        Three-way train/val/test split.

        Returns
        -------
        X_train, X_val, X_test, y_train, y_val, y_test
        """
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        strat = y if self.stratify and y.nunique() <= 20 else None
        val_ratio = self.val_size / (1.0 - self.test_size)

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=strat,
            random_state=self.random_state,
        )

        strat2 = y_train_val if self.stratify and y_train_val.nunique() <= 20 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_ratio,
            stratify=strat2,
            random_state=self.random_state,
        )

        logger.info(f"Split → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
