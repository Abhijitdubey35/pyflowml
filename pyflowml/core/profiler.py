"""
Data Profiler — Analyzes datasets and detects problem type, data issues,
and recommended preprocessing strategies automatically.
"""

import numpy as np
import pandas as pd
from pyflowml.monitoring.logger import get_logger

logger = get_logger("DataProfiler")


class DataProfiler:
    """
    Automatic dataset intelligence.

    Detects:
    - Problem type (classification vs regression)
    - Missing value ratios per column
    - Skewness of numeric features
    - Outlier presence
    - Cardinality of categorical columns
    - Recommended preprocessing actions per column

    Example
    -------
    >>> profiler = DataProfiler(df, target="price")
    >>> profile = profiler.run()
    >>> print(profile["problem_type"])
    'regression'
    >>> print(profile["recommended_actions"])
    {'age': 'impute_median', 'cabin': 'drop_column', ...}
    """

    # Thresholds
    CLASSIFICATION_UNIQUE_THRESHOLD = 20
    HIGH_MISSING_THRESHOLD = 0.50
    MEDIUM_MISSING_THRESHOLD = 0.05
    HIGH_SKEW_THRESHOLD = 1.0
    HIGH_CARDINALITY_THRESHOLD = 50
    OUTLIER_IQR_MULTIPLIER = 1.5
    HIGH_CORRELATION_THRESHOLD = 0.90

    def __init__(self, df: pd.DataFrame, target: str):
        self.df = df.copy()
        self.target = target
        self._profile = {}

    def run(self) -> dict:
        """Run full dataset profiling and return profile dict."""
        logger.info("Profiling dataset...")
        self._profile["shape"] = self.df.shape
        self._profile["problem_type"] = self._detect_problem_type()
        self._profile["missing"] = self._analyze_missing()
        self._profile["skewness"] = self._analyze_skewness()
        self._profile["outliers"] = self._detect_outliers()
        self._profile["cardinality"] = self._analyze_cardinality()
        self._profile["high_correlation_pairs"] = self._find_high_correlation()
        self._profile["recommended_actions"] = self._build_recommendations()
        self._profile["dataset_tier"] = self._get_dataset_tier()
        return self._profile

    def _detect_problem_type(self) -> str:
        y = self.df[self.target]
        if y.dtype == "object" or y.nunique() <= self.CLASSIFICATION_UNIQUE_THRESHOLD:
            return "classification"
        return "regression"

    def _analyze_missing(self) -> dict:
        total = len(self.df)
        missing = {}
        for col in self.df.columns:
            ratio = self.df[col].isna().sum() / total
            if ratio > 0:
                missing[col] = round(ratio, 4)
        return missing

    def _analyze_skewness(self) -> dict:
        skewness = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target in numeric_cols:
            numeric_cols.remove(self.target)
        for col in numeric_cols:
            skew = abs(self.df[col].skew())
            if skew > self.HIGH_SKEW_THRESHOLD:
                skewness[col] = round(skew, 3)
        return skewness

    def _detect_outliers(self) -> dict:
        outlier_cols = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target in numeric_cols:
            numeric_cols.remove(self.target)
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - self.OUTLIER_IQR_MULTIPLIER * IQR
            upper = Q3 + self.OUTLIER_IQR_MULTIPLIER * IQR
            n_outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            if n_outliers > 0:
                outlier_cols[col] = int(n_outliers)
        return outlier_cols

    def _analyze_cardinality(self) -> dict:
        cardinality = {}
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        if self.target in cat_cols:
            cat_cols.remove(self.target)
        for col in cat_cols:
            cardinality[col] = self.df[col].nunique()
        return cardinality

    def _find_high_correlation(self) -> list:
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target in numeric_cols:
            numeric_cols.remove(self.target)
        if len(numeric_cols) < 2:
            return []
        corr_matrix = self.df[numeric_cols].corr().abs()
        pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.HIGH_CORRELATION_THRESHOLD:
                    pairs.append(
                        (corr_matrix.columns[i], corr_matrix.columns[j],
                         round(corr_matrix.iloc[i, j], 3))
                    )
        return pairs

    def _build_recommendations(self) -> dict:
        actions = {}
        missing = self._profile.get("missing", {})
        skewness = self._profile.get("skewness", {})
        cardinality = self._profile.get("cardinality", {})

        for col, ratio in missing.items():
            if col == self.target:
                continue
            if ratio > self.HIGH_MISSING_THRESHOLD:
                actions[col] = "drop_column"
            else:
                dtype = self.df[col].dtype
                if np.issubdtype(dtype, np.number):
                    actions[col] = "impute_median"
                else:
                    actions[col] = "impute_mode"

        for col in skewness:
            current = actions.get(col, "")
            if "drop" not in current:
                actions[col] = (current + " + log_transform").lstrip(" + ")

        for col, n_unique in cardinality.items():
            if n_unique > self.HIGH_CARDINALITY_THRESHOLD:
                actions[col] = "target_encoding"
            elif col not in actions:
                actions[col] = "onehot_encoding"

        # Correlated pairs — drop one
        for col_a, col_b, corr_val in self._profile.get("high_correlation_pairs", []):
            if col_a not in actions:
                actions[col_a] = f"drop_column (corr={corr_val} with {col_b})"

        return actions

    def _get_dataset_tier(self) -> str:
        n = self.df.shape[0]
        if n < 10_000:
            return "small"
        elif n < 500_000:
            return "medium"
        else:
            return "large"

    def report(self):
        """Print a human-readable profile report."""
        p = self._profile
        print("\n" + "─" * 55)
        print("  📊 PyFlowML — Dataset Profile Report")
        print("─" * 55)
        print(f"  Shape         : {p.get('shape', 'N/A')}")
        print(f"  Problem Type  : {p.get('problem_type', 'N/A')}")
        print(f"  Dataset Tier  : {p.get('dataset_tier', 'N/A')}")

        print(f"\n  Missing Values ({len(p.get('missing', {}))}):")
        for col, ratio in p.get("missing", {}).items():
            print(f"    {col:<25} {ratio*100:.1f}%")

        print(f"\n  Skewed Features ({len(p.get('skewness', {}))}):")
        for col, skew in p.get("skewness", {}).items():
            print(f"    {col:<25} skew={skew}")

        print(f"\n  High Correlation Pairs:")
        for a, b, c in p.get("high_correlation_pairs", []):
            print(f"    {a} ↔ {b} = {c}")

        print(f"\n  Recommended Actions:")
        for col, action in p.get("recommended_actions", {}).items():
            print(f"    {col:<25} → {action}")
        print("─" * 55 + "\n")
