"""
MemoryOptimizer — Reduces DataFrame memory usage by downcasting dtypes.
Typically saves 40-60% of memory with no loss of information.
"""

import numpy as np
import pandas as pd
from pyflowml.monitoring.logger import get_logger

logger = get_logger("MemoryOptimizer")


class MemoryOptimizer:
    """
    Reduces DataFrame memory usage by downcasting numeric dtypes.
    
    - int64  → int32
    - float64 → float32
    - object columns with low cardinality → category
    
    Example
    -------
    >>> df = MemoryOptimizer.reduce(df)
    >>> MemoryOptimizer.audit(df)
    """

    @staticmethod
    def reduce(df: pd.DataFrame, categorize_strings: bool = True,
               cardinality_threshold: int = 50) -> pd.DataFrame:
        """
        Downcast all numeric columns and optionally convert low-cardinality
        string columns to 'category' dtype.

        Parameters
        ----------
        df                     : Input DataFrame
        categorize_strings     : Convert low-cardinality object cols to category
        cardinality_threshold  : Max unique values for category conversion

        Returns
        -------
        pd.DataFrame with reduced memory footprint
        """
        mem_before = df.memory_usage(deep=True).sum() / 1024 ** 2
        df = df.copy()

        for col in df.columns:
            col_type = df[col].dtype

            # Downcast integers
            if col_type in ["int64", "int32"]:
                df[col] = pd.to_numeric(df[col], downcast="integer")

            # Downcast floats
            elif col_type in ["float64", "float32"]:
                df[col] = pd.to_numeric(df[col], downcast="float")

            # Convert low-cardinality strings to category
            elif col_type == "object" and categorize_strings:
                if df[col].nunique() <= cardinality_threshold:
                    df[col] = df[col].astype("category")

        mem_after = df.memory_usage(deep=True).sum() / 1024 ** 2
        saved = mem_before - mem_after
        pct = (saved / mem_before * 100) if mem_before > 0 else 0
        logger.info(f"Memory: {mem_before:.1f} MB → {mem_after:.1f} MB (saved {saved:.1f} MB, {pct:.0f}%)")
        return df

    @staticmethod
    def audit(df: pd.DataFrame) -> pd.DataFrame:
        """
        Print memory usage per column, sorted by usage descending.

        Returns
        -------
        pd.DataFrame with columns: [column, dtype, memory_mb]
        """
        usage = df.memory_usage(deep=True)
        audit_df = pd.DataFrame({
            "column": usage.index,
            "dtype": [str(df[c].dtype) if c != "Index" else "index" for c in usage.index],
            "memory_mb": (usage.values / 1024 ** 2).round(4),
        }).sort_values("memory_mb", ascending=False).reset_index(drop=True)

        print("\n  📊 Memory Audit")
        print("  " + "─" * 45)
        for _, row in audit_df.iterrows():
            print(f"  {row['column']:<25} {row['dtype']:<12} {row['memory_mb']:.4f} MB")
        total = audit_df["memory_mb"].sum()
        print("  " + "─" * 45)
        print(f"  {'TOTAL':<25} {'':12} {total:.4f} MB\n")
        return audit_df
