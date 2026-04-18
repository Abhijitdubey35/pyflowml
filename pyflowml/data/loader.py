"""
DataLoader — Load data from CSV, JSON, Excel, and URLs.
Supports chunked loading and optional Dask backend for large files.
"""

import pandas as pd
from pyflowml.monitoring.logger import get_logger

logger = get_logger("DataLoader")


class DataLoader:
    """
    Unified data loading interface supporting multiple formats.

    Supports chunked loading (memory-safe) and optional Dask backend
    for datasets that don't fit in memory.

    Example
    -------
    >>> df = DataLoader.from_csv("titanic.csv")
    >>> df = DataLoader.from_url("https://example.com/data.csv")
    >>> df = DataLoader.from_excel("sales.xlsx", sheet="Q1")
    """

    @staticmethod
    def from_csv(path: str, chunksize: int = None, use_dask: bool = False, **kwargs) -> pd.DataFrame:
        """
        Load a CSV file.

        Parameters
        ----------
        path      : Path to the CSV file
        chunksize : If set, load in chunks and concatenate (memory-safe)
        use_dask  : If True, returns a Dask DataFrame (lazy, for huge files)
        """
        logger.info(f"Loading CSV: {path}")
        if use_dask:
            try:
                import dask.dataframe as dd
                df = dd.read_csv(path, **kwargs)
                logger.info(f"  Loaded with Dask backend")
                return df
            except ImportError:
                logger.warning("Dask not installed. Falling back to pandas.")

        if chunksize:
            chunks = []
            for i, chunk in enumerate(pd.read_csv(path, chunksize=chunksize, **kwargs)):
                chunks.append(chunk)
                logger.info(f"  Loaded chunk {i+1} ({len(chunk)} rows)")
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(path, **kwargs)

        logger.info(f"  Shape: {df.shape}")
        return df

    @staticmethod
    def from_json(path: str, **kwargs) -> pd.DataFrame:
        """Load a JSON file."""
        logger.info(f"Loading JSON: {path}")
        df = pd.read_json(path, **kwargs)
        logger.info(f"  Shape: {df.shape}")
        return df

    @staticmethod
    def from_excel(path: str, sheet: str = 0, **kwargs) -> pd.DataFrame:
        """Load an Excel file."""
        logger.info(f"Loading Excel: {path} (sheet={sheet})")
        df = pd.read_excel(path, sheet_name=sheet, **kwargs)
        logger.info(f"  Shape: {df.shape}")
        return df

    @staticmethod
    def from_url(url: str, chunksize: int = None, **kwargs) -> pd.DataFrame:
        """Load a CSV from a URL."""
        logger.info(f"Fetching URL: {url}")
        return DataLoader.from_csv(url, chunksize=chunksize, **kwargs)

    @staticmethod
    def from_dict(data: dict) -> pd.DataFrame:
        """Create a DataFrame from a Python dict."""
        return pd.DataFrame(data)

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Pass-through for already-loaded DataFrames."""
        return df.copy()

    @staticmethod
    def load_chunks(path: str, chunksize: int = 10_000, **kwargs):
        """
        Generator: yields DataFrame chunks for manual processing.

        Example
        -------
        >>> for chunk in DataLoader.load_chunks("big.csv", chunksize=50000):
        ...     process(chunk)
        """
        for chunk in pd.read_csv(path, chunksize=chunksize, **kwargs):
            yield chunk
