"""
StepTracker — Tracks time and memory usage per pipeline step.
"""

import time
import tracemalloc
import psutil
import os
from pyflowml.monitoring.logger import get_logger

logger = get_logger("Tracker")


class StepTracker:
    """
    Context manager that tracks wall-clock time and memory for a pipeline step.

    Example
    -------
    >>> with StepTracker("Preprocessing") as t:
    ...     X = pipeline.fit_transform(X)
    >>> print(t.elapsed_s, t.memory_delta_mb)
    """

    def __init__(self, step_name: str):
        self.step_name = step_name
        self.elapsed_s = 0.0
        self.memory_delta_mb = 0.0
        self._start_time = None
        self._mem_before = 0.0

    def __enter__(self):
        self._start_time = time.time()
        process = psutil.Process(os.getpid())
        self._mem_before = process.memory_info().rss / 1024 ** 2
        logger.info(f"▶  {self.step_name}...")
        return self

    def __exit__(self, *args):
        self.elapsed_s = time.time() - self._start_time
        process = psutil.Process(os.getpid())
        mem_after = process.memory_info().rss / 1024 ** 2
        self.memory_delta_mb = mem_after - self._mem_before
        sign = "+" if self.memory_delta_mb >= 0 else ""
        logger.info(
            f"✔  {self.step_name} | {self.elapsed_s:.2f}s | "
            f"RAM {sign}{self.memory_delta_mb:.1f} MB"
        )

    def report(self) -> str:
        return (f"Step: {self.step_name} | "
                f"Time: {self.elapsed_s:.2f}s | "
                f"Memory: {self.memory_delta_mb:+.1f} MB")
