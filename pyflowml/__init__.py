"""
PyFlowML — Production-Grade ML Library
========================================
A scalable, intelligent Python library for the full ML lifecycle.
"""

from pyflowml.core.engine import PyFlowEngine
from pyflowml.core.profiler import DataProfiler

# Convenience imports
from pyflowml.data.loader import DataLoader
from pyflowml.data.cleaner import DataCleaner
from pyflowml.data.splitter import DataSplitter
from pyflowml.data.memory import MemoryOptimizer

from pyflowml.preprocessing.scaler import Scaler
from pyflowml.preprocessing.feature_selector import FeatureSelector
from pyflowml.preprocessing.pipeline import SmartPipeline

from pyflowml.models.auto import AutoClassifier, AutoRegressor, AutoClusterer

from pyflowml.evaluation.reporter import Reporter
from pyflowml.evaluation.cross_validate import CrossValidator

from pyflowml.visualization.plotter import Plotter
from pyflowml.visualization.model_viz import ModelViz

from pyflowml.utils.saver import ModelSaver

__version__ = "1.0.4"
__author__ = "PyFlowML Contributors"

__all__ = [
    "PyFlowEngine",
    "DataProfiler",
    "DataLoader",
    "DataCleaner",
    "DataSplitter",
    "MemoryOptimizer",
    "Scaler",
    "FeatureSelector",
    "SmartPipeline",
    "AutoClassifier",
    "AutoRegressor",
    "AutoClusterer",
    "Reporter",
    "CrossValidator",
    "Plotter",
    "ModelViz",
    "ModelSaver",
]
