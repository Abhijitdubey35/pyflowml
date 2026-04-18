"""
End-to-end classification example using PyFlowML.
Uses the Iris dataset to demonstrate the full pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# ─── Load Data ────────────────────────────────────────────────────────────────
iris = load_iris(as_frame=True)
df = iris.frame
df.columns = [*iris.feature_names, "target"]

print("=" * 60)
print("  PyFlowML — Classification Example (Iris Dataset)")
print("=" * 60)
print(f"  Dataset shape: {df.shape}")
print(f"  Target classes: {df['target'].unique()}\n")

# ─── Option 1: Use the Engine (fully automatic) ────────────────────────────────
from pyflowml.core.engine import PyFlowEngine

engine = PyFlowEngine(df, target="target", time_limit=60, metric="f1")
engine.run()

# ─── Option 2: Manual Step-by-Step ────────────────────────────────────────────
print("\n" + "─" * 60)
print("  Manual Pipeline")
print("─" * 60)

from pyflowml.data.memory import MemoryOptimizer
from pyflowml.data.cleaner import DataCleaner
from pyflowml.data.splitter import DataSplitter
from pyflowml.preprocessing.pipeline import SmartPipeline
from pyflowml.models.auto import AutoClassifier
from pyflowml.evaluation.reporter import Reporter
from pyflowml.evaluation.cross_validate import CrossValidator
from pyflowml.visualization.model_viz import ModelViz
from pyflowml.utils.saver import ModelSaver

# Optimize memory
df = MemoryOptimizer.reduce(df)

# Clean
df = DataCleaner(df).handle_nulls().remove_duplicates().result()

# Split
X_train, X_test, y_train, y_test = DataSplitter(df, target="target").split()

# Preprocess
pipe = SmartPipeline(scaler_method="standard", selector_method="correlation")
X_train_t = pipe.fit_transform(X_train, y_train)
X_test_t  = pipe.transform(X_test)

# Train
clf = AutoClassifier(metric="f1", time_limit=30)
clf.fit(X_train_t, y_train)
clf.leaderboard()

# Evaluate
Reporter.classification(clf, X_test_t, y_test)

# Cross-validate
cv = CrossValidator(clf, cv=5, metric="f1")
cv.run(X_train_t, y_train)
cv.summary()

# Visualize
ModelViz.confusion_matrix(clf, X_test_t, y_test)
ModelViz.feature_importance(clf, feature_names=list(X_train.columns))

# Save
ModelSaver.save(clf, "iris_classifier", metadata={
    "dataset": "iris",
    "f1_score": clf.best_score_,
    "features": list(X_train.columns),
})
ModelSaver.list_saved()
