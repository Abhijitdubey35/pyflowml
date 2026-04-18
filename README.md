# PyFlowML 🚀

**A scalable, intelligent Python ML library for the full machine learning lifecycle.**

PyFlowML handles everything from raw, potentially large data to a trained, evaluated, and saved model — automatically.

---

## Features

| Feature | Description |
|---|---|
| 🧠 **AutoML** | Trains multiple models in parallel, picks the best automatically |
| 📊 **Data Profiler** | Detects problem type, missing data, skewness, and outliers |
| ⚙️ **Smart Pipeline** | Cached preprocessing: no redundant recomputation |
| 📦 **Memory Optimizer** | Downcasts dtypes — saves 40–60% RAM |
| 🚀 **Parallel Training** | All models train simultaneously via `joblib` |
| ⏱️ **Time Budget** | `AutoClassifier(time_limit=60)` stops when budget exceeded |
| 📈 **Visualization** | Dark-themed ROC, confusion matrix, learning curves |
| 💾 **Versioned Saves** | Models saved with metadata (date, metrics, features) |
| 📝 **NLP Utilities** | TF-IDF, Bag-of-Words, stopword removal, lemmatization |
| 📡 **Monitoring** | Per-step time and memory tracking |

---

## Installation

```bash
pip install -e .
```

Or install from requirements:

```bash
pip install -r requirements.txt
```

---

## Quickstart

### One-line AutoML Engine

```python
import pandas as pd
from pyflowml.core.engine import PyFlowEngine

df = pd.read_csv("titanic.csv")
engine = PyFlowEngine(df, target="Survived", time_limit=120)
engine.run()
```

### Manual Step-by-Step

```python
from pyflowml.data import DataLoader, DataCleaner, DataSplitter, MemoryOptimizer
from pyflowml.preprocessing import SmartPipeline
from pyflowml.models import AutoClassifier
from pyflowml.evaluation import Reporter
from pyflowml.visualization import ModelViz
from pyflowml.utils import ModelSaver

# Load & optimize
df = DataLoader.from_csv("data.csv")
df = MemoryOptimizer.reduce(df)

# Clean
df = DataCleaner(df).handle_nulls().remove_outliers().remove_duplicates().result()

# Split
X_train, X_test, y_train, y_test = DataSplitter(df, target="label").split()

# Preprocess
pipe = SmartPipeline()
X_train = pipe.fit_transform(X_train, y_train)
X_test  = pipe.transform(X_test)

# Train best model
clf = AutoClassifier(metric="f1", time_limit=60)
clf.fit(X_train, y_train)
clf.leaderboard()

# Evaluate
Reporter.classification(clf, X_test, y_test)

# Visualize
ModelViz.confusion_matrix(clf, X_test, y_test)
ModelViz.roc_curve(clf, X_test, y_test)

# Save
ModelSaver.save(clf, "my_model", metadata={"f1": clf.best_score_})
```

---

## Modules

```
pyflowml/
├── core/          # Engine, Profiler, Optimizer (brain of the system)
├── data/          # DataLoader, DataCleaner, DataSplitter, MemoryOptimizer
├── preprocessing/ # Scaler, FeatureSelector, SmartPipeline
├── models/        # AutoClassifier, AutoRegressor, AutoClusterer
├── evaluation/    # Reporter, CrossValidator
├── tuning/        # HyperTuner
├── visualization/ # Plotter, ModelViz
├── monitoring/    # StepTracker, Logger
├── utils/         # ModelSaver
└── nlp/           # TextCleaner, Vectorizer
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## License

MIT
