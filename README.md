<div align="center">
  <h1>🚀 PyFlowML</h1>
  <p><b>A completely automated, zero-code Machine Learning library that handles your entire data lifecycle instantly from your terminal.</b></p>

  [![PyPI Version](https://img.shields.io/pypi/v/pyflowml.svg)](https://pypi.org/project/pyflowml/)
  [![Python Versions](https://img.shields.io/pypi/pyversions/pyflowml.svg)](https://pypi.org/project/pyflowml/)
  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
  [![Downloads](https://static.pepy.tech/badge/pyflowml)](https://pepy.tech/project/pyflowml)

</div>

---

## ⚡ Why PyFlowML?
Stop manually importing Scikit-Learn tools, cleaning datasets, writing loops, and crashing your Jupyter Notebooks. **PyFlowML** automatically:
1. Optimizes your dataset memory (saving up to 80% RAM).
2. Cleans text outliers, NaNs, and duplicates automatically.
3. Spawns an intelligently parallelized **AutoML** Engine.
4. Searches across XGBoost, LightGBM, Random Forests, SVM, etc.
5. Generates a stunning dark-mode Visual Dashboard!

---

## 🛠️ Installation

Simply install the heavily optimized library via pip:

```bash
pip install pyflowml
```

---

## 🔥 Quickstart: The Magic CLI

The easiest way to use PyFlowML is right from your terminal without writing a single line of code! Just navigate to the folder with your CSV data and type:

```bash
pyflowml
```

Our beautiful interactive menu will guide you through picking your target column, selecting a time budget, and automating the rest!

### Example Terminal Output:
```text
  ✔  Loaded  2,000 rows × 17 columns
  🔍  Profiling dataset…
  📦  Optimising memory… Memory: 1.3 MB → 0.2 MB (saved 87%)
  🧹  Cleaning data…
  🤖  Training AutoML models (budget=60s)…
  ✅  Best: KNN | f1=0.5098
  📈  Generating visualisation dashboard…
```

---

## 💻 Zero-Boilerplate Code (Pro Mode)

If you strictly want to integrate PyFlowML into your Python backend or Jupyter Notebooks, it's as simple as three lines of code:

### AutoML Classification
```python
import pandas as pd
from pyflowml.models.auto import AutoClassifier

# 1. Load Data
df = pd.read_csv("my_dataset.csv")
X_train = df.drop(columns=["target"])
y_train = df["target"]

# 2. Launch your AutoML Engine!
engine = AutoClassifier(metric="f1", time_limit=60)
engine.fit(X_train, y_train)

# 3. View Results
print(f"🥇 Best Model: {engine.best_model_name_}")
engine.leaderboard()
```

*(For Regression, simply swap `AutoClassifier` for `AutoRegressor`!)*

---

## 🌟 Feature Breakdown

| Feature | Description |
|---|---|
| 🧠 **AutoML Search** | Safely threads 6+ state-of-the-art architectures without deadlocking |
| 📊 **Intelligent Profiler** | Detects classification vs regression & profiles correlation/skewness |
| ⚙️ **Smart Pipeline** | Auto Label-Encoding & OneHot features instantly |
| 📦 **Memory Optimizer** | Downcasts int64/float64 seamlessly for massive datasets |
| ⏱️ **Hard Deadline** | `time_limit=60` strictly stops processing to save compute costs |
| 📈 **One-Figure Dashboard** | Dark-themed ROC, Confusion Matrix, & Leaderboards plotted entirely in one frame! |
| 💾 **Safe Versioning** | Instantly saves your `.pkl` and `.json` metadata on successful train |

---

## 🤝 Contributing

We welcome contributions! Have an idea to make PyFlowML faster? Open an Issue or submit a Pull Request.

## 📄 License

This open-source project is heavily protected under the **MIT License**.
