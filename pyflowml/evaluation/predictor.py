"""
PredictionTester — Runs the trained model against the held-out test set,
builds a predicted-vs-actual comparison table, and creates comparison plots.

Works for both classification and regression problems.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

plt.rcParams["figure.facecolor"] = "#1a1a2e"
plt.rcParams["axes.facecolor"]   = "#16213e"
plt.rcParams["text.color"]       = "white"
plt.rcParams["axes.labelcolor"]  = "white"
plt.rcParams["xtick.color"]      = "white"
plt.rcParams["ytick.color"]      = "white"
plt.rcParams["axes.edgecolor"]   = "#444"


class PredictionTester:
    """
    Runs the trained model on the test set and compares predicted vs actual.

    Provides:
    - comparison_df()  → DataFrame with Actual / Predicted / Correct columns
    - summary()        → per-class or per-bucket accuracy summary
    - plot()           → comparison visualization (saved or shown)

    Example
    -------
    >>> tester = PredictionTester(model, X_test, y_test, problem_type="classification")
    >>> tester.run()
    >>> df = tester.comparison_df()
    >>> tester.plot()
    """

    def __init__(self, model, X_test, y_test, problem_type: str = "classification",
                 feature_names: list = None):
        self.model        = model
        self.X_test       = X_test
        self.y_test       = np.array(y_test)
        self.problem_type = problem_type
        self.feature_names = feature_names or (
            list(X_test.columns) if hasattr(X_test, "columns") else []
        )
        self._y_pred   = None
        self._y_proba  = None
        self._df       = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self) -> "PredictionTester":
        """Generate predictions from the model."""
        self._y_pred = self.model.predict(self.X_test)

        if (self.problem_type == "classification"
                and hasattr(self.model, "predict_proba")):
            try:
                self._y_proba = self.model.predict_proba(self.X_test)
            except Exception:
                self._y_proba = None

        self._build_df()
        return self

    def comparison_df(self) -> pd.DataFrame:
        """Return the full Actual / Predicted / Correct comparison table."""
        self._ensure_run()
        return self._df.copy()

    def summary(self) -> pd.DataFrame:
        """
        Classification → per-class accuracy table.
        Regression    → 5 equal-width actual-value buckets with RMSE per bucket.
        """
        self._ensure_run()
        if self.problem_type == "classification":
            return self._classification_summary()
        else:
            return self._regression_summary()

    def plot(self, save_path: str = None, return_fig: bool = False):
        """
        Produce a comparison visualisation.

        Classification → confusion bar + per-class accuracy side by side.
        Regression     → actual vs predicted scatter + residuals distribution.
        """
        self._ensure_run()
        if self.problem_type == "classification":
            fig = self._plot_classification()
        else:
            fig = self._plot_regression()

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if return_fig:
            return fig
        plt.show()
        return fig

    # ── Internals ──────────────────────────────────────────────────────────────

    def _build_df(self):
        df = pd.DataFrame({
            "Actual":    self.y_test,
            "Predicted": self._y_pred,
        })
        if self.problem_type == "classification":
            df["Correct"] = df["Actual"] == df["Predicted"]
            df["Status"]  = df["Correct"].map({True: "✅ Correct", False: "❌ Wrong"})
        else:
            df["Error"]    = df["Predicted"] - df["Actual"]
            df["Abs Error"]= df["Error"].abs()
            df["% Error"]  = (
                (df["Abs Error"] / df["Actual"].replace(0, np.nan)).abs() * 100
            ).round(2)
        self._df = df

    def _classification_summary(self) -> pd.DataFrame:
        classes = np.unique(self.y_test)
        rows = []
        for cls in classes:
            mask = self.y_test == cls
            total = mask.sum()
            correct = ((self.y_test[mask] == self._y_pred[mask])).sum()
            rows.append({
                "Class": cls,
                "Total Samples": int(total),
                "Correct": int(correct),
                "Wrong": int(total - correct),
                "Accuracy %": round(correct / total * 100, 1) if total else 0,
            })
        return pd.DataFrame(rows)

    def _regression_summary(self) -> pd.DataFrame:
        df = self._df.copy()
        df["Bucket"] = pd.cut(df["Actual"], bins=5)
        return (
            df.groupby("Bucket")["Abs Error"]
            .agg(["count", "mean", "median"])
            .rename(columns={"count": "Samples", "mean": "Mean Abs Error",
                              "median": "Median Abs Error"})
            .round(4)
            .reset_index()
        )

    # ── Plots ──────────────────────────────────────────────────────────────────

    def _plot_classification(self):
        summary = self._classification_summary()
        classes = summary["Class"].astype(str).tolist()
        correct = summary["Correct"].tolist()
        wrong   = summary["Wrong"].tolist()
        acc     = summary["Accuracy %"].tolist()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Prediction Testing — Classification", color="white",
                     fontsize=15, fontweight="bold", y=1.01)

        # Left: stacked bar per class
        x = np.arange(len(classes))
        bars_c = axes[0].bar(x, correct, label="Correct ✅", color="#4ade80", width=0.5)
        bars_w = axes[0].bar(x, wrong, bottom=correct, label="Wrong ❌",
                             color="#f87171", width=0.5)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(classes, rotation=30, ha="right")
        axes[0].set_title("Correct vs Wrong per Class", color="white")
        axes[0].set_ylabel("Number of Samples")
        axes[0].legend(facecolor="#16213e", labelcolor="white")
        # labels on bars
        for bar, n in zip(bars_c, correct):
            if n > 0:
                axes[0].text(bar.get_x() + bar.get_width()/2, n/2,
                             str(n), ha="center", va="center", color="white", fontsize=9)
        for bar, c_val, w in zip(bars_w, correct, wrong):
            if w > 0:
                axes[0].text(bar.get_x() + bar.get_width()/2, c_val + w/2,
                             str(w), ha="center", va="center", color="white", fontsize=9)

        # Right: accuracy % per class
        colors = ["#4ade80" if a >= 70 else "#facc15" if a >= 50 else "#f87171"
                  for a in acc]
        axes[1].barh(classes, acc, color=colors, edgecolor="white")
        axes[1].axvline(80, color="#facc15", linestyle="--", linewidth=1, label="80% line")
        axes[1].set_xlim(0, 105)
        axes[1].set_title("Per-Class Accuracy %", color="white")
        axes[1].set_xlabel("Accuracy %")
        for i, (a, cls) in enumerate(zip(acc, classes)):
            axes[1].text(a + 1, i, f"{a:.1f}%", va="center", color="white", fontsize=9)
        axes[1].legend(facecolor="#16213e", labelcolor="white")

        return fig

    def _plot_regression(self):
        actual    = self._df["Actual"].values
        predicted = self._df["Predicted"].values
        errors    = self._df["Error"].values

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Prediction Testing — Regression", color="white",
                     fontsize=15, fontweight="bold", y=1.01)

        # Left: Actual vs Predicted scatter
        axes[0].scatter(actual, predicted, alpha=0.5, color="#e94560",
                        edgecolors="white", linewidths=0.3, s=40)
        min_v, max_v = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        axes[0].plot([min_v, max_v], [min_v, max_v], "--", color="#4ade80",
                     linewidth=1.5, label="Perfect fit")
        axes[0].set_xlabel("Actual")
        axes[0].set_ylabel("Predicted")
        axes[0].set_title("Actual vs Predicted", color="white")
        axes[0].legend(facecolor="#16213e", labelcolor="white")

        # Right: Residuals distribution
        axes[1].hist(errors, bins=40, color="#818cf8", edgecolor="white", alpha=0.8)
        axes[1].axvline(0, color="#4ade80", linestyle="--", linewidth=1.5, label="Zero error")
        axes[1].set_xlabel("Residual (Predicted – Actual)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Residuals Distribution", color="white")
        axes[1].legend(facecolor="#16213e", labelcolor="white")

        return fig

    def _ensure_run(self):
        if self._y_pred is None:
            raise RuntimeError("Call .run() first.")
