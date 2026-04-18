"""
ModelViz — Confusion matrix, ROC curve, learning curves, feature importance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve,
)
from sklearn.model_selection import learning_curve

plt.rcParams["figure.facecolor"] = "#1a1a2e"
plt.rcParams["axes.facecolor"] = "#16213e"
plt.rcParams["text.color"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"
plt.rcParams["axes.edgecolor"] = "#444"


class ModelViz:
    """
    Model-specific visualizations.

    Example
    -------
    >>> ModelViz.confusion_matrix(clf, X_test, y_test)
    >>> ModelViz.roc_curve(clf, X_test, y_test)
    >>> ModelViz.learning_curve(clf, X, y)
    >>> ModelViz.feature_importance(clf, feature_names)
    """

    @staticmethod
    def confusion_matrix(model, X_test, y_test, labels=None,
                         figsize=(8, 6), save_path: str = None, return_fig: bool = False):
        """Plot a color-coded confusion matrix."""
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                    linewidths=0.5, ax=ax,
                    annot_kws={"size": 12, "color": "white"},
                    cbar_kws={"shrink": 0.8})

        ax.set_title("Confusion Matrix", color="white", fontsize=14, fontweight="bold")
        ax.set_ylabel("True Label", color="white")
        ax.set_xlabel("Predicted Label", color="white")
        if labels is not None:
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if return_fig:
            return fig
        plt.show()
        return fig

    @staticmethod
    def roc_curve(model, X_test, y_test, figsize=(8, 6), save_path: str = None,
                  return_fig: bool = False):
        """Plot ROC curve with AUC score (binary classification)."""
        if not hasattr(model, "predict_proba"):
            model = getattr(model, "best_model_", model)
        if not hasattr(model, "predict_proba"):
            print("Model does not support probability predictions.")
            return None

        y_score = model.predict_proba(X_test)[:, 1]
        from sklearn.preprocessing import LabelBinarizer
        lb = LabelBinarizer()
        y_bin = lb.fit_transform(y_test).ravel()
        fpr, tpr, _ = roc_curve(y_bin, y_score)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, color="#e94560", lw=2,
                label=f"ROC Curve (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], color="#888", linestyle="--", lw=1, label="Random")
        ax.fill_between(fpr, tpr, alpha=0.15, color="#e94560")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve", fontsize=14, fontweight="bold", color="white")
        ax.legend(loc="lower right", facecolor="#16213e", labelcolor="white")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if return_fig:
            return fig
        plt.show()
        return fig

    @staticmethod
    def learning_curve(model, X, y, cv: int = 5, scoring: str = "f1_weighted",
                       figsize=(10, 6), save_path: str = None, return_fig: bool = False):
        """Plot learning curves to diagnose overfitting/underfitting."""
        inner_model = getattr(model, "best_model_", model)
        train_sizes, train_scores, val_scores = learning_curve(
            inner_model, X, y, cv=cv, scoring=scoring,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
        )

        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(train_sizes, train_mean, "o-", color="#e94560", label="Training Score")
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.15, color="#e94560")
        ax.plot(train_sizes, val_mean, "o-", color="#0f3460", label="Validation Score")
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.15, color="#0f3460")

        ax.set_title("Learning Curves", fontsize=14, fontweight="bold", color="white")
        ax.set_xlabel("Training Samples")
        ax.set_ylabel(scoring)
        ax.legend(facecolor="#16213e", labelcolor="white")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if return_fig:
            return fig
        plt.show()
        return fig

    @staticmethod
    def feature_importance(model, feature_names: list, top_n: int = 20,
                           figsize=(10, 8), save_path: str = None, return_fig: bool = False):
        """Plot feature importance bar chart."""
        inner = getattr(model, "best_model_", model)
        if not hasattr(inner, "feature_importances_"):
            print("Model does not have feature_importances_ attribute.")
            return None

        importances = inner.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        names   = [feature_names[i] for i in indices]
        values  = importances[indices]

        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))
        bars   = ax.barh(range(len(names)), values[::-1], color=colors[::-1], edgecolor="white")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names[::-1], fontsize=10)
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Top {top_n} Feature Importances", color="white",
                     fontsize=14, fontweight="bold")
        ax.invert_xaxis()

        for bar, val in zip(bars, values[::-1]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9, color="white")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if return_fig:
            return fig
        plt.show()
        return fig
