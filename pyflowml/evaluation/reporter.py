"""
Reporter — Pretty-print evaluation reports for classification and regression.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)
from pyflowml.monitoring.logger import get_logger
from pyflowml.utils.console import ensure_utf8_console

logger = get_logger("Reporter")


class Reporter:
    """
    Generates formatted evaluation reports.

    Example
    -------
    >>> Reporter.classification(clf, X_test, y_test)
    >>> Reporter.regression(reg, X_test, y_test)
    """

    @staticmethod
    def _resolve_average(y_true) -> str:
        """
        Determine the correct 'average' mode for sklearn metrics.

        Use 'binary' only when labels are exactly {0, 1} or {True, False}
        (where pos_label=1 is valid). Fall back to 'weighted' for all other
        binary cases (e.g. 'yes'/'no', 'cat'/'dog') to avoid pos_label errors.
        """
        unique = set(np.unique(y_true))
        is_binary = len(unique) == 2
        is_numeric_binary = unique in ({0, 1}, {True, False}, {0.0, 1.0})
        if is_binary and is_numeric_binary:
            return "binary"
        elif is_binary:
            return "weighted"   # string / arbitrary binary labels
        else:
            return "weighted"   # multiclass

    @staticmethod
    def classification(model, X_test, y_test, print_report: bool = True) -> dict:
        """Print and return classification metrics."""
        ensure_utf8_console()
        y_pred = model.predict(X_test)
        avg = Reporter._resolve_average(y_test)

        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4),
            "f1":        round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4),
        }

        # ROC-AUC — works for both numeric and string binary labels
        unique_labels = np.unique(y_test)
        if len(unique_labels) == 2 and hasattr(model, "predict_proba"):
            try:
                from sklearn.preprocessing import LabelBinarizer
                lb = LabelBinarizer()
                y_bin = lb.fit_transform(y_test).ravel()
                proba = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = round(roc_auc_score(y_bin, proba), 4)
            except Exception:
                pass

        if print_report:
            Reporter._print_classification(metrics)

        return metrics

    @staticmethod
    def regression(model, X_test, y_test, print_report: bool = True) -> dict:
        """Print and return regression metrics."""
        ensure_utf8_console()
        y_pred = model.predict(X_test)
        metrics = {
            "MAE": round(mean_absolute_error(y_test, y_pred), 4),
            "MSE": round(mean_squared_error(y_test, y_pred), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            "R2": round(r2_score(y_test, y_pred), 4),
            "MAPE": round(Reporter._mape(y_test, y_pred), 4),
        }
        if print_report:
            Reporter._print_regression(metrics)
        return metrics

    @staticmethod
    def _mape(y_true, y_pred) -> float:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mask = y_true != 0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    @staticmethod
    def _print_classification(metrics: dict):
        print("\n" + "╔" + "═" * 42 + "╗")
        print("║{:^42}║".format(" Classification Report "))
        print("╠" + "═" * 42 + "╣")
        for k, v in metrics.items():
            bar = Reporter._bar(v)
            print(f"║  {k:<12} : {v:.4f}  {bar:<14}║")
        print("╚" + "═" * 42 + "╝\n")

    @staticmethod
    def _print_regression(metrics: dict):
        print("\n" + "╔" + "═" * 42 + "╗")
        print("║{:^42}║".format(" Regression Report "))
        print("╠" + "═" * 42 + "╣")
        for k, v in metrics.items():
            print(f"║  {k:<12} : {v:<24.4f}║")
        print("╚" + "═" * 42 + "╝\n")

    @staticmethod
    def _bar(value: float, width: int = 10) -> str:
        filled = int(round(value * width))
        return "█" * filled + "░" * (width - filled)
