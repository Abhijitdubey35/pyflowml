"""
PyFlowML Interactive CLI
========================
Run the full ML pipeline on any CSV or JSON file.

Usage:
    python run.py
    python run.py --file data.csv --target price
    python run.py --file data.json --target label --metric f1 --time 60
"""

import os
import sys
import argparse
import pandas as pd


# ─── Colour helpers ───────────────────────────────────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def c(text, colour): return f"{colour}{text}{RESET}"
def banner():
    print(c("""
╔══════════════════════════════════════════════════════╗
║                                                      ║
║          🚀  PyFlowML  —  AutoML Engine              ║
║      Train · Evaluate · Visualise · Save             ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
""", CYAN + BOLD))


# ─── File loading ─────────────────────────────────────────────────────────────

def load_file(path: str) -> pd.DataFrame:
    """Load CSV or JSON file into a DataFrame."""
    ext = os.path.splitext(path)[-1].lower()

    if not os.path.exists(path):
        print(c(f"\n  ✖  File not found: {path}", RED))
        sys.exit(1)

    size_mb = os.path.getsize(path) / 1024 / 1024
    print(c(f"\n  📂  Loading {ext.upper()} file  ({size_mb:.2f} MB)…", CYAN))

    try:
        if ext == ".csv":
            # Use chunked loading for large files (> 50 MB)
            if size_mb > 50:
                print(c("     Large file detected — using chunked loading…", YELLOW))
                chunks = []
                for chunk in pd.read_csv(path, chunksize=50_000):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(path)

        elif ext in (".json", ".jsonl"):
            df = pd.read_json(path)

        else:
            print(c(f"\n  ✖  Unsupported file type: '{ext}' — use .csv or .json", RED))
            sys.exit(1)

    except Exception as e:
        print(c(f"\n  ✖  Failed to load file: {e}", RED))
        sys.exit(1)

    print(c(f"  ✔  Loaded  {df.shape[0]:,} rows × {df.shape[1]} columns", GREEN))
    return df


# ─── Interactive prompts ───────────────────────────────────────────────────────

def prompt_file() -> str:
    """Ask the user for a CSV or JSON file path."""
    while True:
        path = input(c("\n  📁  Path to your CSV or JSON file: ", BOLD)).strip().strip('"').strip("'")
        if not path:
            print(c("     Please enter a file path.", YELLOW))
            continue
        if not os.path.exists(path):
            print(c(f"     File not found: {path}", RED))
            retry = input(c("     Try again? (y/n): ", YELLOW)).strip().lower()
            if retry != "y":
                sys.exit(0)
            continue
        ext = os.path.splitext(path)[-1].lower()
        if ext not in (".csv", ".json", ".jsonl"):
            print(c(f"     Unsupported type '{ext}'. Only .csv and .json are supported.", RED))
            continue
        return path


def prompt_target(df: pd.DataFrame) -> str:
    """Ask which column is the target."""
    print(c("\n  Columns in your dataset:", BOLD))
    cols = list(df.columns)
    for i, col in enumerate(cols):
        dtype = str(df[col].dtype)
        nuniq = df[col].nunique()
        print(f"    [{i:>2}]  {col:<35} dtype={dtype:<10} unique={nuniq}")

    while True:
        raw = input(c("\n  🎯  Target column name (or index): ", BOLD)).strip()
        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(cols):
                return cols[idx]
            print(c(f"     Index {idx} out of range (0–{len(cols)-1}).", RED))
        elif raw in cols:
            return raw
        else:
            print(c(f"     '{raw}' not found. Try the column name or its index number.", RED))


def prompt_metric(problem_type: str) -> str:
    """Ask for the evaluation metric."""
    if problem_type == "classification":
        options = {"1": "f1", "2": "accuracy", "3": "roc_auc"}
        labels  = {"1": "F1 Score (recommended)", "2": "Accuracy", "3": "ROC-AUC (binary only)"}
    else:
        options = {"1": "rmse", "2": "r2", "3": "mae"}
        labels  = {"1": "RMSE (recommended)", "2": "R² Score", "3": "MAE"}

    print(c(f"\n  Problem type detected: {problem_type.upper()}", GREEN))
    print(c("\n  Select evaluation metric:", BOLD))
    for k, label in labels.items():
        print(f"    [{k}]  {label}")

    while True:
        choice = input(c("\n  Your choice (1/2/3, default=1): ", BOLD)).strip() or "1"
        if choice in options:
            return options[choice]
        print(c("     Enter 1, 2, or 3.", RED))


def prompt_time_limit() -> int:
    """Ask for training time budget in seconds, with clear labelling."""
    print(c("\n  ⏱️   Training time budget", BOLD))
    print(c("       This is the MAXIMUM time the AutoML engine will spend.", YELLOW))
    print(c("       Models train in parallel — it may finish faster if they converge early.", YELLOW))
    print(f"       Examples: {c('30', GREEN)} = 30 sec  |  {c('60', GREEN)} = 1 min  |  {c('300', GREEN)} = 5 min")
    while True:
        raw = input(c("\n       Enter seconds (default=60): ", BOLD)).strip()
        if not raw:
            return 60
        if raw.isdigit() and int(raw) > 0:
            secs = int(raw)
            mins = secs / 60
            print(c(f"       ✔  Budget set to {secs}s ({mins:.1f} min)", GREEN))
            return secs
        print(c("       Please enter a positive integer number of seconds.", RED))


def prompt_visualise() -> bool:
    ans = input(c("\n  📈  Show visualisation plots? (y/n, default=y): ", BOLD)).strip().lower()
    return ans != "n"


def prompt_save() -> bool:
    ans = input(c("\n  💾  Save the best model to disk? (y/n, default=y): ", BOLD)).strip().lower()
    return ans != "n"


# ─── Unified Visualisation Dashboard ─────────────────────────────────────────

def _render_dashboard(df, raw_file, model, X_train_t, X_test_t,
                      y_train, y_test, problem_type):
    """
    Render ALL visualisation plots inside a single matplotlib figure (one sheet).

    Regression layout  (3 rows × 2 cols):
      [0,0] Correlation Heatmap       [0,1] Feature Importance
      [1,0] Residuals vs Predicted    [1,1] Actual vs Predicted
      [2,0] Learning Curves           [2,1] Model Leaderboard

    Classification layout:
      [0,0] Correlation Heatmap       [0,1] Feature Importance
      [1,0] Confusion Matrix          [1,1] ROC Curve
      [2,0] Learning Curves           [2,1] Model Leaderboard
    """
    import traceback
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("TkAgg") if False else None   # keep current backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns

    # ── Global dark theme ────────────────────────────────────────────────────
    BG      = "#0d1117"
    PANEL   = "#161b22"
    TEXT    = "#e6edf3"
    MUTED   = "#8b949e"
    ACCENT  = "#58a6ff"
    RED     = "#f78166"
    GREEN   = "#56d364"
    BLUE    = "#79c0ff"
    BORDER  = "#30363d"
    GRID    = "#21262d"

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL,
        "text.color": TEXT, "axes.labelcolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": BORDER, "grid.color": GRID,
        "axes.titlecolor": ACCENT, "axes.titlesize": 12,
        "axes.titleweight": "bold", "font.family": "DejaVu Sans",
    })

    fig = plt.figure(figsize=(24, 18), facecolor=BG)
    fig.suptitle(
        f"📊  PyFlowML — Visualisation Dashboard   "
        f"[ Best model: {model.best_model_name_} ]",
        fontsize=16, fontweight="bold", color=ACCENT, y=0.995,
    )

    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        hspace=0.50, wspace=0.32,
        top=0.96, bottom=0.06, left=0.06, right=0.97,
    )

    inner = getattr(model, "best_model_", model)

    def _title(ax, txt):
        ax.set_title(txt, color=ACCENT, fontsize=12, fontweight="bold", pad=8)

    def _err(ax, msg):
        """Show a styled error message inside a panel."""
        print(c(f"  ⚠  Dashboard panel error: {msg}", YELLOW))
        ax.set_facecolor(PANEL)
        ax.text(0.5, 0.5, f"⚠  {msg}", ha="center", va="center",
                transform=ax.transAxes, color=RED, fontsize=9,
                wrap=True, multialignment="center")

    # ═══════════════════════════════════════════════════════════════════════════
    # Panel [0,0] — Correlation Heatmap (numeric cols of raw cleaned df)
    # ═══════════════════════════════════════════════════════════════════════════
    ax0 = fig.add_subplot(gs[0, 0])
    try:
        # Reload original file for the heatmap so we get full float64 precision
        raw_df = pd.read_csv(raw_file) if raw_file.lower().endswith(".csv") \
                 else pd.read_json(raw_file)
        num_df = raw_df.select_dtypes(include=[np.number])
        corr   = num_df.corr()
        mask   = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask,
            annot=True, fmt=".2f", annot_kws={"size": 7, "color": TEXT},
            cmap="RdYlGn", vmin=-1, vmax=1,
            linewidths=0.3, linecolor=BORDER, ax=ax0,
            cbar_kws={"shrink": 0.75, "pad": 0.02},
        )
        ax0.tick_params(axis="x", rotation=45, labelsize=7)
        ax0.tick_params(axis="y", rotation=0,  labelsize=7)
    except Exception:
        _err(ax0, traceback.format_exc(limit=1).strip().splitlines()[-1])
    _title(ax0, "Correlation Heatmap")

    # ═══════════════════════════════════════════════════════════════════════════
    # Panel [0,1] — Feature Importance
    # ═══════════════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, 1])
    try:
        if hasattr(inner, "feature_importances_"):
            importances = inner.feature_importances_
        elif hasattr(inner, "coef_"):
            importances = np.abs(inner.coef_[0]) if len(inner.coef_.shape) > 1 else np.abs(inner.coef_)
        else:
            raise AttributeError("Model has no feature importances or coefficients")
        # Ensure X_train_t has column names
        if hasattr(X_train_t, "columns"):
            feat_names = list(X_train_t.columns)
        else:
            feat_names = [f"f{i}" for i in range(len(importances))]
        top_n   = min(20, len(feat_names))
        # Sort descending by importance
        idx     = np.argsort(importances)[::-1][:top_n]
        names   = [feat_names[i] for i in idx]
        values  = importances[idx]
        # Plot in ascending order so best is at top of horizontal bar chart
        colors  = plt.cm.RdYlGn(np.linspace(0.2, 0.85, top_n))[::-1]
        y_pos   = np.arange(top_n)
        ax1.barh(y_pos, values[::-1], color=colors, edgecolor=BORDER, height=0.65)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(names[::-1], fontsize=8)
        ax1.set_xlabel("Importance Score", fontsize=9)
        ax1.grid(axis="x", alpha=0.25, color=GRID)
        for i, v in enumerate(values[::-1]):
            ax1.text(v + values.max() * 0.01, i, f"{v:.4f}",
                     va="center", fontsize=7.5, color=TEXT)
    except Exception:
        _err(ax1, traceback.format_exc(limit=1).strip().splitlines()[-1])
    _title(ax1, f"Top Feature Importances — {model.best_model_name_}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Panel [1,0] — Residuals (reg) / Confusion Matrix (clf)
    # ═══════════════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1, 0])
    try:
        y_pred = model.predict(X_test_t)
        if problem_type == "regression":
            y_arr     = np.array(y_test)
            residuals = y_arr - y_pred
            ax2.scatter(y_pred, residuals, alpha=0.45, s=20, color=BLUE, edgecolors="none")
            ax2.axhline(0, color=RED, linestyle="--", lw=1.5, label="Zero error")
            # Loess-style trend line (rolling mean)
            order = np.argsort(y_pred)
            xo, ro = y_pred[order], residuals[order]
            window = max(5, len(xo) // 20)
            rm = pd.Series(ro).rolling(window, center=True, min_periods=1).mean().values
            ax2.plot(xo, rm, color=GREEN, lw=1.5, alpha=0.7, label="Trend")
            ax2.set_xlabel("Predicted", fontsize=9)
            ax2.set_ylabel("Residual (actual − predicted)", fontsize=9)
            ax2.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)
            ax2.grid(True, alpha=0.2, color=GRID)
        else:
            from sklearn.metrics import confusion_matrix as cm_fn
            cm_arr  = cm_fn(y_test, y_pred)
            cm_norm = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True)
            sns.heatmap(cm_norm, annot=cm_arr, fmt="d", cmap="Blues",
                        linewidths=0.3, ax=ax2,
                        annot_kws={"size": 9, "color": TEXT},
                        cbar_kws={"shrink": 0.8})
            ax2.set_ylabel("True label", fontsize=9)
            ax2.set_xlabel("Predicted label", fontsize=9)
    except Exception:
        _err(ax2, traceback.format_exc(limit=1).strip().splitlines()[-1])
    _title(ax2, "Residuals vs Predicted" if problem_type == "regression" else "Confusion Matrix")

    # ═══════════════════════════════════════════════════════════════════════════
    # Panel [1,1] — Actual vs Predicted (reg) / ROC Curve (clf)
    # ═══════════════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[1, 1])
    try:
        if problem_type == "regression":
            y_pred_b = model.predict(X_test_t)
            y_arr    = np.array(y_test)
            mn = min(y_arr.min(), y_pred_b.min())
            mx = max(y_arr.max(), y_pred_b.max())
            ax3.scatter(y_arr, y_pred_b, alpha=0.4, s=20, color=BLUE, edgecolors="none", label="Samples")
            ax3.plot([mn, mx], [mn, mx], color=RED, lw=1.5, ls="--", label="Perfect fit")
            ax3.set_xlabel("Actual value", fontsize=9)
            ax3.set_ylabel("Predicted value", fontsize=9)
            ax3.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)
            ax3.grid(True, alpha=0.2, color=GRID)
        else:
            from sklearn.metrics import roc_curve as roc_fn, auc
            from sklearn.preprocessing import LabelBinarizer
            lb_  = LabelBinarizer()
            y_bin = lb_.fit_transform(y_test).ravel()
            proba = model.predict_proba(X_test_t)
            # Use col-1 for binary; for multi-class use OvR macro-avg
            if proba.shape[1] == 2:
                score_col = proba[:, 1]
                fpr, tpr, _ = roc_fn(y_bin, score_col)
                roc_auc_v   = auc(fpr, tpr)
                ax3.plot(fpr, tpr, color=RED, lw=2, label=f"AUC = {roc_auc_v:.3f}")
                ax3.fill_between(fpr, tpr, alpha=0.12, color=RED)
            else:
                from sklearn.metrics import roc_auc_score
                ax3.text(0.5, 0.5, "Multi-class ROC\nnot supported here",
                         ha="center", va="center", transform=ax3.transAxes,
                         color=MUTED, fontsize=10)
            ax3.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
            ax3.set_xlim([0, 1]); ax3.set_ylim([0, 1.05])
            ax3.set_xlabel("False Positive Rate", fontsize=9)
            ax3.set_ylabel("True Positive Rate", fontsize=9)
            ax3.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)
            ax3.grid(True, alpha=0.2, color=GRID)
    except Exception:
        _err(ax3, traceback.format_exc(limit=1).strip().splitlines()[-1])
    _title(ax3, "Actual vs Predicted" if problem_type == "regression" else "ROC Curve")

    # ═══════════════════════════════════════════════════════════════════════════
    # Panel [2,0] — Learning Curves
    # ═══════════════════════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(gs[2, 0])
    try:
        from sklearn.model_selection import learning_curve as lc_fn
        scoring  = "r2" if problem_type == "regression" else "f1_weighted"
        X_lc     = X_train_t.values if hasattr(X_train_t, "values") else X_train_t
        y_lc     = np.array(y_train)
        sizes, tr_sc, val_sc = lc_fn(
            inner, X_lc, y_lc, cv=5, scoring=scoring,
            n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 7),
        )
        tr_m, tr_s  = tr_sc.mean(1), tr_sc.std(1)
        va_m, va_s  = val_sc.mean(1), val_sc.std(1)
        ax4.plot(sizes, tr_m, "o-", color=RED,   lw=2, label="Train")
        ax4.fill_between(sizes, tr_m - tr_s, tr_m + tr_s, alpha=0.15, color=RED)
        ax4.plot(sizes, va_m, "o-", color=GREEN, lw=2, label="Validation")
        ax4.fill_between(sizes, va_m - va_s, va_m + va_s, alpha=0.15, color=GREEN)
        ax4.set_xlabel("Training samples", fontsize=9)
        ax4.set_ylabel(scoring, fontsize=9)
        ax4.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)
        ax4.grid(True, alpha=0.2, color=GRID)
    except Exception:
        _err(ax4, traceback.format_exc(limit=1).strip().splitlines()[-1])
    _title(ax4, "Learning Curves")

    # ═══════════════════════════════════════════════════════════════════════════
    # Panel [2,1] — Model Leaderboard (horizontal bar chart)
    # ═══════════════════════════════════════════════════════════════════════════
    ax5 = fig.add_subplot(gs[2, 1])
    try:
        lb_data    = model._leaderboard   # list of dicts, sorted best-first
        names_lb   = [r["model"]          for r in lb_data]
        scores_lb  = [abs(r["score"])     for r in lb_data]   # abs so bars are positive
        bar_colors = [GREEN if i == 0 else ACCENT for i in range(len(names_lb))]
        # Plot in reverse so best is at the top
        y_pos_lb   = np.arange(len(names_lb))
        bars = ax5.barh(y_pos_lb, scores_lb[::-1],
                        color=bar_colors[::-1], edgecolor=BORDER, height=0.6)
        ax5.set_yticks(y_pos_lb)
        ax5.set_yticklabels(names_lb[::-1], fontsize=9)
        max_sc = max(scores_lb) if scores_lb else 1
        for bar, sc in zip(bars, scores_lb[::-1]):
            ax5.text(bar.get_width() + max_sc * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f"{sc:.4f}", va="center", fontsize=8, color=TEXT)
        metric_name = model._leaderboard[0]["model"] if lb_data else "score"
        ax5.set_xlabel(f"|{getattr(model, 'metric', 'score')}| — lower is better for error metrics",
                       fontsize=8)
        ax5.set_xlim(0, max_sc * 1.18)
        ax5.grid(axis="x", alpha=0.2, color=GRID)
    except Exception:
        _err(ax5, traceback.format_exc(limit=1).strip().splitlines()[-1])
    _title(ax5, "Model Leaderboard")

    plt.show()
    print(c("  ✔  Dashboard displayed.", GREEN))


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(file_path: str, target: str, metric: str = "auto",
                 time_limit: int = 120, visualise: bool = True, save: bool = True):
    """Run the full PyFlowML pipeline on the given file."""
    from pyflowml.data.memory import MemoryOptimizer
    from pyflowml.data.cleaner import DataCleaner
    from pyflowml.data.splitter import DataSplitter
    from pyflowml.core.profiler import DataProfiler
    from pyflowml.preprocessing.pipeline import SmartPipeline
    from pyflowml.models.auto import AutoClassifier, AutoRegressor
    from pyflowml.evaluation.reporter import Reporter
    from pyflowml.utils.saver import ModelSaver

    # 1. Load
    df = load_file(file_path)

    # 2. Validate target
    if target not in df.columns:
        print(c(f"\n  ✖  Target column '{target}' not in dataset.", RED))
        sys.exit(1)

    # 3. Profile
    print(c("\n  🔍  Profiling dataset…", CYAN))
    profiler = DataProfiler(df, target=target)
    profile  = profiler.run()
    profiler.report()

    problem_type = profile["problem_type"]

    # 4. Metric
    if metric == "auto":
        metric = "f1" if problem_type == "classification" else "rmse"
    print(c(f"\n  ✔  Metric selected: {metric}", GREEN))

    # 5. Optimise memory
    print(c("\n  📦  Optimising memory…", CYAN))
    df = MemoryOptimizer.reduce(df)

    # 6. Clean
    print(c("\n  🧹  Cleaning data…", CYAN))
    cleaner = DataCleaner(df)
    cleaner.normalize_text().handle_nulls().remove_duplicates().remove_outliers()
    df = cleaner.result()
    print(c(f"  ✔  Clean shape: {df.shape}", GREEN))

    # 7. Split
    X_train, X_test, y_train, y_test = DataSplitter(df, target=target).split()

    # 8. Preprocess
    print(c("\n  ⚙️   Preprocessing…", CYAN))
    pipe = SmartPipeline(
        scaler_method="standard",
        selector_method="correlation",
        problem_type=problem_type,
    )
    X_train_t = pipe.fit_transform(X_train, y_train)
    X_test_t  = pipe.transform(X_test)

    # 9. Train
    print(c(f"\n  🤖  Training AutoML models (budget={time_limit}s)…", CYAN))
    if problem_type == "classification":
        model = AutoClassifier(metric=metric, time_limit=time_limit)
    else:
        model = AutoRegressor(metric=metric, time_limit=time_limit)

    model.fit(X_train_t, y_train)
    model.leaderboard()

    # 10. Evaluate
    print(c("\n  📊  Evaluation Results:", CYAN))
    if problem_type == "classification":
        Reporter.classification(model, X_test_t, y_test)
    else:
        Reporter.regression(model, X_test_t, y_test)

    # 11. Visualise — all plots in ONE dashboard figure
    if visualise:
        print(c("\n  📈  Generating visualisation dashboard…", CYAN))
        _render_dashboard(
            df=df,
            raw_file=file_path,
            model=model,
            X_train_t=X_train_t,
            X_test_t=X_test_t,
            y_train=y_train,
            y_test=y_test,
            problem_type=problem_type,
        )


    # 12. Save
    if save:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        model_name = f"{base_name}_{problem_type}_model"
        saved_path = ModelSaver.save(model, model_name, metadata={
            "file": file_path,
            "target": target,
            "problem_type": problem_type,
            "metric": metric,
            "best_model": model.best_model_name_,
            "score": model.best_score_,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        })
        print(c(f"\n  ✅  Model saved → {saved_path}", GREEN))

    print(c("\n  🎉  Done! PyFlowML pipeline complete.\n", GREEN + BOLD))
    return model


# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="PyFlowML — Run AutoML on any CSV or JSON file",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--file",   "-f", type=str, help="Path to CSV or JSON file")
    parser.add_argument("--target", "-t", type=str, help="Target column name")
    parser.add_argument("--metric", "-m", type=str, default="auto",
                        help="Metric: f1|accuracy|roc_auc|rmse|r2|mae (default: auto)")
    parser.add_argument("--time",   "-T", type=int, default=120,
                        help="Training time budget in seconds (default: 120)")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualisation")
    parser.add_argument("--no-save", action="store_true", help="Skip model saving")
    return parser.parse_args()


def main():
    banner()
    args = parse_args()

    # ── File ────────────────────────────────────────────────────────────────
    file_path = args.file if args.file else prompt_file()

    # Load early so we can show columns for target selection
    df_preview = load_file(file_path)

    # ── Target ──────────────────────────────────────────────────────────────
    target = args.target if args.target else prompt_target(df_preview)
    print(c(f"\n  ✔  Target: {target}", GREEN))

    # ── Profile problem type for metric prompt ───────────────────────────────
    from pyflowml.core.profiler import DataProfiler
    profile = DataProfiler(df_preview, target=target).run()
    problem_type = profile["problem_type"]

    # ── Metric ──────────────────────────────────────────────────────────────
    if args.metric == "auto":
        metric = prompt_metric(problem_type)
    else:
        metric = args.metric

    # ── Time limit ──────────────────────────────────────────────────────────
    time_limit = args.time if args.time != 120 or args.file else prompt_time_limit()

    # ── Viz & save ──────────────────────────────────────────────────────────
    visualise = (not args.no_viz) and prompt_visualise()
    save      = (not args.no_save) and prompt_save()

    print(c("\n" + "─" * 56, CYAN))
    print(c(f"  File    : {file_path}", BOLD))
    print(c(f"  Target  : {target}", BOLD))
    print(c(f"  Metric  : {metric}", BOLD))
    print(c(f"  Budget  : {time_limit}s", BOLD))
    print(c(f"  Plots   : {'yes' if visualise else 'no'}", BOLD))
    print(c(f"  Save    : {'yes' if save else 'no'}", BOLD))
    print(c("─" * 56, CYAN))

    confirm = input(c("\n  ▶  Start pipeline? (y/n, default=y): ", BOLD)).strip().lower()
    if confirm == "n":
        print(c("\n  Cancelled.\n", YELLOW))
        sys.exit(0)

    run_pipeline(
        file_path  = file_path,
        target     = target,
        metric     = metric,
        time_limit = time_limit,
        visualise  = visualise,
        save       = save,
    )


if __name__ == "__main__":
    main()
