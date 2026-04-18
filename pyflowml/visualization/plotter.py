"""
Plotter — EDA visualizations: distributions, heatmaps, pairplots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["figure.facecolor"] = "#1a1a2e"
plt.rcParams["axes.facecolor"] = "#16213e"
plt.rcParams["text.color"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"


class Plotter:
    """
    EDA visualization tools.

    Example
    -------
    >>> Plotter.distribution(df, column="age")
    >>> Plotter.correlation_heatmap(df)
    >>> Plotter.pairplot(df, hue="target")
    """

    @staticmethod
    def distribution(df: pd.DataFrame, column: str, bins: int = 30,
                     figsize=(10, 5), save_path: str = None, return_fig: bool = False):
        """Plot the distribution of a single column."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f"Distribution: {column}", color="white", fontsize=14, fontweight="bold")

        # Histogram
        axes[0].hist(df[column].dropna(), bins=bins, color="#e94560", edgecolor="white", alpha=0.8)
        axes[0].set_title("Histogram", color="white")
        axes[0].set_xlabel(column, color="white")

        # Box plot
        axes[1].boxplot(df[column].dropna(), patch_artist=True,
                        boxprops=dict(facecolor="#0f3460", color="white"),
                        medianprops=dict(color="#e94560", linewidth=2))
        axes[1].set_title("Box Plot", color="white")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if return_fig:
            return fig
        plt.show()
        return fig

    @staticmethod
    def correlation_heatmap(df: pd.DataFrame, figsize=(12, 10), save_path: str = None,
                            return_fig: bool = False):
        """Plot correlation heatmap for all numeric columns."""
        numeric = df.select_dtypes(include=[np.number])
        corr = numeric.corr()

        fig, ax = plt.subplots(figsize=figsize)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", vmin=-1, vmax=1,
            linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Correlation Heatmap", color="white", fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if return_fig:
            return fig
        plt.show()
        return fig

    @staticmethod
    def pairplot(df: pd.DataFrame, hue: str = None, columns: list = None,
                 save_path: str = None):
        """Pairplot of selected numeric columns."""
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if columns:
            numeric = [c for c in columns if c in numeric]
        if hue:
            numeric = [c for c in numeric if c != hue]
            plot_df = df[numeric + [hue]].dropna()
        else:
            plot_df = df[numeric].dropna()

        g = sns.pairplot(plot_df, hue=hue, diag_kind="kde",
                         plot_kws={"alpha": 0.6},
                         palette="Set2")
        g.figure.suptitle("Pair Plot", y=1.02, color="white", fontsize=14)
        if save_path:
            g.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    @staticmethod
    def missing_values(df: pd.DataFrame, figsize=(12, 5), save_path: str = None,
                       return_fig: bool = False):
        """Plot missing values per column as a bar chart."""
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            print("No missing values found.")
            return None

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(missing.index, missing.values / len(df) * 100,
               color="#e94560", edgecolor="white")
        ax.set_xlabel("Column", color="white")
        ax.set_ylabel("Missing %", color="white")
        ax.set_title("Missing Values by Column", color="white", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if return_fig:
            return fig
        plt.show()
        return fig
