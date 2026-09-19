"""Model comparison table generation and visualization."""

from pathlib import Path
from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COMPARISON_COLUMNS: List[str] = [
    "model",
    "cv_accuracy_mean",
    "cv_accuracy_std",
    "cv_macro_f1_mean",
    "cv_macro_f1_std",
    "validation_accuracy",
    "validation_macro_precision",
    "validation_macro_recall",
    "validation_macro_f1",
    "validation_weighted_f1",
    "normal_recall",
    "warning_recall",
    "critical_recall",
    "critical_f1",
]


def format_model_comparison_table(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format evaluation outputs into the standardized model comparison DataFrame.

    Args:
        results_list: List of dictionaries with model metrics.

    Returns:
        DataFrame conforming strictly to COMPARISON_COLUMNS.
    """
    rows = []
    for r in results_list:
        row = {col: r.get(col, 0.0) for col in COMPARISON_COLUMNS}
        rows.append(row)

    df = pd.DataFrame(rows)
    return df[COMPARISON_COLUMNS]


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Generate a side-by-side bar chart of key validation metrics across models.

    Args:
        comparison_df: Model comparison DataFrame.
        output_path: Destination image path.

    Returns:
        Resolved output Path.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    models = comparison_df["model"].tolist()
    n_models = len(models)
    x = np.arange(n_models)
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    val_acc = comparison_df["validation_accuracy"].values
    val_macro_f1 = comparison_df["validation_macro_f1"].values
    val_crit_rec = comparison_df["critical_recall"].values

    rects1 = ax.bar(x - width, val_acc, width, label="Val Accuracy", color="#2b5c8f", alpha=0.9)
    rects2 = ax.bar(x, val_macro_f1, width, label="Val Macro F1", color="#3b9ab2", alpha=0.9)
    rects3 = ax.bar(x + width, val_crit_rec, width, label="CRITICAL Recall", color="#e84a5f", alpha=0.9)

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Candidate Model Performance Comparison (Validation Set)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=11, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    # Add text labels on top of bars
    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
