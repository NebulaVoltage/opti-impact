"""Multimodal ablation analysis and visualization."""

from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_ablation_results(
    ablation_df: pd.DataFrame,
    output_path: Path,
    title: str = "Multimodal Ablation Study (Validation Performance)",
) -> Path:
    """Plot comparative metrics across feature ablation groups.

    Args:
        ablation_df: DataFrame with columns feature_group, validation_accuracy,
                     validation_macro_f1, normal_recall, warning_recall, critical_recall.
        output_path: Target image path.
        title: Plot title.

    Returns:
        Resolved output Path.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    groups = ablation_df["feature_group"].tolist()
    labels = [
        g.replace("group_", "").replace("_", " ").title() for g in groups
    ]

    x = np.arange(len(groups))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    val_acc = ablation_df["validation_accuracy"].values
    val_macro_f1 = ablation_df["validation_macro_f1"].values
    crit_rec = ablation_df["critical_recall"].values

    rects1 = ax.bar(x - width, val_acc, width, label="Val Accuracy", color="#2b5c8f", alpha=0.9)
    rects2 = ax.bar(x, val_macro_f1, width, label="Val Macro F1", color="#3b9ab2", alpha=0.9)
    rects3 = ax.bar(x + width, crit_rec, width, label="CRITICAL Recall", color="#e84a5f", alpha=0.9)

    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True)

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
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
