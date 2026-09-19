"""Plotting and visualization utilities for confusion matrices."""

from pathlib import Path
from typing import List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.metrics import CLASS_ORDER


def plot_confusion_matrix(
    cm_df: pd.DataFrame,
    title: str,
    output_path: Path,
    normalize: bool = False,
    cmap: str = "Blues",
) -> Path:
    """Plot and save a formatted confusion matrix heatmap.

    Args:
        cm_df: Confusion matrix DataFrame with rows as True labels, cols as Predicted.
        title: Title of the figure.
        output_path: Destination file path.
        normalize: Whether values represent normalized proportions.
        cmap: Matplotlib colormap name.

    Returns:
        Resolved output Path.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    data = cm_df.values
    im = ax.imshow(data, interpolation="nearest", cmap=cmap)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion" if normalize else "Count", rotation=270, labelpad=15)

    classes = list(cm_df.index)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=11, fontweight="bold")

    thresh = data.max() / 2.0 if data.max() > 0 else 0.5
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text = f"{val:.3f}" if normalize else f"{int(val)}"
            color = "white" if val > thresh else "black"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=color,
                fontsize=12,
                fontweight="bold",
            )

    ax.set_ylabel("True Structural State", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_xlabel("Predicted Structural State", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def save_confusion_matrix_pair(
    raw_cm: pd.DataFrame,
    norm_cm: pd.DataFrame,
    model_name: str,
    output_dir: Path,
) -> List[Path]:
    """Save both raw count and row-normalized confusion matrix plots for a model.

    Args:
        raw_cm: Raw count confusion matrix.
        norm_cm: Row-normalized confusion matrix.
        model_name: Label or identifier for the model.
        output_dir: Output directory.

    Returns:
        List of generated image paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / f"{model_name}_confusion_matrix_raw.png"
    norm_path = out_dir / f"{model_name}_confusion_matrix_normalized.png"

    plot_confusion_matrix(
        raw_cm,
        title=f"Confusion Matrix (Counts): {model_name}",
        output_path=raw_path,
        normalize=False,
    )
    plot_confusion_matrix(
        norm_cm,
        title=f"Confusion Matrix (Normalized): {model_name}",
        output_path=norm_path,
        normalize=True,
    )

    return [raw_path, norm_path]
