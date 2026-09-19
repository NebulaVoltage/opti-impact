"""Feature importance computation, categorization, and visualization."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from ml.feature_groups import FEATURE_DOMAIN_MAP

DOMAIN_COLORS = {
    "VIBRATION": "#2b5c8f",
    "FREQUENCY": "#3b9ab2",
    "OPTICAL": "#e84a5f",
    "MULTIMODAL": "#e9b872",
}


def compute_permutation_importance(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation feature importance strictly on validation data.

    Args:
        model: Trained classifier.
        X_val: Transformed validation feature matrix.
        y_val: Validation target array.
        feature_names: Feature names corresponding to columns of X_val.
        n_repeats: Number of permutation iterations.
        random_state: Random seed.

    Returns:
        DataFrame containing feature, importance_mean, importance_std, and domain,
        sorted descending by importance_mean.
    """
    perm_res = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="f1_macro",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    rows = []
    for idx, name in enumerate(feature_names):
        mean_imp = float(perm_res.importances_mean[idx])
        std_imp = float(perm_res.importances_std[idx])
        # Find domain
        raw_feat_name = name.replace("_zscore", "")
        domain = FEATURE_DOMAIN_MAP.get(raw_feat_name, "OTHER")

        rows.append(
            {
                "feature": name,
                "importance_mean": mean_imp,
                "importance_std": std_imp,
                "domain": domain,
            }
        )

    df_imp = pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df_imp


def plot_top_feature_importance(
    df_imp: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
    title: str = "Top 15 Feature Importances (Validation Set Permutation)",
) -> Path:
    """Plot top N feature importances color-coded by physical domain.

    Args:
        df_imp: Importance DataFrame.
        output_path: Destination image path.
        top_n: Number of top features to display.
        title: Plot title.

    Returns:
        Resolved output Path.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    top_df = df_imp.head(top_n).iloc[::-1]  # ascending for horizontal bar chart

    features = top_df["feature"].tolist()
    means = top_df["importance_mean"].values
    stds = top_df["importance_std"].values
    domains = top_df["domain"].tolist()

    colors = [DOMAIN_COLORS.get(d, "#999999") for d in domains]

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    bars = ax.barh(features, means, xerr=stds, color=colors, alpha=0.85, capsize=3, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Mean Drop in Macro F1 (Validation Permutation)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Custom legend for domains
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=col, edgecolor="black", label=dom)
        for dom, col in DOMAIN_COLORS.items()
        if dom in domains
    ]
    ax.legend(handles=legend_elements, loc="lower right", title="Feature Modality", frameon=True)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
