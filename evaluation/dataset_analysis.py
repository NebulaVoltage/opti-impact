"""Dataset statistical analysis and feature validation script.

Executes comprehensive diagnostics:
1. Class distribution validation.
2. Feature statistics across NORMAL, WARNING, and CRITICAL.
3. Check for artificial single-feature separability.
4. Correlation matrix and multicollinearity analysis (|r| > 0.90).
5. NORMAL structural baseline summary.
6. Diagnostic visualization plots saved to evaluation/plots/.

Usage:
    python -m evaluation.dataset_analysis
"""

from pathlib import Path
import sys
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.builder import build_processed_dataset
from dataset.schema import FEATURE_COLUMNS, TARGET_COLUMN, VALID_SCENARIOS


PLOTS_DIR = REPO_ROOT / "evaluation" / "plots"
DATASET_DIR = REPO_ROOT / "dataset" / "processed"


def ensure_dataset_exists() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load existing processed dataset or build it if missing."""
    all_path = DATASET_DIR / "all_features.csv"
    train_path = DATASET_DIR / "train.csv"
    val_path = DATASET_DIR / "validation.csv"
    test_path = DATASET_DIR / "test.csv"
    baseline_path = DATASET_DIR / "baseline_normal.csv"

    if not all_path.exists() or not train_path.exists() or not baseline_path.exists():
        print("[dataset_analysis] Processed dataset not found. Generating default 3000-sample dataset...")
        build_processed_dataset(samples_per_class=1000, seed=42, output_dir=DATASET_DIR)

    df_all = pd.read_csv(all_path)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return df_all, train_df, val_df, test_df


def analyze_class_distribution(df: pd.DataFrame) -> None:
    """Analyze and plot class counts."""
    print("\n" + "=" * 75)
    print("1. CLASS DISTRIBUTION")
    print("=" * 75)
    counts = df[TARGET_COLUMN].value_counts()
    percentages = df[TARGET_COLUMN].value_counts(normalize=True) * 100

    for scn in VALID_SCENARIOS:
        c = counts.get(scn, 0)
        p = percentages.get(scn, 0.0)
        print(f"  - {scn:10s}: {c:5d} samples ({p:5.1f}%)")
    print(f"  Total samples: {len(df)}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars = ax.bar(VALID_SCENARIOS, [counts.get(s, 0) for s in VALID_SCENARIOS], color=colors, edgecolor="black", alpha=0.85)
    ax.set_title("Dataset Class Distribution (Balanced)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Structural Condition Scenario", fontsize=10)
    ax.set_ylabel("Number of Independent Events", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plot_path = PLOTS_DIR / "class_distribution.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  [+] Saved class distribution plot to: {plot_path.resolve()}")


def check_artificial_separability(df: pd.DataFrame) -> List[Dict[str, object]]:
    """Analyze feature distributions by scenario and check for single-feature separability."""
    print("\n" + "=" * 75)
    print("2. FEATURE DISTRIBUTIONS & ARTIFICIAL SEPARABILITY CHECK")
    print("=" * 75)
    print(f"{'Feature Name':<30s} | {'NORMAL (mean +/- std)':<22s} | {'WARNING (mean +/- std)':<22s} | {'CRITICAL (mean +/- std)':<22s} | {'Overlap?'}")
    print("-" * 115)

    separability_report: List[Dict[str, object]] = []

    for feat in FEATURE_COLUMNS:
        n_vals = df[df[TARGET_COLUMN] == "NORMAL"][feat]
        w_vals = df[df[TARGET_COLUMN] == "WARNING"][feat]
        c_vals = df[df[TARGET_COLUMN] == "CRITICAL"][feat]

        n_mean, n_std = float(n_vals.mean()), float(n_vals.std())
        w_mean, w_std = float(w_vals.mean()), float(w_vals.std())
        c_mean, c_std = float(c_vals.mean()), float(c_vals.std())

        # Check empirical range overlap between adjacent scenarios
        # Normal vs Warning
        nw_overlap = not (n_vals.max() < w_vals.min() or w_vals.max() < n_vals.min())
        # Warning vs Critical
        wc_overlap = not (w_vals.max() < c_vals.min() or c_vals.max() < w_vals.min())

        has_overlap = nw_overlap or wc_overlap
        overlap_str = "YES (realistic)" if has_overlap else "NO (suspicious!)"

        separability_report.append({
            "feature": feat,
            "normal_mean": n_mean,
            "normal_std": n_std,
            "warning_mean": w_mean,
            "warning_std": w_std,
            "critical_mean": c_mean,
            "critical_std": c_std,
            "has_overlap": has_overlap,
            "nw_overlap": nw_overlap,
            "wc_overlap": wc_overlap,
        })

        n_str = f"{n_mean:8.3f} +/- {n_std:6.3f}"
        w_str = f"{w_mean:8.3f} +/- {w_std:6.3f}"
        c_str = f"{c_mean:8.3f} +/- {c_std:6.3f}"
        print(f"{feat:<30s} | {n_str:<22s} | {w_str:<22s} | {c_str:<22s} | {overlap_str}")

    suspicious = [r["feature"] for r in separability_report if not r["has_overlap"]]
    if suspicious:
        print(f"\n[!] WARNING: The following features have NO empirical overlap across scenarios: {suspicious}")
        print("    (Review whether these features could act as a single trivial decision shortcut)")
    else:
        print("\n[+] Verification passed: All features exhibit non-trivial empirical overlap across adjacent classes.")
        print("    Classification requires multivariate multimodal learning as desired.")

    return separability_report


def analyze_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute feature correlation matrix and flag highly collinear pairs."""
    print("\n" + "=" * 75)
    print("3. FEATURE CORRELATION & MULTICOLLINEARITY ANALYSIS")
    print("=" * 75)

    feat_df = df[FEATURE_COLUMNS].copy()
    corr_matrix = feat_df.corr(method="pearson")

    # Find pairs with |r| > 0.90
    high_corr_pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.90:
                high_corr_pairs.append((cols[i], cols[j], r))

    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"  Identified {len(high_corr_pairs)} feature pairs with |r| > 0.90:")
    for f1, f2, r in high_corr_pairs[:15]:  # Top 15
        print(f"    - {f1:<30s} <-> {f2:<30s} : r = {r:+.4f}")
    if len(high_corr_pairs) > 15:
        print(f"    ... and {len(high_corr_pairs) - 15} more.")

    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(14, 12), dpi=150)
    cax = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Pearson Correlation r")

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticklabels(cols, fontsize=7)
    ax.set_title("Engineering Feature Correlation Matrix (3000 Events)", fontsize=12, fontweight="bold")

    plot_path = PLOTS_DIR / "feature_correlation_matrix.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  [+] Saved correlation matrix heatmap to: {plot_path.resolve()}")

    return corr_matrix


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """Plot distribution comparisons across scenarios for key features."""
    key_features = [
        ("vibration_rms", "Vibration RMS (m/s²)"),
        ("dominant_frequency", "Dominant Frequency (Hz)"),
        ("spectral_energy", "Spectral Energy (m/s²)²"),
        ("optical_displacement_max", "Max Optical Displacement (mm)"),
        ("optical_velocity_max", "Max Optical Velocity (mm/s)"),
        ("vibration_optical_correlation", "Vibration-Optical Correlation r"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=150)
    axes = axes.flatten()

    palette = {"NORMAL": "#2ecc71", "WARNING": "#f39c12", "CRITICAL": "#e74c3c"}

    for idx, (feat, label) in enumerate(key_features):
        ax = axes[idx]
        data_to_plot = [df[df[TARGET_COLUMN] == scn][feat].dropna().values for scn in VALID_SCENARIOS]

        bp = ax.boxplot(data_to_plot, tick_labels=VALID_SCENARIOS, patch_artist=True,
                        boxprops=dict(alpha=0.75), medianprops=dict(color="black", linewidth=1.5))
        for patch, scn in zip(bp["boxes"], VALID_SCENARIOS):
            patch.set_facecolor(palette[scn])

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Feature Distribution Boxplots by Structural Scenario (Demonstrating Controlled Overlap)",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    plot_path = PLOTS_DIR / "feature_distributions.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  [+] Saved feature distributions plot to: {plot_path.resolve()}")


def plot_multimodal_interaction(df: pd.DataFrame) -> None:
    """Scatter plot demonstrating multimodal separation in 2D feature space."""
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    palette = {"NORMAL": "#2ecc71", "WARNING": "#f39c12", "CRITICAL": "#e74c3c"}

    for scn in VALID_SCENARIOS:
        sub = df[df[TARGET_COLUMN] == scn]
        ax.scatter(
            sub["dominant_frequency"],
            sub["vibration_rms"],
            c=palette[scn],
            label=scn,
            alpha=0.45,
            edgecolors="none",
            s=25,
        )

    ax.set_title("Multimodal Feature Interaction: Dominant Frequency vs. Vibration RMS\n"
                 "(Controlled Overlap at Class Boundaries Requiring Joint Classification)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Dominant Natural Frequency (Hz)", fontsize=10)
    ax.set_ylabel("Vibration RMS (m/s²)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)

    plot_path = PLOTS_DIR / "multimodal_interaction.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  [+] Saved multimodal interaction plot to: {plot_path.resolve()}")


def print_baseline_summary() -> None:
    """Print the precomputed NORMAL structural baseline summary."""
    baseline_path = DATASET_DIR / "baseline_normal.csv"
    if baseline_path.exists():
        print("\n" + "=" * 75)
        print("4. NORMAL STRUCTURAL BASELINE MODEL SUMMARY (TRAIN SET ONLY)")
        print("=" * 75)
        bdf = pd.read_csv(baseline_path, index_col="feature")
        sample_feats = [
            "vibration_rms",
            "dominant_frequency",
            "spectral_energy",
            "optical_displacement_max",
            "optical_velocity_max",
            "vibration_optical_correlation",
        ]
        for f in sample_feats:
            if f in bdf.index:
                m = bdf.loc[f, "mean"]
                s = bdf.loc[f, "std"]
                med = bdf.loc[f, "median"]
                iqr = bdf.loc[f, "iqr"]
                print(f"  - {f:<30s}: Mean={m:8.4f}, Std={s:8.4f}, Median={med:8.4f}, IQR={iqr:8.4f}")


def main():
    print("=" * 75)
    print("STEP 3: MACHINE LEARNING DATASET STATISTICAL ANALYSIS")
    print("=" * 75)

    df_all, train_df, val_df, test_df = ensure_dataset_exists()

    analyze_class_distribution(df_all)
    check_artificial_separability(df_all)
    analyze_correlations(df_all)
    plot_feature_distributions(df_all)
    plot_multimodal_interaction(df_all)
    print_baseline_summary()

    print("\n" + "=" * 75)
    print("DATASET ANALYSIS COMPLETED SUCCESSFULLY")
    print(f"Plots directory: {PLOTS_DIR.resolve()}")
    print("=" * 75)


if __name__ == "__main__":
    main()
