"""Multimodal and baseline feature ablation studies.

Assesses:
  1. Multimodal Ablation: Measures incremental contribution of optical sensing:
     - Group A: Vibration Only (17 features)
     - Group B: Optical Only (12 features)
     - Group C: Vibration + Optical (29 features)
     - Group D: Full Multimodal (32 features)
  2. Baseline Ablation: Evaluates raw features vs baseline-normalized z-scores vs augmented.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from sklearn.base import clone

from dataset.schema import split_features_and_target
from evaluation.ablation_analysis import plot_ablation_results
from evaluation.baseline_analysis import plot_baseline_comparison
from evaluation.metrics import compute_classification_metrics
from ml.feature_groups import ABLATION_GROUPS
from ml.models import RANDOM_STATE, get_model_zoo
from ml.preprocessing import StructuralPreprocessor


def run_multimodal_ablation(
    model_name: str,
    train_path: Path = Path("dataset/processed/train.csv"),
    val_path: Path = Path("dataset/processed/validation.csv"),
    output_dir: Path = Path("evaluation"),
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Run ablation across Groups A, B, C, and D using a specific model family.

    Args:
        model_name: Model name key from get_model_zoo().
        train_path: Path to training data.
        val_path: Path to validation data.
        output_dir: Evaluation output directory.
        random_state: Seed.

    Returns:
        DataFrame with ablation metrics.
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train_df, y_train_series = split_features_and_target(train_df)
    X_val_df, y_val_series = split_features_and_target(val_df)

    model_zoo = get_model_zoo(random_state=random_state)
    base_model = model_zoo[model_name]

    records = []

    for group_key, feat_cols in ABLATION_GROUPS.items():
        print(f"  [Ablation] Testing {group_key} ({len(feat_cols)} features)...")

        prep = StructuralPreprocessor(feature_columns=feat_cols, mode="raw", scale=True)
        prep.fit(X_train_df, y_train_series)

        X_tr = prep.transform(X_train_df)
        X_val = prep.transform(X_val_df)

        model = clone(base_model)
        model.fit(X_tr, y_train_series.values)

        y_pred = model.predict(X_val)
        m = compute_classification_metrics(y_val_series.values, y_pred)

        records.append(
            {
                "feature_group": group_key,
                "feature_count": len(feat_cols),
                "validation_accuracy": m["accuracy"],
                "validation_macro_f1": m["macro_f1"],
                "normal_recall": m["normal_recall"],
                "warning_recall": m["warning_recall"],
                "critical_recall": m["critical_recall"],
                "critical_f1": m["critical_f1"],
            }
        )

    df_results = pd.DataFrame(records)
    out_csv = output_dir / "ablation_results.csv"
    df_results.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"Saved multimodal ablation results to {out_csv}")

    ablation_plot_path = output_dir / "plots" / "ablation" / "multimodal_ablation.png"
    plot_ablation_results(df_results, ablation_plot_path)
    print(f"Saved multimodal ablation plot to {ablation_plot_path}")

    return df_results


def run_baseline_ablation(
    model_name: str,
    train_path: Path = Path("dataset/processed/train.csv"),
    val_path: Path = Path("dataset/processed/validation.csv"),
    output_dir: Path = Path("evaluation"),
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Run baseline feature representation comparison (raw vs baseline_normalized vs augmented).

    Args:
        model_name: Model name key.
        train_path: Path to training data.
        val_path: Path to validation data.
        output_dir: Evaluation output directory.
        random_state: Seed.

    Returns:
        DataFrame with baseline comparison metrics.
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train_df, y_train_series = split_features_and_target(train_df)
    X_val_df, y_val_series = split_features_and_target(val_df)

    model_zoo = get_model_zoo(random_state=random_state)
    base_model = model_zoo[model_name]

    modes = ["raw", "baseline_normalized", "augmented"]
    records = []

    for mode in modes:
        print(f"  [Baseline Ablation] Testing mode='{mode}'...")

        prep = StructuralPreprocessor(mode=mode, scale=True)
        prep.fit(X_train_df, y_train_series)

        X_tr = prep.transform(X_train_df)
        X_val = prep.transform(X_val_df)

        model = clone(base_model)
        model.fit(X_tr, y_train_series.values)

        y_pred = model.predict(X_val)
        m = compute_classification_metrics(y_val_series.values, y_pred)

        records.append(
            {
                "representation": mode,
                "feature_count": X_tr.shape[1],
                "validation_accuracy": m["accuracy"],
                "validation_macro_f1": m["macro_f1"],
                "normal_recall": m["normal_recall"],
                "warning_recall": m["warning_recall"],
                "critical_recall": m["critical_recall"],
                "critical_f1": m["critical_f1"],
            }
        )

    df_results = pd.DataFrame(records)
    out_csv = output_dir / "baseline_comparison.csv"
    df_results.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"Saved baseline comparison results to {out_csv}")

    baseline_plot_path = output_dir / "plots" / "ablation" / "baseline_comparison.png"
    plot_baseline_comparison(df_results, baseline_plot_path)
    print(f"Saved baseline comparison plot to {baseline_plot_path}")

    return df_results


def run_all_ablations(
    model_name: Optional[str] = None,
    output_dir: Path = Path("evaluation"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute all ablation studies."""
    if model_name is None:
        # Load from metadata or model_comparison
        meta_file = Path("models/step4/step4_v1_metadata.json")
        if meta_file.exists():
            with open(meta_file, "r") as f:
                meta = json.load(f)
                model_name = meta.get("model_type", "random_forest")
        else:
            model_name = "random_forest"

    print(f"\nRunning Ablation Studies with model: {model_name}")
    df_multi = run_multimodal_ablation(model_name=model_name, output_dir=output_dir)
    df_base = run_baseline_ablation(model_name=model_name, output_dir=output_dir)
    return df_multi, df_base


if __name__ == "__main__":
    run_all_ablations()
