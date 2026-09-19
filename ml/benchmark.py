"""Benchmarking candidate models across cross-validation and validation sets.

Executes:
  1. 5-fold StratifiedKFold cross-validation strictly on train.csv.
  2. Candidate evaluation on validation.csv.
  3. Generation of evaluation/model_comparison.csv.
  4. Confusion matrices and comparison plots.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from dataset.schema import TARGET_COLUMN, split_features_and_target
from evaluation.confusion_matrix import save_confusion_matrix_pair
from evaluation.metrics import compute_classification_metrics, compute_confusion_matrices
from evaluation.model_comparison import format_model_comparison_table, plot_model_comparison
from ml.models import RANDOM_STATE, get_model_zoo
from ml.preprocessing import StructuralPreprocessor


def run_benchmark(
    train_path: Path = Path("dataset/processed/train.csv"),
    val_path: Path = Path("dataset/processed/validation.csv"),
    output_dir: Path = Path("evaluation"),
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Run full benchmarking on training cross-validation and validation sets.

    Args:
        train_path: Path to train.csv.
        val_path: Path to validation.csv.
        output_dir: Directory for evaluation outputs.
        random_state: Seed for cross-validation splitting.

    Returns:
        DataFrame containing model comparison metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    cm_dir = plots_dir / "confusion_matrix"
    mc_dir = plots_dir / "model_comparison"
    cm_dir.mkdir(parents=True, exist_ok=True)
    mc_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"Loading validation data from {val_path}...")
    val_df = pd.read_csv(val_path)

    X_train_df, y_train_series = split_features_and_target(train_df)
    X_val_df, y_val_series = split_features_and_target(val_df)

    y_train = y_train_series.values
    y_val = y_val_series.values

    model_zoo = get_model_zoo(random_state=random_state)
    results_list: List[Dict[str, Any]] = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    print("\n" + "=" * 70)
    print(f"STARTING BENCHMARK ACROSS {len(model_zoo)} CANDIDATE MODEL FAMILIES")
    print("=" * 70)

    for model_name, model in model_zoo.items():
        print(f"\n[Benchmarking] {model_name}...")

        # 1. 5-Fold Cross-Validation on TRAIN only
        cv_accs = []
        cv_macro_f1s = []
        cv_weighted_f1s = []

        for fold, (train_idx, val_fold_idx) in enumerate(skf.split(X_train_df, y_train), 1):
            fold_X_train = X_train_df.iloc[train_idx]
            fold_y_train = y_train_series.iloc[train_idx]
            fold_X_val = X_train_df.iloc[val_fold_idx]
            fold_y_val = y_train_series.iloc[val_fold_idx]

            # Fit preprocessor on fold train data only
            prep = StructuralPreprocessor(mode="raw", scale=True)
            prep.fit(fold_X_train, fold_y_train)

            X_tr_proc = prep.transform(fold_X_train)
            X_vf_proc = prep.transform(fold_X_val)

            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_tr_proc, fold_y_train.values)
            y_pred_fold = fold_model.predict(X_vf_proc)

            m = compute_classification_metrics(fold_y_val.values, y_pred_fold)
            cv_accs.append(m["accuracy"])
            cv_macro_f1s.append(m["macro_f1"])
            cv_weighted_f1s.append(m["weighted_f1"])

        cv_acc_mean = float(np.mean(cv_accs))
        cv_acc_std = float(np.std(cv_accs))
        cv_f1_mean = float(np.mean(cv_macro_f1s))
        cv_f1_std = float(np.std(cv_macro_f1s))

        print(f"  CV 5-Fold: Accuracy = {cv_acc_mean:.4f} (+/- {cv_acc_std:.4f}) | Macro F1 = {cv_f1_mean:.4f} (+/- {cv_f1_std:.4f})")

        # 2. Fit on full TRAIN split and evaluate on VALIDATION split
        full_prep = StructuralPreprocessor(mode="raw", scale=True)
        full_prep.fit(X_train_df, y_train_series)

        X_train_proc = full_prep.transform(X_train_df)
        X_val_proc = full_prep.transform(X_val_df)

        from sklearn.base import clone
        cand_model = clone(model)
        cand_model.fit(X_train_proc, y_train)
        y_val_pred = cand_model.predict(X_val_proc)

        val_metrics = compute_classification_metrics(y_val, y_val_pred)
        print(f"  Validation: Accuracy = {val_metrics['accuracy']:.4f} | Macro F1 = {val_metrics['macro_f1']:.4f} | CRITICAL Recall = {val_metrics['critical_recall']:.4f}")

        # Confusion matrix for candidate
        raw_cm, norm_cm = compute_confusion_matrices(y_val, y_val_pred)
        save_confusion_matrix_pair(raw_cm, norm_cm, model_name, cm_dir)

        model_record: Dict[str, Any] = {
            "model": model_name,
            "cv_accuracy_mean": cv_acc_mean,
            "cv_accuracy_std": cv_acc_std,
            "cv_macro_f1_mean": cv_f1_mean,
            "cv_macro_f1_std": cv_f1_std,
            "validation_accuracy": val_metrics["accuracy"],
            "validation_macro_precision": val_metrics["macro_precision"],
            "validation_macro_recall": val_metrics["macro_recall"],
            "validation_macro_f1": val_metrics["macro_f1"],
            "validation_weighted_f1": val_metrics["weighted_f1"],
            "normal_recall": val_metrics["normal_recall"],
            "warning_recall": val_metrics["warning_recall"],
            "critical_recall": val_metrics["critical_recall"],
            "critical_f1": val_metrics["critical_f1"],
        }
        results_list.append(model_record)

    comparison_df = format_model_comparison_table(results_list)
    csv_out = output_dir / "model_comparison.csv"
    comparison_df.to_csv(csv_out, index=False, float_format="%.4f")
    print(f"\nSaved model comparison table to {csv_out}")

    plot_out = mc_dir / "model_comparison_validation.png"
    plot_model_comparison(comparison_df, plot_out)
    print(f"Saved model comparison plot to {plot_out}")

    return comparison_df


if __name__ == "__main__":
    run_benchmark()
