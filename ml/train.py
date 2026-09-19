"""Model training, selection rule execution, and artifact serialization.

Applies the predefined model selection rule:
  1. Primary: Validation Macro F1
  2. Secondary: CRITICAL Recall
  3. Tertiary: Validation Accuracy
  4. Tie-breaker: Model Simplicity (logistic_regression > svm_rbf > random_forest > hist_gradient_boosting > xgboost)

Saves finalized model bundle to models/step4/ with full provenance metadata.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from sklearn.base import clone

from dataset.schema import split_features_and_target
from ml.benchmark import run_benchmark
from ml.feature_groups import (
    ABLATION_GROUPS,
    FEATURE_DOMAIN_MAP,
    MULTIMODAL_FEATURES,
    OPTICAL_FEATURES,
    VIBRATION_FREQ_FEATURES,
    VIBRATION_TIME_FEATURES,
)
from ml.model_io import get_software_versions, save_model_bundle
from ml.models import RANDOM_STATE, get_model_zoo
from ml.preprocessing import StructuralPreprocessor

MODEL_COMPLEXITY_ORDER: List[str] = [
    "logistic_regression",
    "svm_rbf",
    "random_forest",
    "hist_gradient_boosting",
    "xgboost",
]


def select_best_model_name(comparison_df: pd.DataFrame, tolerance: float = 1e-4) -> str:
    """Apply the predefined hierarchical model selection rule to choose the winner.

    Args:
        comparison_df: Model comparison DataFrame containing validation metrics.
        tolerance: Numerical tolerance for considering metric scores tied.

    Returns:
        Winning model name string.
    """
    df = comparison_df.copy()

    # Sort descending by primary, secondary, tertiary criteria
    # To handle ties cleanly with simplicity order:
    complexity_map = {name: idx for idx, name in enumerate(MODEL_COMPLEXITY_ORDER)}
    df["complexity_rank"] = df["model"].map(complexity_map).fillna(999)

    # Sort primarily by validation_macro_f1 desc, critical_recall desc, validation_accuracy desc, complexity_rank asc
    df_sorted = df.sort_values(
        by=["validation_macro_f1", "critical_recall", "validation_accuracy", "complexity_rank"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    winner = df_sorted.iloc[0]["model"]
    return winner


def train_and_save_final_model(
    train_path: Path = Path("dataset/processed/train.csv"),
    val_path: Path = Path("dataset/processed/validation.csv"),
    output_dir: Path = Path("models/step4"),
    version_prefix: str = "step4_v1",
    random_state: int = RANDOM_STATE,
) -> Tuple[Any, StructuralPreprocessor, Dict[str, Any]]:
    """Execute benchmark, select winner, train on train.csv, and serialize bundle.

    Args:
        train_path: Path to training dataset.
        val_path: Path to validation dataset.
        output_dir: Target serialization directory.
        version_prefix: Model version tag.
        random_state: Random state seed.

    Returns:
        Tuple of (model, preprocessor, metadata).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run or load benchmark
    comparison_csv = Path("evaluation/model_comparison.csv")
    if comparison_csv.exists():
        comparison_df = pd.read_csv(comparison_csv)
    else:
        comparison_df = run_benchmark(train_path=train_path, val_path=val_path, random_state=random_state)

    # 2. Select winning model
    selected_model_name = select_best_model_name(comparison_df)
    print(f"\n======================================================================")
    print(f"SELECTION RULE EXECUTED")
    print(f"Selected Model: {selected_model_name}")
    print(f"Selection Rule: 1. Val Macro F1 -> 2. Critical Recall -> 3. Val Accuracy -> 4. Simplicity")
    print(f"======================================================================")

    # Retrieve selected model instance from zoo
    model_zoo = get_model_zoo(random_state=random_state)
    raw_model = model_zoo[selected_model_name]

    # 3. Train Preprocessor and Model on train.csv
    train_df = pd.read_csv(train_path)
    X_train_df, y_train_series = split_features_and_target(train_df)

    preprocessor = StructuralPreprocessor(mode="raw", scale=True)
    preprocessor.fit(X_train_df, y_train_series)

    X_train_proc = preprocessor.transform(X_train_df)
    final_model = clone(raw_model)
    final_model.fit(X_train_proc, y_train_series.values)

    # 4. Extract metrics from comparison_df for selected model
    model_row = comparison_df[comparison_df["model"] == selected_model_name].iloc[0].to_dict()

    # 5. Build comprehensive metadata dictionary
    metadata: Dict[str, Any] = {
        "model_type": selected_model_name,
        "model_version": version_prefix,
        "dataset_version": "step3_v1",
        "random_seed": random_state,
        "selection_rule": {
            "primary": "validation_macro_f1",
            "secondary": "critical_recall",
            "tertiary": "validation_accuracy",
            "tie_breaker": "model_simplicity",
        },
        "preprocessing_configuration": {
            "mode": preprocessor.mode,
            "scale": preprocessor.scale,
            "input_feature_count": len(preprocessor.feature_columns),
            "output_feature_count": len(preprocessor.output_feature_names_),
        },
        "feature_names": preprocessor.output_feature_names_,
        "feature_groups": {
            "vibration_time_features": VIBRATION_TIME_FEATURES,
            "vibration_freq_features": VIBRATION_FREQ_FEATURES,
            "optical_features": OPTICAL_FEATURES,
            "multimodal_features": MULTIMODAL_FEATURES,
        },
        "hyperparameters": final_model.get_params(),
        "cross_validation_metrics": {
            "accuracy_mean": float(model_row.get("cv_accuracy_mean", 0.0)),
            "accuracy_std": float(model_row.get("cv_accuracy_std", 0.0)),
            "macro_f1_mean": float(model_row.get("cv_macro_f1_mean", 0.0)),
            "macro_f1_std": float(model_row.get("cv_macro_f1_std", 0.0)),
        },
        "validation_metrics": {
            "accuracy": float(model_row.get("validation_accuracy", 0.0)),
            "macro_precision": float(model_row.get("validation_macro_precision", 0.0)),
            "macro_recall": float(model_row.get("validation_macro_recall", 0.0)),
            "macro_f1": float(model_row.get("validation_macro_f1", 0.0)),
            "weighted_f1": float(model_row.get("validation_weighted_f1", 0.0)),
            "normal_recall": float(model_row.get("normal_recall", 0.0)),
            "warning_recall": float(model_row.get("warning_recall", 0.0)),
            "critical_recall": float(model_row.get("critical_recall", 0.0)),
            "critical_f1": float(model_row.get("critical_f1", 0.0)),
        },
        "software_versions": get_software_versions(),
        "disclaimer": (
            "Trained exclusively on physics-inspired synthetic structural-response data. "
            "Demonstrates software-pipeline feasibility only and does not establish "
            "real-world bridge damage-detection accuracy, structural safety thresholds, "
            "or deployment readiness."
        ),
    }

    # 6. Save bundle
    model_path, preprocessor_path, meta_path = save_model_bundle(
        model=final_model,
        preprocessor=preprocessor,
        metadata=metadata,
        output_dir=output_dir,
        prefix=version_prefix,
    )
    print(f"\nSaved final model bundle:")
    print(f"  Model: {model_path}")
    print(f"  Preprocessor: {preprocessor_path}")
    print(f"  Metadata: {meta_path}")

    return final_model, preprocessor, metadata


if __name__ == "__main__":
    train_and_save_final_model()
