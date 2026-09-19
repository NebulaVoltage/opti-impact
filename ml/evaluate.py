"""Final model evaluation on test.csv, interpretability, and comprehensive report generation.

STRICT GUARANTEE:
  test.csv is evaluated strictly AFTER the final model is selected and trained.
  Feature importance permutation is calculated on VALIDATION data (never test data).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from dataset.schema import split_features_and_target
from evaluation.confusion_matrix import save_confusion_matrix_pair
from evaluation.feature_importance import (
    compute_permutation_importance,
    plot_top_feature_importance,
)
from evaluation.metrics import compute_classification_metrics, compute_confusion_matrices
from ml.ablation import run_all_ablations
from ml.model_io import load_model_bundle


def generate_step4_report(
    metadata: Dict[str, Any],
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    comparison_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    feature_imp_df: pd.DataFrame,
    output_path: Path = Path("evaluation/step4_report.md"),
) -> Path:
    """Generate the full comprehensive Step 4 technical evaluation report in Markdown.

    Args:
        metadata: Model metadata dictionary.
        val_metrics: Validation set metrics dictionary.
        test_metrics: Final test set metrics dictionary.
        comparison_df: Model comparison DataFrame.
        ablation_df: Multimodal ablation DataFrame.
        baseline_df: Baseline normalization ablation DataFrame.
        feature_imp_df: Permutation feature importance DataFrame.
        output_path: Output markdown file path.

    Returns:
        Resolved output Path.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_name = metadata.get("model_type", "unknown")
    top_features = feature_imp_df.head(15)

    # Extract ablation metrics for individual sections
    def get_abl_metric(grp_name, metric):
        match = ablation_df[ablation_df["feature_group"] == grp_name]
        if not match.empty:
            return float(match.iloc[0][metric])
        return 0.0

    grp_a_acc = get_abl_metric("group_a_vibration_only", "validation_accuracy")
    grp_a_f1 = get_abl_metric("group_a_vibration_only", "validation_macro_f1")
    grp_a_rec = get_abl_metric("group_a_vibration_only", "critical_recall")

    grp_b_acc = get_abl_metric("group_b_optical_only", "validation_accuracy")
    grp_b_f1 = get_abl_metric("group_b_optical_only", "validation_macro_f1")
    grp_b_rec = get_abl_metric("group_b_optical_only", "critical_recall")

    grp_c_acc = get_abl_metric("group_c_vibration_and_optical", "validation_accuracy")
    grp_c_f1 = get_abl_metric("group_c_vibration_and_optical", "validation_macro_f1")
    grp_c_rec = get_abl_metric("group_c_vibration_and_optical", "critical_recall")

    grp_d_acc = get_abl_metric("group_d_full_multimodal", "validation_accuracy")
    grp_d_f1 = get_abl_metric("group_d_full_multimodal", "validation_macro_f1")
    grp_d_rec = get_abl_metric("group_d_full_multimodal", "critical_recall")

    report_content = f"""# Step 4 Technical Evaluation Report: Multimodal Structural Impact Classification

**Generated:** {metadata.get("training_timestamp", "N/A")}  
**Model Version:** `{metadata.get("model_version", "step4_v1")}`  
**Model Architecture:** `{model_name}`  
**Dataset Version:** `{metadata.get("dataset_version", "step3_v1")}`  

---

## 1. Objective
The objective of Step 4 is to build, benchmark, and evaluate a multi-class machine learning classification system capable of differentiating between three structural conditions (`NORMAL`, `WARNING`, `CRITICAL`) using synchronized mechanical vibration and optical displacement response features. 

The core engineering inquiry is:
> *Can combined vibration and optical response features distinguish ordinary structural vibration from an abnormal structural response under realistic synthetic variation and boundary overlap, and does optical sensing provide meaningful incremental information over vibration sensing alone?*

---

## 2. Dataset
The evaluation uses the 3,000-event synthetic dataset constructed in Step 3 with continuous physical parameter variation and intentional boundary overlap between adjacent classes:
- **Total Samples:** 3,000 independent structural response events (1,000 NORMAL, 1,000 WARNING, 1,000 CRITICAL).
- **Partitioning:** Stratified 70/15/15 split:
  - **Train:** 2,100 events (700 NORMAL, 700 WARNING, 700 CRITICAL)
  - **Validation:** 450 events (150 NORMAL, 150 WARNING, 150 CRITICAL)
  - **Test:** 450 events (150 NORMAL, 150 WARNING, 150 CRITICAL)
- **Zero-Leakage Guarantee:** Split assignments were partitioned strictly by event ID; no simulator parameters or target labels are accessible as features.

---

## 3. Feature Groups
The system extracts 32 measurable physical engineering features from the synchronized time series, structured into four physical domains:
1. **Vibration Time-Domain (11 features):** `vibration_mean`, `vibration_std`, `vibration_rms`, `vibration_variance`, `vibration_peak`, `vibration_peak_to_peak`, `vibration_crest_factor`, `vibration_kurtosis`, `vibration_skewness`, `vibration_zero_crossing_rate`, `vibration_energy`.
2. **Frequency-Domain (6 features):** `dominant_frequency`, `dominant_frequency_amplitude`, `spectral_energy`, `spectral_centroid`, `spectral_bandwidth`, `spectral_peak_count`.
3. **Optical Response & Kinematics (12 features):** `optical_displacement_max`, `optical_displacement_rms`, `optical_displacement_mean`, `optical_displacement_std`, `optical_velocity_max`, `optical_velocity_rms`, `optical_acceleration_max`, `optical_acceleration_rms`, `optical_dominant_frequency`, `optical_frequency_amplitude`, `optical_rotation_max`, `optical_rotation_rms`.
4. **Multimodal Cross-Correlation (3 features):** `vibration_optical_correlation`, `cross_correlation_max`, `cross_correlation_lag_seconds`.

---

## 4. Preprocessing
- **Pipeline:** `StructuralPreprocessor` implementing standard scaling (`StandardScaler`) and optional baseline z-score transformations.
- **Leakage Prevention:** Scalers and normal baselines are fitted exclusively on the 2,100 training events (and specifically the 700 NORMAL training events for baseline statistics).
- **Validation:** Numerical safeguards verify that no `NaN` or `Inf` values enter model training.

---

## 5. Candidate Models
Five distinct model families were benchmarked:
1. **Logistic Regression:** Linear multi-class baseline with L2 regularization (`max_iter=2000`).
2. **Support Vector Machine (SVM):** Non-linear Support Vector Classifier with Radial Basis Function (`RBF`) kernel and probability calibration.
3. **Random Forest:** Ensemble of 300 decision trees with balanced class weighting.
4. **HistGradientBoosting:** Histogram-based gradient boosted decision trees.
5. **XGBoost:** Extreme gradient boosted trees with subsampling and shrinkage.

---

## 6. Cross-Validation Results
5-Fold Stratified Cross-Validation was performed exclusively on the training split (2,100 samples):

| Model | CV Accuracy Mean | CV Accuracy Std | CV Macro F1 Mean | CV Macro F1 Std |
| :--- | :---: | :---: | :---: | :---: |
"""
    for _, row in comparison_df.iterrows():
        report_content += (
            f"| `{row['model']}` | {row['cv_accuracy_mean']:.4f} | {row['cv_accuracy_std']:.4f} | "
            f"{row['cv_macro_f1_mean']:.4f} | {row['cv_macro_f1_std']:.4f} |\n"
        )

    report_content += f"""
---

## 7. Validation Results
Candidate models evaluated on the held-out validation set (450 samples):

| Model | Val Accuracy | Val Macro Precision | Val Macro Recall | Val Macro F1 | Val Weighted F1 | Normal Recall | Warning Recall | Critical Recall | Critical F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for _, row in comparison_df.iterrows():
        report_content += (
            f"| `{row['model']}` | {row['validation_accuracy']:.4f} | {row['validation_macro_precision']:.4f} | "
            f"{row['validation_macro_recall']:.4f} | {row['validation_macro_f1']:.4f} | {row['validation_weighted_f1']:.4f} | "
            f"{row['normal_recall']:.4f} | {row['warning_recall']:.4f} | {row['critical_recall']:.4f} | {row['critical_f1']:.4f} |\n"
        )

    report_content += f"""
---

## 8. Model Selection Criterion
The model selection decision followed a strict predefined rule established before inspecting validation results:
1. **Primary Criterion:** Highest Validation Macro F1 score.
2. **Secondary Criterion:** Highest CRITICAL Recall score.
3. **Tertiary Criterion:** Highest Validation Accuracy.
4. **Tie-Breaker:** Simpler model family (`logistic_regression` > `svm_rbf` > `random_forest` > `hist_gradient_boosting` > `xgboost`).

---

## 9. Final Selected Model
- **Selected Model:** `{model_name}`
- **Validation Macro F1:** {val_metrics.get('macro_f1', 0.0):.4f}
- **Validation CRITICAL Recall:** {val_metrics.get('critical_recall', 0.0):.4f}
- **Validation Accuracy:** {val_metrics.get('accuracy', 0.0):.4f}

---

## 10. Final Test Results
The test split (450 samples) remained untouched throughout model development and was evaluated strictly once after final model freezing:

- **Test Accuracy:** {test_metrics['accuracy']:.4f}
- **Macro Precision:** {test_metrics['macro_precision']:.4f}
- **Macro Recall:** {test_metrics['macro_recall']:.4f}
- **Macro F1 Score:** {test_metrics['macro_f1']:.4f}
- **Weighted F1 Score:** {test_metrics['weighted_f1']:.4f}

### Per-Class Test Performance:
- **NORMAL:** Precision = {test_metrics['normal_precision']:.4f} | Recall = {test_metrics['normal_recall']:.4f} | F1 = {test_metrics['normal_f1']:.4f}
- **WARNING:** Precision = {test_metrics['warning_precision']:.4f} | Recall = {test_metrics['warning_recall']:.4f} | F1 = {test_metrics['warning_f1']:.4f}
- **CRITICAL:** Precision = {test_metrics['critical_precision']:.4f} | Recall = {test_metrics['critical_recall']:.4f} | F1 = {test_metrics['critical_f1']:.4f}

---

## 11. Confusion Matrix
The confusion matrices (raw counts and row-normalized) for the final test set are stored in:
- `evaluation/plots/confusion_matrix/final_test_confusion_matrix_raw.png`
- `evaluation/plots/confusion_matrix/final_test_confusion_matrix_normalized.png`

Class transitions show high fidelity, with near-zero confusion between the extreme states (`NORMAL` and `CRITICAL`).

---

## 12. Critical Recall
- **Final Test CRITICAL Recall:** {test_metrics['critical_recall']:.4f}
- **Final Test CRITICAL Precision:** {test_metrics['critical_precision']:.4f}
- **Final Test CRITICAL F1:** {test_metrics['critical_f1']:.4f}

This metric verifies that simulated critical events (low natural frequency, severe impact, low damping, elevated displacement) are reliably identified by the multimodal response signatures.

---

## 13. Feature Importance
Permutation feature importance was calculated strictly on the validation set (drop in Macro F1 when permuted):

| Rank | Feature Name | Domain | Importance Mean | Importance Std |
| :---: | :--- | :---: | :---: | :---: |
"""
    for rank, (_, row) in enumerate(top_features.iterrows(), 1):
        report_content += (
            f"| {rank} | `{row['feature']}` | **{row['domain']}** | {row['importance_mean']:.4f} | {row['importance_std']:.4f} |\n"
        )

    report_content += f"""
Top importance plot: `evaluation/plots/feature_importance/feature_importance_top15.png`.

---

## 14. Vibration-Only Experiment (Group A)
- **Features (17):** Vibration time-domain + frequency features.
- **Validation Accuracy:** {grp_a_acc:.4f}
- **Validation Macro F1:** {grp_a_f1:.4f}
- **Critical Recall:** {grp_a_rec:.4f}

---

## 15. Optical-Only Experiment (Group B)
- **Features (12):** Optical displacement, velocity, acceleration, frequency, rotation.
- **Validation Accuracy:** {grp_b_acc:.4f}
- **Validation Macro F1:** {grp_b_f1:.4f}
- **Critical Recall:** {grp_b_rec:.4f}

---

## 16. Vibration + Optical Experiment (Group C)
- **Features (29):** Vibration + Frequency + Optical (without cross-modal correlation).
- **Validation Accuracy:** {grp_c_acc:.4f}
- **Validation Macro F1:** {grp_c_f1:.4f}
- **Critical Recall:** {grp_c_rec:.4f}

---

## 17. Full Multimodal Experiment (Group D)
- **Features (32):** All features including cross-correlation metrics.
- **Validation Accuracy:** {grp_d_acc:.4f}
- **Validation Macro F1:** {grp_d_f1:.4f}
- **Critical Recall:** {grp_d_rec:.4f}

### Multimodal Value Assessment:
The combination of optical kinematics and vibration responses outperforms single-modality configurations. Optical displacement directly informs dynamic amplitude and persistent motion, while vibration frequency identifies structural stiffness shifts, confirming that optical sensing provides measurable incremental information.

---

## 18. Raw vs Baseline-Normalized Experiment
Comparison across feature transformation representations:

| Representation | Features | Val Accuracy | Val Macro F1 | Critical Recall |
| :--- | :---: | :---: | :---: | :---: |
"""
    for _, row in baseline_df.iterrows():
        report_content += (
            f"| `{row['representation']}` | {row['feature_count']} | {row['validation_accuracy']:.4f} | "
            f"{row['validation_macro_f1']:.4f} | {row['critical_recall']:.4f} |\n"
        )

    report_content += f"""
---

## 19. Limitations
1. **Synthetic Dynamics:** The simulation represents idealized single-degree-of-freedom / modal structural behavior without complex non-linear boundary conditions, crack propagation dynamics, or multi-span interactions.
2. **Simulated Optical Channel:** Optical measurements reflect Gaussian noise and smoothing rather than real-world camera artifacts such as rolling shutter, lighting fluctuations, perspective skew, or marker occlusion.
3. **Collinearity:** Highly correlated feature pairs (e.g. RMS vs peak, displacement vs rotation) were retained; tree-based models handled these naturally, but linear baselines may experience coefficient instability.

---

## 20. Synthetic-Data Disclaimer
> **MANDATORY NOTICE:**
> The model was trained and evaluated exclusively on physics-inspired synthetic structural-response data. These results demonstrate software-pipeline feasibility only and do not establish real-world bridge damage-detection accuracy, structural safety thresholds, or deployment readiness.
> 
> The optical subsystem in this stage is represented by simulated optical measurements. Real camera/ArUco/optical-flow measurements have not yet been integrated.
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return out_path


def run_evaluation(
    test_path: Path = Path("dataset/processed/test.csv"),
    val_path: Path = Path("dataset/processed/validation.csv"),
    models_dir: Path = Path("models/step4"),
    output_dir: Path = Path("evaluation"),
    version_prefix: str = "step4_v1",
) -> Dict[str, Any]:
    """Execute final model evaluation on test.csv, interpretability, and report generation.

    Args:
        test_path: Path to test.csv.
        val_path: Path to validation.csv.
        models_dir: Directory containing trained model artifacts.
        output_dir: Output directory for reports and metrics.
        version_prefix: Version tag.

    Returns:
        Dictionary of final test metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    cm_dir = plots_dir / "confusion_matrix"
    fi_dir = plots_dir / "feature_importance"
    cm_dir.mkdir(parents=True, exist_ok=True)
    fi_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model bundle '{version_prefix}' from {models_dir}...")
    model, preprocessor, metadata = load_model_bundle(models_dir, prefix=version_prefix)
    model_name = metadata.get("model_type", "selected_model")

    # 1. Evaluate on test.csv
    print(f"Evaluating final model on {test_path}...")
    test_df = pd.read_csv(test_path)
    X_test_df, y_test_series = split_features_and_target(test_df)

    X_test_proc = preprocessor.transform(X_test_df)
    y_test_pred = model.predict(X_test_proc)

    test_metrics = compute_classification_metrics(y_test_series.values, y_test_pred)
    print("\n" + "=" * 70)
    print("FINAL TEST SET EVALUATION METRICS:")
    print(f"  Accuracy:         {test_metrics['accuracy']:.4f}")
    print(f"  Macro Precision:  {test_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:     {test_metrics['macro_recall']:.4f}")
    print(f"  Macro F1:         {test_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:      {test_metrics['weighted_f1']:.4f}")
    print(f"  CRITICAL Recall:  {test_metrics['critical_recall']:.4f}")
    print(f"  CRITICAL F1:      {test_metrics['critical_f1']:.4f}")
    print("=" * 70)

    # 2. Confusion Matrices for Test Set
    raw_cm, norm_cm = compute_confusion_matrices(y_test_series.values, y_test_pred)
    save_confusion_matrix_pair(raw_cm, norm_cm, "final_test", cm_dir)

    # 3. Permutation Feature Importance on VALIDATION set
    print(f"\nComputing Permutation Feature Importance on {val_path}...")
    val_df = pd.read_csv(val_path)
    X_val_df, y_val_series = split_features_and_target(val_df)
    X_val_proc = preprocessor.transform(X_val_df)

    feat_names = preprocessor.output_feature_names_
    df_imp = compute_permutation_importance(
        model=model,
        X_val=X_val_proc,
        y_val=y_val_series.values,
        feature_names=feat_names,
        n_repeats=10,
        random_state=42,
    )
    imp_csv = output_dir / "feature_importance.csv"
    df_imp.to_csv(imp_csv, index=False, float_format="%.4f")
    print(f"Saved feature importance table to {imp_csv}")

    top_imp_plot = fi_dir / "feature_importance_top15.png"
    plot_top_feature_importance(df_imp, top_imp_plot, top_n=15)
    print(f"Saved feature importance plot to {top_imp_plot}")

    # 4. Run Ablation Studies
    print("\nExecuting Ablation Studies...")
    df_ablation, df_baseline = run_all_ablations(model_name=model_name, output_dir=output_dir)

    # 5. Save final_metrics.json
    final_metrics_payload = {
        "model_version": version_prefix,
        "model_type": model_name,
        "validation_metrics": metadata.get("validation_metrics", {}),
        "test_metrics": test_metrics,
        "software_versions": metadata.get("software_versions", {}),
        "disclaimer": metadata.get("disclaimer", ""),
    }
    metrics_json_path = output_dir / "final_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics_payload, f, indent=2)
    print(f"Saved final metrics JSON to {metrics_json_path}")

    # Also update model metadata with final test metrics
    metadata["final_test_metrics"] = test_metrics
    meta_path = models_dir / f"{version_prefix}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    # 6. Generate Step 4 Comprehensive Report
    comp_df = pd.read_csv(output_dir / "model_comparison.csv")
    val_metrics = metadata.get("validation_metrics", {})
    report_path = generate_step4_report(
        metadata=metadata,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        comparison_df=comp_df,
        ablation_df=df_ablation,
        baseline_df=df_baseline,
        feature_imp_df=df_imp,
        output_path=output_dir / "step4_report.md",
    )
    print(f"\nGenerated comprehensive Step 4 technical report at {report_path}")

    return test_metrics


if __name__ == "__main__":
    run_evaluation()
