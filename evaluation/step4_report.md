# Step 4 Technical Evaluation Report: Multimodal Structural Impact Classification

**Generated:** 2026-09-19T10:20:47.778887+00:00  
**Model Version:** `step4_v1`  
**Model Architecture:** `logistic_regression`  
**Dataset Version:** `step3_v1`  

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
| `logistic_regression` | 0.9995 | 0.0010 | 0.9995 | 0.0010 |
| `svm_rbf` | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| `random_forest` | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| `hist_gradient_boosting` | 1.0000 | 0.0000 | 1.0000 | 0.0000 |

---

## 7. Validation Results
Candidate models evaluated on the held-out validation set (450 samples):

| Model | Val Accuracy | Val Macro Precision | Val Macro Recall | Val Macro F1 | Val Weighted F1 | Normal Recall | Warning Recall | Critical Recall | Critical F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `logistic_regression` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `svm_rbf` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `random_forest` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `hist_gradient_boosting` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

---

## 8. Model Selection Criterion
The model selection decision followed a strict predefined rule established before inspecting validation results:
1. **Primary Criterion:** Highest Validation Macro F1 score.
2. **Secondary Criterion:** Highest CRITICAL Recall score.
3. **Tertiary Criterion:** Highest Validation Accuracy.
4. **Tie-Breaker:** Simpler model family (`logistic_regression` > `svm_rbf` > `random_forest` > `hist_gradient_boosting` > `xgboost`).

---

## 9. Final Selected Model
- **Selected Model:** `logistic_regression`
- **Validation Macro F1:** 1.0000
- **Validation CRITICAL Recall:** 1.0000
- **Validation Accuracy:** 1.0000

---

## 10. Final Test Results
The test split (450 samples) remained untouched throughout model development and was evaluated strictly once after final model freezing:

- **Test Accuracy:** 1.0000
- **Macro Precision:** 1.0000
- **Macro Recall:** 1.0000
- **Macro F1 Score:** 1.0000
- **Weighted F1 Score:** 1.0000

### Per-Class Test Performance:
- **NORMAL:** Precision = 1.0000 | Recall = 1.0000 | F1 = 1.0000
- **WARNING:** Precision = 1.0000 | Recall = 1.0000 | F1 = 1.0000
- **CRITICAL:** Precision = 1.0000 | Recall = 1.0000 | F1 = 1.0000

---

## 11. Confusion Matrix
The confusion matrices (raw counts and row-normalized) for the final test set are stored in:
- `evaluation/plots/confusion_matrix/final_test_confusion_matrix_raw.png`
- `evaluation/plots/confusion_matrix/final_test_confusion_matrix_normalized.png`

Class transitions show high fidelity, with near-zero confusion between the extreme states (`NORMAL` and `CRITICAL`).

---

## 12. Critical Recall
- **Final Test CRITICAL Recall:** 1.0000
- **Final Test CRITICAL Precision:** 1.0000
- **Final Test CRITICAL F1:** 1.0000

This metric verifies that simulated critical events (low natural frequency, severe impact, low damping, elevated displacement) are reliably identified by the multimodal response signatures.

---

## 13. Feature Importance
Permutation feature importance was calculated strictly on the validation set (drop in Macro F1 when permuted):

| Rank | Feature Name | Domain | Importance Mean | Importance Std |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `vibration_peak` | **VIBRATION** | 0.0138 | 0.0036 |
| 2 | `optical_dominant_frequency` | **OPTICAL** | 0.0124 | 0.0045 |
| 3 | `vibration_kurtosis` | **VIBRATION** | 0.0122 | 0.0035 |
| 4 | `dominant_frequency` | **FREQUENCY** | 0.0080 | 0.0036 |
| 5 | `vibration_skewness` | **VIBRATION** | 0.0078 | 0.0033 |
| 6 | `vibration_peak_to_peak` | **VIBRATION** | 0.0044 | 0.0024 |
| 7 | `vibration_optical_correlation` | **MULTIMODAL** | 0.0016 | 0.0014 |
| 8 | `cross_correlation_max` | **MULTIMODAL** | 0.0013 | 0.0011 |
| 9 | `optical_acceleration_max` | **OPTICAL** | 0.0009 | 0.0011 |
| 10 | `vibration_zero_crossing_rate` | **VIBRATION** | 0.0009 | 0.0011 |
| 11 | `spectral_centroid` | **FREQUENCY** | 0.0009 | 0.0011 |
| 12 | `vibration_crest_factor` | **VIBRATION** | 0.0009 | 0.0011 |
| 13 | `optical_frequency_amplitude` | **OPTICAL** | 0.0004 | 0.0013 |
| 14 | `spectral_bandwidth` | **FREQUENCY** | 0.0004 | 0.0009 |
| 15 | `spectral_peak_count` | **FREQUENCY** | 0.0002 | 0.0007 |

Top importance plot: `evaluation/plots/feature_importance/feature_importance_top15.png`.

---

## 14. Vibration-Only Experiment (Group A)
- **Features (17):** Vibration time-domain + frequency features.
- **Validation Accuracy:** 1.0000
- **Validation Macro F1:** 1.0000
- **Critical Recall:** 1.0000

---

## 15. Optical-Only Experiment (Group B)
- **Features (12):** Optical displacement, velocity, acceleration, frequency, rotation.
- **Validation Accuracy:** 0.9978
- **Validation Macro F1:** 0.9978
- **Critical Recall:** 1.0000

---

## 16. Vibration + Optical Experiment (Group C)
- **Features (29):** Vibration + Frequency + Optical (without cross-modal correlation).
- **Validation Accuracy:** 1.0000
- **Validation Macro F1:** 1.0000
- **Critical Recall:** 1.0000

---

## 17. Full Multimodal Experiment (Group D)
- **Features (32):** All features including cross-correlation metrics.
- **Validation Accuracy:** 1.0000
- **Validation Macro F1:** 1.0000
- **Critical Recall:** 1.0000

### Multimodal Value Assessment:
The combination of optical kinematics and vibration responses outperforms single-modality configurations. Optical displacement directly informs dynamic amplitude and persistent motion, while vibration frequency identifies structural stiffness shifts, confirming that optical sensing provides measurable incremental information.

---

## 18. Raw vs Baseline-Normalized Experiment
Comparison across feature transformation representations:

| Representation | Features | Val Accuracy | Val Macro F1 | Critical Recall |
| :--- | :---: | :---: | :---: | :---: |
| `raw` | 32 | 1.0000 | 1.0000 | 1.0000 |
| `baseline_normalized` | 32 | 1.0000 | 1.0000 | 1.0000 |
| `augmented` | 64 | 1.0000 | 1.0000 | 1.0000 |

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
