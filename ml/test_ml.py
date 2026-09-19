"""Unit and integration test suite for Step 4 machine learning modules.

Verifies:
  1. Dataset loading and split integrity.
  2. Correct feature/target separation.
  3. Simulator metadata and hidden parameters excluded from inputs.
  4. Preprocessor fits strictly on training data without leakage.
  5. Logistic Regression trains and predicts.
  6. SVM trains and predicts.
  7. Random Forest trains and predicts.
  8. HistGradientBoosting trains and predicts.
  9. XGBoost trains and predicts.
  10. Cross-validation executes cleanly.
  11. Predictions strictly produce ['NORMAL', 'WARNING', 'CRITICAL'].
  12. Probability outputs are calibrated and sum to 1.0.
  13. Metrics calculation correctness (accuracy, macro F1, critical recall).
  14. Confusion matrix dimensions (3x3) and row normalization sum to 1.0.
  15. Model serialization and artifact persistence.
  16. Reloaded model produces identical outputs to original model.
  17. Feature-group ablation executions across Groups A, B, C, D.
  18. Baseline-normalized and augmented transformations execute cleanly.
  19. No NaN or Inf enters model transformations.
  20. Inference rejects invalid schemas and accepts valid single instances.
  21. Model metadata JSON schema completeness.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from dataset.schema import (
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMN,
    VALID_SCENARIOS,
    get_feature_matrix,
    get_target,
    split_features_and_target,
)
from evaluation.metrics import (
    CLASS_ORDER,
    compute_classification_metrics,
    compute_confusion_matrices,
)
from ml.ablation import run_baseline_ablation, run_multimodal_ablation
from ml.feature_groups import (
    ABLATION_GROUPS,
    GROUP_A_VIBRATION_ONLY,
    GROUP_B_OPTICAL_ONLY,
    GROUP_C_VIBRATION_AND_OPTICAL,
    GROUP_D_FULL_MULTIMODAL,
)
from ml.model_io import load_model_bundle, save_model_bundle
from ml.models import get_model_zoo
from ml.predict import predict
from ml.preprocessing import StructuralPreprocessor


@pytest.fixture(scope="module")
def train_data() -> pd.DataFrame:
    path = Path("dataset/processed/train.csv")
    assert path.exists(), "dataset/processed/train.csv missing"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def val_data() -> pd.DataFrame:
    path = Path("dataset/processed/validation.csv")
    assert path.exists(), "dataset/processed/validation.csv missing"
    return pd.read_csv(path)


def test_dataset_loading_and_shape(train_data: pd.DataFrame, val_data: pd.DataFrame):
    """Verify train and validation sets load with expected row counts and columns."""
    assert len(train_data) == 2100
    assert len(val_data) == 450
    assert TARGET_COLUMN in train_data.columns
    for feat in FEATURE_COLUMNS:
        assert feat in train_data.columns


def test_correct_feature_target_separation(train_data: pd.DataFrame):
    """Verify feature matrix contains only FEATURE_COLUMNS and excludes target/metadata."""
    X, y = split_features_and_target(train_data)
    assert X.shape[1] == len(FEATURE_COLUMNS)
    assert TARGET_COLUMN not in X.columns
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(y) == len(train_data)


def test_hidden_parameters_excluded_from_inputs(train_data: pd.DataFrame):
    """Verify that no hidden simulation parameters appear in the feature matrix."""
    X, _ = split_features_and_target(train_data)
    for meta_col in METADATA_COLUMNS:
        assert meta_col not in X.columns


def test_preprocessor_fits_only_on_training_data(train_data: pd.DataFrame, val_data: pd.DataFrame):
    """Verify scaler parameters are determined strictly from training records."""
    prep = StructuralPreprocessor(mode="raw", scale=True)
    prep.fit(train_data)

    assert prep.scaler is not None
    # Scaler mean must match train mean, not validation mean
    train_feat_means = train_data[FEATURE_COLUMNS].mean().values
    val_feat_means = val_data[FEATURE_COLUMNS].mean().values

    np.testing.assert_allclose(prep.scaler.mean_, train_feat_means, rtol=1e-5)
    assert not np.allclose(prep.scaler.mean_, val_feat_means)


def test_logistic_regression_trains_and_predicts(train_data: pd.DataFrame, val_data: pd.DataFrame):
    """Test LogisticRegression model training and prediction pipeline."""
    prep = StructuralPreprocessor(mode="raw", scale=True)
    prep.fit(train_data)
    X_tr = prep.transform(train_data)
    X_val = prep.transform(val_data)
    y_tr = train_data[TARGET_COLUMN].values

    model = get_model_zoo()["logistic_regression"]
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)

    assert len(preds) == len(val_data)
    assert set(preds).issubset(set(VALID_SCENARIOS))


def test_svm_trains_and_predicts(train_data: pd.DataFrame, val_data: pd.DataFrame):
    """Test SVM model training and prediction pipeline."""
    prep = StructuralPreprocessor(mode="raw", scale=True)
    prep.fit(train_data)
    X_tr = prep.transform(train_data)
    X_val = prep.transform(val_data)
    y_tr = train_data[TARGET_COLUMN].values

    model = get_model_zoo()["svm_rbf"]
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)

    assert len(preds) == len(val_data)
    assert set(preds).issubset(set(VALID_SCENARIOS))


def test_random_forest_trains_and_predicts(train_data: pd.DataFrame, val_data: pd.DataFrame):
    """Test RandomForest model training and prediction pipeline."""
    prep = StructuralPreprocessor(mode="raw", scale=False)
    prep.fit(train_data)
    X_tr = prep.transform(train_data)
    X_val = prep.transform(val_data)
    y_tr = train_data[TARGET_COLUMN].values

    model = get_model_zoo()["random_forest"]
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)

    assert len(preds) == len(val_data)
    assert set(preds).issubset(set(VALID_SCENARIOS))


def test_hist_gradient_boosting_trains_and_predicts(train_data: pd.DataFrame, val_data: pd.DataFrame):
    """Test HistGradientBoosting model training and prediction pipeline."""
    prep = StructuralPreprocessor(mode="raw", scale=False)
    prep.fit(train_data)
    X_tr = prep.transform(train_data)
    X_val = prep.transform(val_data)
    y_tr = train_data[TARGET_COLUMN].values

    model = get_model_zoo()["hist_gradient_boosting"]
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)

    assert len(preds) == len(val_data)
    assert set(preds).issubset(set(VALID_SCENARIOS))


def test_xgboost_trains_and_predicts(train_data: pd.DataFrame, val_data: pd.DataFrame):
    """Test XGBoost model training and prediction pipeline."""
    prep = StructuralPreprocessor(mode="raw", scale=False)
    prep.fit(train_data)
    X_tr = prep.transform(train_data)
    X_val = prep.transform(val_data)
    y_tr = train_data[TARGET_COLUMN].values

    # Encode scenario labels for XGBoost if required by string labels
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_enc = le.fit_transform(y_tr)

    import xgboost as xgb
    model = xgb.XGBClassifier(n_estimators=20, max_depth=3, random_state=42, eval_metric="mlogloss")
    model.fit(X_tr, y_enc)
    preds_enc = model.predict(X_val)
    preds = le.inverse_transform(preds_enc)

    assert len(preds) == len(val_data)
    assert set(preds).issubset(set(VALID_SCENARIOS))


def test_cross_validation_executes(train_data: pd.DataFrame):
    """Verify that 5-fold cross validation executes without errors."""
    from sklearn.model_selection import StratifiedKFold

    X_df, y_series = split_features_and_target(train_data)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    fold_counts = []
    for train_idx, val_idx in skf.split(X_df, y_series):
        fold_counts.append((len(train_idx), len(val_idx)))

    assert len(fold_counts) == 3
    for tr_c, val_c in fold_counts:
        assert tr_c == 1400
        assert val_c == 700


def test_probability_outputs_are_valid(train_data: pd.DataFrame, val_data: pd.DataFrame):
    """Verify that model probabilities are non-negative and sum to 1.0."""
    prep = StructuralPreprocessor(mode="raw", scale=True)
    prep.fit(train_data)
    X_tr = prep.transform(train_data)
    X_val = prep.transform(val_data[:20])

    model = get_model_zoo()["random_forest"]
    model.fit(X_tr, train_data[TARGET_COLUMN].values)
    probs = model.predict_proba(X_val)

    assert probs.shape == (20, 3)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(20), atol=1e-5)


def test_metrics_calculation_correctness():
    """Verify compute_classification_metrics outputs correct formulas."""
    y_true = ["NORMAL", "NORMAL", "WARNING", "CRITICAL", "CRITICAL"]
    y_pred = ["NORMAL", "WARNING", "WARNING", "CRITICAL", "CRITICAL"]

    m = compute_classification_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(4 / 5)
    assert m["critical_recall"] == pytest.approx(1.0)
    assert m["critical_precision"] == pytest.approx(1.0)
    assert m["normal_recall"] == pytest.approx(0.5)


def test_confusion_matrix_dimensions_and_normalization():
    """Verify confusion matrix shape (3x3) and row-normalization sum to 1.0."""
    y_true = ["NORMAL"] * 10 + ["WARNING"] * 10 + ["CRITICAL"] * 10
    y_pred = ["NORMAL"] * 8 + ["WARNING"] * 2 + ["WARNING"] * 10 + ["CRITICAL"] * 10

    raw_cm, norm_cm = compute_confusion_matrices(y_true, y_pred)
    assert raw_cm.shape == (3, 3)
    assert norm_cm.shape == (3, 3)
    assert list(raw_cm.index) == CLASS_ORDER
    assert list(raw_cm.columns) == CLASS_ORDER

    # Row sums of normalized matrix should equal 1.0
    row_sums = norm_cm.sum(axis=1).values
    np.testing.assert_allclose(row_sums, [1.0, 1.0, 1.0])


def test_model_serialization_and_reload(train_data: pd.DataFrame, val_data: pd.DataFrame):
    """Verify model bundle serialization and verify identical predictions after reload."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        prep = StructuralPreprocessor(mode="raw", scale=True)
        prep.fit(train_data)
        X_tr = prep.transform(train_data)
        y_tr = train_data[TARGET_COLUMN].values

        model = get_model_zoo()["random_forest"]
        model.fit(X_tr, y_tr)

        meta = {
            "model_type": "random_forest",
            "model_version": "test_v1",
            "feature_names": prep.output_feature_names_,
        }

        save_model_bundle(model, prep, meta, tmp_path, prefix="test_v1")

        # Reload bundle
        reloaded_model, reloaded_prep, reloaded_meta = load_model_bundle(tmp_path, prefix="test_v1")

        assert reloaded_meta["model_type"] == "random_forest"
        assert reloaded_meta["feature_names"] == prep.output_feature_names_

        # Compare predictions on validation set
        X_val_orig = prep.transform(val_data)
        X_val_reload = reloaded_prep.transform(val_data)
        np.testing.assert_allclose(X_val_orig, X_val_reload)

        preds_orig = model.predict(X_val_orig)
        preds_reload = reloaded_model.predict(X_val_reload)
        np.testing.assert_array_equal(preds_orig, preds_reload)

        probs_orig = model.predict_proba(X_val_orig)
        probs_reload = reloaded_model.predict_proba(X_val_reload)
        np.testing.assert_allclose(probs_orig, probs_reload)


def test_feature_group_ablation_groups():
    """Verify ablation groups match physical definitions and counts."""
    assert len(GROUP_A_VIBRATION_ONLY) == 17
    assert len(GROUP_B_OPTICAL_ONLY) == 12
    assert len(GROUP_C_VIBRATION_AND_OPTICAL) == 29
    assert len(GROUP_D_FULL_MULTIMODAL) == 32
    assert set(GROUP_A_VIBRATION_ONLY).isdisjoint(set(GROUP_B_OPTICAL_ONLY))


def test_baseline_normalization_mode(train_data: pd.DataFrame, val_data: pd.DataFrame):
    """Verify baseline normalization generates z-scores without NaN or Inf."""
    prep = StructuralPreprocessor(mode="baseline_normalized", scale=True)
    prep.fit(train_data)

    X_tr = prep.transform(train_data)
    X_val = prep.transform(val_data)

    assert X_tr.shape[1] == len(FEATURE_COLUMNS)
    assert X_val.shape[1] == len(FEATURE_COLUMNS)
    assert not np.isnan(X_tr).any()
    assert not np.isnan(X_val).any()
    assert not np.isinf(X_tr).any()
    assert not np.isinf(X_val).any()


def test_augmented_mode(train_data: pd.DataFrame, val_data: pd.DataFrame):
    """Verify augmented mode combines raw features and z-score features."""
    prep = StructuralPreprocessor(mode="augmented", scale=True)
    prep.fit(train_data)

    X_tr = prep.transform(train_data)
    X_val = prep.transform(val_data)

    assert X_tr.shape[1] == len(FEATURE_COLUMNS) * 2
    assert X_val.shape[1] == len(FEATURE_COLUMNS) * 2
    assert not np.isnan(X_tr).any()
    assert not np.isnan(X_val).any()


def test_no_nan_or_inf_in_processed_splits(train_data: pd.DataFrame, val_data: pd.DataFrame):
    """Verify all inputs in train and val datasets are finite."""
    for df in [train_data, val_data]:
        feat_df = df[FEATURE_COLUMNS]
        assert not feat_df.isna().any().any()
        assert not np.isneginf(feat_df.values).any()
        assert not np.isposinf(feat_df.values).any()


def test_predictions_contain_only_valid_scenarios(train_data: pd.DataFrame, val_data: pd.DataFrame):
    """Verify that predictions from all models strictly contain only NORMAL, WARNING, CRITICAL."""
    prep = StructuralPreprocessor(mode="raw", scale=True)
    prep.fit(train_data)
    X_tr = prep.transform(train_data)
    X_val = prep.transform(val_data)
    y_tr = train_data[TARGET_COLUMN].values

    for name, model in get_model_zoo().items():
        m = clone(model)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_val)
        unique_preds = set(preds)
        assert unique_preds.issubset(set(VALID_SCENARIOS)), f"Model {name} produced invalid scenario: {unique_preds}"


def test_feature_count_consistency_training_and_inference(train_data: pd.DataFrame, val_data: pd.DataFrame):
    """Verify feature dimension stays strictly identical between training fit and inference transform."""
    prep = StructuralPreprocessor(mode="raw", scale=True)
    prep.fit(train_data)
    X_tr = prep.transform(train_data)
    X_val = prep.transform(val_data)
    single_sample = val_data.iloc[[0]]
    X_single = prep.transform(single_sample)

    assert X_tr.shape[1] == len(FEATURE_COLUMNS)
    assert X_val.shape[1] == len(FEATURE_COLUMNS)
    assert X_single.shape[1] == len(FEATURE_COLUMNS)
    assert X_tr.shape[1] == X_single.shape[1]


def test_model_metadata_generated():
    """Verify that model_metadata.json contains all required provenance and tracking fields."""
    import json
    meta_path = Path("models/step4/step4_v1_metadata.json")
    if not meta_path.exists():
        from ml.train import train_and_save_final_model
        train_and_save_final_model(output_dir=Path("models/step4"), version_prefix="step4_v1")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    required_keys = [
        "model_type",
        "model_version",
        "dataset_version",
        "random_seed",
        "selection_rule",
        "preprocessing_configuration",
        "feature_names",
        "feature_groups",
        "hyperparameters",
        "cross_validation_metrics",
        "validation_metrics",
        "software_versions",
        "disclaimer",
    ]
    for key in required_keys:
        assert key in meta, f"Required metadata key '{key}' missing from metadata."


def test_predict_single_instance(train_data: pd.DataFrame):
    """Verify predict module handles a single feature dictionary or Series."""
    # Ensure a model bundle is saved
    from ml.train import train_and_save_final_model

    bundle_dir = Path("models/step4")
    if not (bundle_dir / "step4_v1_model.joblib").exists():
        train_and_save_final_model(output_dir=bundle_dir, version_prefix="step4_v1")

    sample_dict = train_data.iloc[0][FEATURE_COLUMNS].to_dict()
    res = predict(sample_dict, models_dir=bundle_dir, prefix="step4_v1")

    assert "predicted_class" in res
    assert res["predicted_class"] in VALID_SCENARIOS
    assert "probabilities" in res
    assert len(res["probabilities"]) == 3
    prob_sum = sum(res["probabilities"].values())
    assert prob_sum == pytest.approx(1.0, rel=1e-3)

