"""Automated test suite for Step 3: Dataset Generation, Splitting, and Baseline Modeling.

Verifies:
1. Correct total dataset size.
2. Correct class balance.
3. Reproducibility (same seed -> identical dataset).
4. Different seeds generate different datasets.
5. Absence of NaN values across all columns.
6. Absence of infinite values across all numeric columns.
7. Correct train / validation / test proportions (70% / 15% / 15%).
8. Stratification maintains class balance in all splits.
9. No event ID overlap between train, validation, and test splits.
10. Target label ('scenario') is strictly excluded from input features.
11. Hidden simulator parameters are excluded from model features.
12. Baseline uses exclusively training NORMAL data.
13. Baseline statistics are bitwise reproducible.
14. Z-score calculation handles zero standard deviation safely.
15. Dataset metadata JSON is generated with all required fields.
"""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.generator import DatasetGenerator
from dataset.schema import (
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMN,
    VALID_SCENARIOS,
    get_feature_matrix,
    get_target,
    get_metadata,
)
from dataset.validation import validate_dataset, DatasetValidationError
from dataset.splitter import split_dataset
from dataset.baseline import NormalBaseline
from dataset.builder import build_processed_dataset


@pytest.fixture(scope="module")
def sample_dataset():
    """Fixture providing a fast, reproducible synthetic dataset for testing."""
    generator = DatasetGenerator(seed=42)
    # 20 samples per class = 60 total rows
    df = generator.generate_dataset(samples_per_class=20, verbose=False)
    return df


# 1. Correct total dataset size
def test_correct_total_dataset_size(sample_dataset):
    expected_rows = 20 * 3
    assert len(sample_dataset) == expected_rows


# 2. Correct class balance
def test_correct_class_balance(sample_dataset):
    counts = sample_dataset[TARGET_COLUMN].value_counts().to_dict()
    for scn in VALID_SCENARIOS:
        assert counts[scn] == 20


# 3. Reproducibility (same seed -> identical dataset)
def test_reproducibility():
    gen1 = DatasetGenerator(seed=999)
    df1 = gen1.generate_dataset(samples_per_class=5, verbose=False)

    gen2 = DatasetGenerator(seed=999)
    df2 = gen2.generate_dataset(samples_per_class=5, verbose=False)

    pd.testing.assert_frame_equal(df1, df2)


# 4. Different seeds generate different datasets
def test_different_seeds_generate_different_datasets():
    gen1 = DatasetGenerator(seed=101)
    df1 = gen1.generate_dataset(samples_per_class=5, verbose=False)

    gen2 = DatasetGenerator(seed=202)
    df2 = gen2.generate_dataset(samples_per_class=5, verbose=False)

    assert not np.allclose(df1["vibration_rms"], df2["vibration_rms"])
    assert not np.allclose(df1["dominant_frequency"], df2["dominant_frequency"])


# 5. No NaN values exist
def test_no_nan_values(sample_dataset):
    assert not sample_dataset.isna().any().any(), "Dataset contains NaN values"


# 6. No infinite values exist
def test_no_infinite_values(sample_dataset):
    num_cols = sample_dataset.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        assert not np.isinf(sample_dataset[col]).any(), f"Column '{col}' contains Inf"


# 7. Correct train / validation / test proportions (70% / 15% / 15%)
def test_split_proportions(sample_dataset):
    train_df, val_df, test_df = split_dataset(sample_dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)
    total = len(sample_dataset)

    assert len(train_df) == int(round(total * 0.70))
    assert len(val_df) == int(round(total * 0.15))
    assert len(test_df) == int(round(total * 0.15))
    assert len(train_df) + len(val_df) + len(test_df) == total


# 8. Stratification maintains class balance across all splits
def test_stratification(sample_dataset):
    train_df, val_df, test_df = split_dataset(sample_dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)

    for scn in VALID_SCENARIOS:
        train_count = (train_df[TARGET_COLUMN] == scn).sum()
        val_count = (val_df[TARGET_COLUMN] == scn).sum()
        test_count = (test_df[TARGET_COLUMN] == scn).sum()

        assert train_count == 14  # 70% of 20
        assert val_count == 3    # 15% of 20
        assert test_count == 3   # 15% of 20


# 9. No event ID overlap between splits
def test_no_event_id_overlap(sample_dataset):
    train_df, val_df, test_df = split_dataset(sample_dataset, random_state=42)

    train_ids = set(train_df["event_id"])
    val_ids = set(val_df["event_id"])
    test_ids = set(test_df["event_id"])

    assert len(train_ids.intersection(val_ids)) == 0, "Train and Validation share event IDs"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test share event IDs"
    assert len(val_ids.intersection(test_ids)) == 0, "Validation and Test share event IDs"


# 10. Target label is not included as an input feature
def test_target_label_excluded_from_features(sample_dataset):
    features_df = get_feature_matrix(sample_dataset)
    assert TARGET_COLUMN not in features_df.columns
    assert "scenario" not in features_df.columns
    assert len(features_df.columns) == len(FEATURE_COLUMNS)


# 11. Hidden simulator parameters are excluded from model features
def test_hidden_parameters_excluded_from_features(sample_dataset):
    features_df = get_feature_matrix(sample_dataset)
    for hidden_col in METADATA_COLUMNS:
        assert hidden_col not in features_df.columns, f"Hidden parameter '{hidden_col}' found in feature matrix!"


# 12. Baseline uses exclusively training NORMAL data
def test_baseline_uses_only_normal_train_data(sample_dataset):
    train_df, _, _ = split_dataset(sample_dataset, random_state=42)
    baseline = NormalBaseline.fit(train_df)

    assert baseline.stats is not None
    # Compute manual normal mean from train_df
    normal_train = train_df[train_df[TARGET_COLUMN] == "NORMAL"]
    for feat in ["vibration_rms", "dominant_frequency", "optical_displacement_max"]:
        manual_mean = float(normal_train[feat].mean())
        baseline_mean = baseline.stats.loc[feat, "mean"]
        assert np.isclose(manual_mean, baseline_mean, rtol=1e-5)

    # Verify that WARNING/CRITICAL were not included in baseline computation
    warning_train_rms = float(train_df[train_df[TARGET_COLUMN] == "WARNING"]["vibration_rms"].mean())
    baseline_rms = baseline.stats.loc["vibration_rms", "mean"]
    assert not np.isclose(warning_train_rms, baseline_rms, rtol=1e-2)


# 13. Baseline statistics are reproducible
def test_baseline_reproducibility(sample_dataset):
    train_df, _, _ = split_dataset(sample_dataset, random_state=42)
    b1 = NormalBaseline.fit(train_df)
    b2 = NormalBaseline.fit(train_df)

    pd.testing.assert_frame_equal(b1.stats, b2.stats)


# 14. Z-score calculation handles zero standard deviation safely
def test_z_score_safe_division(sample_dataset):
    train_df, _, test_df = split_dataset(sample_dataset, random_state=42)
    baseline = NormalBaseline.fit(train_df)

    # Artificially set std of a feature to 0.0 to test zero-division protection
    test_feature = "spectral_peak_count"
    baseline.stats.loc[test_feature, "std"] = 0.0

    z_scores = baseline.compute_z_scores(test_df)
    col_name = f"{test_feature}_zscore"
    assert col_name in z_scores.columns
    assert not np.isnan(z_scores[col_name]).any(), "NaN found in z-score with zero std"
    assert not np.isinf(z_scores[col_name]).any(), "Inf found in z-score with zero std"
    assert np.all(z_scores[col_name] == 0.0), "Expected 0.0 z-score for zero-variance feature"


# 15. Dataset metadata is generated correctly
def test_metadata_generation(tmp_path):
    saved_paths = build_processed_dataset(samples_per_class=10, seed=42, output_dir=tmp_path, verbose=False)
    meta_path = saved_paths["metadata"]
    assert meta_path.exists()

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "dataset_version" in meta
    assert "disclaimer" in meta
    assert meta["samples_per_class"] == 10
    assert meta["total_samples"] == 30
    assert meta["split_counts"]["train"] == 21
    assert meta["split_counts"]["validation"] == 4
    assert meta["split_counts"]["test"] == 5
    assert len(meta["feature_names"]) == len(FEATURE_COLUMNS)
    assert meta["target_column"] == "scenario"
