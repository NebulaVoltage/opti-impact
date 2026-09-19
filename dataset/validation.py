"""Dataset validation and integrity checking module.

Verifies completeness, absence of invalid numerical values (NaN/Inf),
class balance, schema conformance, and physical feature range plausibility.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from dataset.schema import FEATURE_COLUMNS, METADATA_COLUMNS, TARGET_COLUMN, VALID_SCENARIOS


class DatasetValidationError(ValueError):
    """Raised when a generated dataset fails integrity or sanity checks."""

    pass


def validate_dataset(
    df: pd.DataFrame,
    expected_samples_per_class: Optional[int] = None,
    expected_scenarios: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Validate dataset structure, absence of invalid values, and physical plausibility.

    Args:
        df: Input pandas DataFrame to validate.
        expected_samples_per_class: Expected count of samples per scenario (if known).
        expected_scenarios: List of expected scenario labels.

    Returns:
        Dictionary containing validation report summary.

    Raises:
        DatasetValidationError: If any critical validation check fails.
    """
    scenarios = expected_scenarios or VALID_SCENARIOS
    errors: List[str] = []

    # 1. Row count and empty check
    if df.empty or len(df) == 0:
        raise DatasetValidationError("Dataset is empty.")

    # 2. Required columns check
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        errors.append(f"Missing required feature columns: {missing_features}")

    missing_meta = [c for c in METADATA_COLUMNS if c not in df.columns]
    if missing_meta:
        errors.append(f"Missing required metadata columns: {missing_meta}")

    if TARGET_COLUMN not in df.columns:
        errors.append(f"Missing required target column '{TARGET_COLUMN}'.")

    if errors:
        raise DatasetValidationError("; ".join(errors))

    # 3. Target values and class balance check
    unique_scenarios = sorted(df[TARGET_COLUMN].unique())
    class_counts = df[TARGET_COLUMN].value_counts().to_dict()

    for scn in scenarios:
        if scn not in class_counts:
            errors.append(f"Scenario '{scn}' has 0 samples.")
        elif expected_samples_per_class is not None:
            actual = class_counts[scn]
            if actual != expected_samples_per_class:
                errors.append(
                    f"Class '{scn}' count mismatch: expected {expected_samples_per_class}, got {actual}."
                )

    # 4. NaN / Null checks
    nan_counts = df.isna().sum()
    columns_with_nan = nan_counts[nan_counts > 0]
    if not columns_with_nan.empty:
        errors.append(f"Columns contain NaN values: {columns_with_nan.to_dict()}")

    # 5. Infinity checks across numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_cols = {}
    for col in numeric_cols:
        inf_count = int(np.isinf(df[col]).sum())
        if inf_count > 0:
            inf_cols[col] = inf_count

    if inf_cols:
        errors.append(f"Columns contain infinite values: {inf_cols}")

    # 6. Physical feature range plausibility checks
    if "vibration_rms" in df.columns:
        invalid_rms = int((df["vibration_rms"] <= 0).sum())
        if invalid_rms > 0:
            errors.append(f"{invalid_rms} samples have vibration_rms <= 0.")

    if "dominant_frequency" in df.columns:
        invalid_freq = int(((df["dominant_frequency"] <= 0) | (df["dominant_frequency"] > 50)).sum())
        if invalid_freq > 0:
            errors.append(f"{invalid_freq} samples have invalid dominant_frequency (outside 0-50 Hz).")

    if "optical_displacement_max" in df.columns:
        invalid_disp = int((df["optical_displacement_max"] <= 0).sum())
        if invalid_disp > 0:
            errors.append(f"{invalid_disp} samples have optical_displacement_max <= 0.")

    if "vibration_optical_correlation" in df.columns:
        invalid_corr = int(((df["vibration_optical_correlation"] < -1.0) | (df["vibration_optical_correlation"] > 1.0)).sum())
        if invalid_corr > 0:
            errors.append(f"{invalid_corr} samples have vibration_optical_correlation outside [-1, 1].")

    # 7. Uniqueness of event_id
    if "event_id" in df.columns:
        dup_ids = int(df["event_id"].duplicated().sum())
        if dup_ids > 0:
            errors.append(f"Found {dup_ids} duplicated event_id values.")

    if errors:
        raise DatasetValidationError("Dataset validation failed:\n - " + "\n - ".join(errors))

    return {
        "status": "VALID",
        "total_samples": len(df),
        "class_counts": class_counts,
        "feature_count": len(FEATURE_COLUMNS),
        "metadata_count": len(METADATA_COLUMNS),
        "nan_count": 0,
        "inf_count": 0,
    }
