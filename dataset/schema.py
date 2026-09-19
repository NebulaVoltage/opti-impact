"""Dataset schema definitions and feature-metadata separation utilities.

Ensures clean boundaries between measurable input features, metadata,
and target labels to prevent target and parameter leakage during model training.
"""

from typing import List, Tuple
import pandas as pd

# Measurable physical engineering features (inputs for downstream ML models)
FEATURE_COLUMNS: List[str] = [
    # 1. Vibration Time-Domain Features
    "vibration_mean",
    "vibration_std",
    "vibration_rms",
    "vibration_variance",
    "vibration_peak",
    "vibration_peak_to_peak",
    "vibration_crest_factor",
    "vibration_kurtosis",
    "vibration_skewness",
    "vibration_zero_crossing_rate",
    "vibration_energy",
    # 2. Vibration Frequency-Domain Features
    "dominant_frequency",
    "dominant_frequency_amplitude",
    "spectral_energy",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_peak_count",
    # 3. Optical Displacement & Kinematic Features
    "optical_displacement_max",
    "optical_displacement_rms",
    "optical_displacement_mean",
    "optical_displacement_std",
    "optical_velocity_max",
    "optical_velocity_rms",
    "optical_acceleration_max",
    "optical_acceleration_rms",
    "optical_dominant_frequency",
    "optical_frequency_amplitude",
    "optical_rotation_max",
    "optical_rotation_rms",
    # 4. Multimodal Cross-Sensor Correlation Features
    "vibration_optical_correlation",
    "cross_correlation_max",
    "cross_correlation_lag_seconds",
]

# Hidden simulator parameters and tracking identifiers (for debugging/audit, NEVER model inputs)
METADATA_COLUMNS: List[str] = [
    "event_id",
    "random_seed",
    "sim_natural_frequency",
    "sim_amplitude",
    "sim_damping_ratio",
    "sim_impact_strength",
    "sim_noise_level",
    "sim_optical_noise",
    "sim_sampling_rate",
    "sim_duration",
]

# Classification target label
TARGET_COLUMN: str = "scenario"

# Valid class labels
VALID_SCENARIOS: List[str] = ["NORMAL", "WARNING", "CRITICAL"]


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract strictly measurable engineering feature columns from a dataset.

    Guarantees that target labels and hidden simulator parameters are excluded.

    Args:
        df: Input DataFrame containing dataset records.

    Returns:
        DataFrame containing only FEATURE_COLUMNS.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
    return df[FEATURE_COLUMNS].copy()


def get_target(df: pd.DataFrame) -> pd.Series:
    """Extract the scenario target label Series.

    Args:
        df: Input DataFrame containing dataset records.

    Returns:
        Series of scenario target strings.
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in DataFrame.")
    return df[TARGET_COLUMN].copy()


def get_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Extract metadata and tracking columns from a dataset.

    Args:
        df: Input DataFrame containing dataset records.

    Returns:
        DataFrame containing available METADATA_COLUMNS and TARGET_COLUMN.
    """
    cols = [c for c in METADATA_COLUMNS if c in df.columns]
    if TARGET_COLUMN in df.columns and TARGET_COLUMN not in cols:
        cols.append(TARGET_COLUMN)
    return df[cols].copy()


def split_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataset into feature matrix X and target vector y.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (X, y).
    """
    return get_feature_matrix(df), get_target(df)
