"""Sklearn-compatible preprocessing pipeline for structural response features.

Features:
  - Enforces schema column ordering and rejects metadata/labels.
  - Safeguards against NaN, Inf, or empty values.
  - Fits scaling (StandardScaler) strictly on training data.
  - Supports feature subsetting for ablation studies.
  - Supports configurable baseline normalization using NormalBaseline.
"""

from typing import List, Literal, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from dataset.baseline import NormalBaseline
from dataset.schema import FEATURE_COLUMNS, TARGET_COLUMN


class StructuralPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessing transformer for engineering features with leak-free scaling and baseline transforms."""

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        mode: Literal["raw", "baseline_normalized", "augmented"] = "raw",
        scale: bool = True,
        normal_baseline: Optional[NormalBaseline] = None,
    ):
        """Initialize preprocessor.

        Args:
            feature_columns: List of feature names to use. Defaults to FEATURE_COLUMNS.
            mode: Feature representation mode:
                - 'raw': standard raw engineering features.
                - 'baseline_normalized': z-scores relative to NORMAL baseline.
                - 'augmented': raw features concatenated with baseline z-scores.
            scale: Whether to apply StandardScaler fitted on the training split.
            normal_baseline: Optional precomputed NormalBaseline instance.
        """
        self.feature_columns = feature_columns if feature_columns is not None else list(FEATURE_COLUMNS)
        self.mode = mode
        self.scale = scale
        self.normal_baseline = normal_baseline
        self.scaler: Optional[StandardScaler] = None
        self.output_feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "StructuralPreprocessor":
        """Fit the preprocessor strictly on training data.

        If baseline normalization is required and normal_baseline was not provided,
        it will be fitted on the NORMAL training events if 'scenario' is present in X or y.

        Args:
            X: Training DataFrame containing feature columns (and optionally metadata/target).
            y: Optional training target series.

        Returns:
            self.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        # Ensure features are present
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise KeyError(f"Missing required feature columns in training DataFrame: {missing}")

        # Setup baseline if needed and not already provided
        if self.mode in ("baseline_normalized", "augmented") and self.normal_baseline is None:
            # Look for scenario target in X or y
            train_df_for_baseline = X.copy()
            if TARGET_COLUMN not in train_df_for_baseline.columns:
                if y is not None:
                    train_df_for_baseline[TARGET_COLUMN] = y.values
                else:
                    raise ValueError(
                        f"Baseline normalization requires '{TARGET_COLUMN}' in training DataFrame or y to isolate NORMAL events."
                    )
            self.normal_baseline = NormalBaseline.fit(train_df_for_baseline)

        # Extract transformed features without scaling to fit scaler
        X_trans = self._extract_features(X)

        # Verify no NaN / Inf
        if np.isneginf(X_trans.values).any() or np.isposinf(X_trans.values).any() or np.isnan(X_trans.values).any():
            raise ValueError("Training features contain NaN or Inf values after transformation.")

        self.output_feature_names_ = list(X_trans.columns)

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(X_trans.values)
        else:
            self.scaler = None

        return self

    def _extract_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Helper to extract and format features according to mode."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise KeyError(f"Missing required feature columns: {missing}")

        df_raw = X[self.feature_columns].copy().astype(float)
        # Clean potential non-finite values safely
        df_raw = df_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if self.mode == "raw":
            return df_raw

        elif self.mode == "baseline_normalized":
            if self.normal_baseline is None:
                raise ValueError("NormalBaseline must be fitted before transforming in 'baseline_normalized' mode.")
            z_df = self.normal_baseline.compute_z_scores(df_raw)
            # Retain only requested feature z-scores
            keep_cols = [f"{c}_zscore" for c in self.feature_columns if f"{c}_zscore" in z_df.columns]
            return z_df[keep_cols]

        elif self.mode == "augmented":
            if self.normal_baseline is None:
                raise ValueError("NormalBaseline must be fitted before transforming in 'augmented' mode.")
            z_df = self.normal_baseline.compute_z_scores(df_raw)
            keep_cols = [f"{c}_zscore" for c in self.feature_columns if f"{c}_zscore" in z_df.columns]
            combined = pd.concat([df_raw, z_df[keep_cols]], axis=1)
            return combined

        else:
            raise ValueError(f"Unknown mode '{self.mode}'. Expected 'raw', 'baseline_normalized', or 'augmented'.")

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform input DataFrame into numerical array ready for model consumption.

        Args:
            X: Input DataFrame.

        Returns:
            2D numpy array of transformed (and optionally scaled) features.
        """
        X_trans = self._extract_features(X)

        if self.scaler is not None:
            scaled_vals = self.scaler.transform(X_trans.values)
            return scaled_vals
        return X_trans.values

    def transform_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input DataFrame and return as a DataFrame with column names."""
        transformed = self.transform(X)
        return pd.DataFrame(transformed, index=X.index, columns=self.output_feature_names_)
