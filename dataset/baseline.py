"""Normal structural baseline modeling module.

Establishes descriptive baseline statistical distributions strictly from NORMAL
training events. Calculates normalized z-score deviations without imposing
arbitrary damage thresholds or safety classifications.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd

from dataset.schema import FEATURE_COLUMNS, TARGET_COLUMN


class NormalBaseline:
    """Computes and applies baseline statistics strictly derived from NORMAL training data."""

    def __init__(self, baseline_stats: Optional[pd.DataFrame] = None):
        """Initialize with optional precomputed baseline statistics DataFrame."""
        self.stats = baseline_stats

    @classmethod
    def fit(cls, train_df: pd.DataFrame) -> "NormalBaseline":
        """Compute descriptive baseline distribution parameters exclusively on NORMAL training events.

        Args:
            train_df: Training set DataFrame containing 'scenario' and FEATURE_COLUMNS.

        Returns:
            NormalBaseline instance with computed statistics.
        """
        if TARGET_COLUMN not in train_df.columns:
            raise KeyError(f"Target column '{TARGET_COLUMN}' required to identify NORMAL baseline events.")

        # STRICT ISOLATION: Filter ONLY NORMAL training events
        normal_events = train_df[train_df[TARGET_COLUMN] == "NORMAL"]
        if normal_events.empty:
            raise ValueError("No NORMAL events found in training dataset to establish baseline.")

        records = []
        for feat in FEATURE_COLUMNS:
            if feat not in normal_events.columns:
                continue
            series = normal_events[feat].dropna().astype(float)
            mean_val = float(series.mean())
            std_val = float(series.std(ddof=1)) if len(series) > 1 else 0.0
            median_val = float(series.median())
            q25_val = float(series.quantile(0.25))
            q75_val = float(series.quantile(0.75))
            iqr_val = float(q75_val - q25_val)
            min_val = float(series.min())
            max_val = float(series.max())

            records.append(
                {
                    "feature": feat,
                    "mean": mean_val,
                    "std": std_val,
                    "median": median_val,
                    "q25": q25_val,
                    "q75": q75_val,
                    "iqr": iqr_val,
                    "min": min_val,
                    "max": max_val,
                }
            )

        baseline_df = pd.DataFrame(records).set_index("feature")
        return cls(baseline_stats=baseline_df)

    def to_csv(self, filepath: Union[str, Path]) -> Path:
        """Save baseline statistics to a CSV file.

        Args:
            filepath: Destination CSV path.

        Returns:
            Resolved Path.
        """
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if self.stats is None:
            raise ValueError("Baseline statistics have not been fitted yet.")
        self.stats.to_csv(out_path, float_format="%.6f")
        return out_path

    @classmethod
    def from_csv(cls, filepath: Union[str, Path]) -> "NormalBaseline":
        """Load baseline statistics from an existing CSV file."""
        baseline_df = pd.read_csv(filepath, index_col="feature")
        return cls(baseline_stats=baseline_df)

    def compute_z_scores(
        self,
        df: pd.DataFrame,
        epsilon: float = 1e-9,
    ) -> pd.DataFrame:
        """Compute normalized z-score deviations from the NORMAL baseline.

        Formula:
            z = (x - mean_baseline) / std_baseline

        Safe handling:
            If std_baseline < epsilon, z-score is safely assigned to 0.0 to
            prevent division by zero or infinite values.

        Args:
            df: Input DataFrame containing features.
            epsilon: Minimum allowable standard deviation threshold.

        Returns:
            DataFrame containing z-score deviations named '{feature}_zscore'.
        """
        if self.stats is None:
            raise ValueError("Baseline statistics must be fitted or loaded before computing z-scores.")

        z_scores = pd.DataFrame(index=df.index)

        for feat in self.stats.index:
            if feat in df.columns:
                mean_val = self.stats.loc[feat, "mean"]
                std_val = self.stats.loc[feat, "std"]

                if std_val > epsilon:
                    z = (df[feat] - mean_val) / std_val
                else:
                    z = pd.Series(0.0, index=df.index)

                # Ensure no NaN/Inf
                z = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
                z_scores[f"{feat}_zscore"] = z

        return z_scores
