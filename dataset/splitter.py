"""Dataset splitting and leakage prevention module.

Performs stratified train / validation / test partitioning (70% / 15% / 15%)
ensuring zero event overlap between splits and strict isolation of evaluation sets.
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from dataset.schema import TARGET_COLUMN


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform leak-free stratified train / validation / test partition.

    Args:
        df: Input DataFrame containing features, metadata, and target scenario.
        train_ratio: Proportion of data allocated to train (default 0.70).
        val_ratio: Proportion of data allocated to validation (default 0.15).
        test_ratio: Proportion of data allocated to test (default 0.15).
        random_state: Seed for deterministic stratified partitioning.

    Returns:
        Tuple of (train_df, val_df, test_df) as independent DataFrames.

    Raises:
        ValueError: If split proportions do not sum to 1.0 or if data leakage is detected.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.999 <= total_ratio <= 1.001):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio} + {val_ratio} + {test_ratio} = {total_ratio}"
        )

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' required for stratified splitting.")

    # 1. First split: separate train (70%) from holdout validation+test (30%)
    holdout_size = val_ratio + test_ratio
    train_df, holdout_df = train_test_split(
        df,
        test_size=holdout_size,
        stratify=df[TARGET_COLUMN],
        random_state=random_state,
    )

    # 2. Second split: partition holdout into validation (15%) and test (15%)
    val_relative_fraction = val_ratio / holdout_size
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=(1.0 - val_relative_fraction),
        stratify=holdout_df[TARGET_COLUMN],
        random_state=random_state,
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # 3. Leakage Verification: Check zero event_id overlap
    if "event_id" in df.columns:
        train_ids = set(train_df["event_id"])
        val_ids = set(val_df["event_id"])
        test_ids = set(test_df["event_id"])

        train_val_overlap = train_ids.intersection(val_ids)
        train_test_overlap = train_ids.intersection(test_ids)
        val_test_overlap = val_ids.intersection(test_ids)

        if train_val_overlap or train_test_overlap or val_test_overlap:
            raise ValueError(
                f"Data leakage detected! Event IDs overlap between splits: "
                f"train/val: {len(train_val_overlap)}, "
                f"train/test: {len(train_test_overlap)}, "
                f"val/test: {len(val_test_overlap)}"
            )

    return train_df, val_df, test_df
