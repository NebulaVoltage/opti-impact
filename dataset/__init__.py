"""Machine learning dataset generation, validation, splitting, and baseline package.

Provides modular tools for parameter-varying simulation synthesis,
leak-free stratified splitting, and NORMAL structural baseline computation.
"""

from dataset.schema import (
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMN,
    VALID_SCENARIOS,
    get_feature_matrix,
    get_target,
    get_metadata,
    split_features_and_target,
)
from dataset.variation import (
    ParameterRange,
    ScenarioVariationConfig,
    DEFAULT_VARIATION_PRESETS,
    get_variation_config,
)
from dataset.generator import DatasetGenerator
from dataset.validation import (
    DatasetValidationError,
    validate_dataset,
)
from dataset.splitter import split_dataset
from dataset.baseline import NormalBaseline
from dataset.builder import build_processed_dataset

__all__ = [
    "FEATURE_COLUMNS",
    "METADATA_COLUMNS",
    "TARGET_COLUMN",
    "VALID_SCENARIOS",
    "get_feature_matrix",
    "get_target",
    "get_metadata",
    "split_features_and_target",
    "ParameterRange",
    "ScenarioVariationConfig",
    "DEFAULT_VARIATION_PRESETS",
    "get_variation_config",
    "DatasetGenerator",
    "DatasetValidationError",
    "validate_dataset",
    "split_dataset",
    "NormalBaseline",
    "build_processed_dataset",
]
