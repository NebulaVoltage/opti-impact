"""Model and artifact input/output serialization routines.

Handles saving and loading trained models, preprocessing transformers,
and rich provenance metadata in joblib and JSON formats.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost as xgb


def get_software_versions() -> Dict[str, str]:
    """Retrieve runtime software and library version stamps."""
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "xgboost": xgb.__version__,
    }


def save_model_bundle(
    model: Any,
    preprocessor: Any,
    metadata: Dict[str, Any],
    output_dir: Path,
    prefix: str = "step4_v1",
) -> Tuple[Path, Path, Path]:
    """Serialize model, preprocessor, and metadata to disk.

    Args:
        model: Trained scikit-learn or XGBoost estimator.
        preprocessor: Fitted StructuralPreprocessor instance.
        metadata: Comprehensive metadata dictionary.
        output_dir: Target directory path.
        prefix: Filename prefix (e.g. 'step4_v1').

    Returns:
        Tuple of (model_path, preprocessor_path, metadata_path).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{prefix}_model.joblib"
    preprocessor_path = out_dir / f"{prefix}_preprocessor.joblib"
    metadata_path = out_dir / f"{prefix}_metadata.json"

    # Save model and preprocessor
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    # Ensure software versions and timestamp are included
    if "software_versions" not in metadata:
        metadata["software_versions"] = get_software_versions()
    if "training_timestamp" not in metadata:
        metadata["training_timestamp"] = datetime.now(timezone.utc).isoformat()

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return model_path, preprocessor_path, metadata_path


def load_model_bundle(
    artifact_dir: Path,
    prefix: str = "step4_v1",
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load model, preprocessor, and metadata from an artifact directory.

    Args:
        artifact_dir: Directory containing serialized artifacts.
        prefix: Filename prefix.

    Returns:
        Tuple of (model, preprocessor, metadata).
    """
    dir_path = Path(artifact_dir)
    model_path = dir_path / f"{prefix}_model.joblib"
    preprocessor_path = dir_path / f"{prefix}_preprocessor.joblib"
    metadata_path = dir_path / f"{prefix}_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, preprocessor, metadata
