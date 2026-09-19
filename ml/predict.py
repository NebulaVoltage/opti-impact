"""Inference prediction module for structural response classification.

Accepts single feature dictionaries, pandas Series, or DataFrames and produces
predicted structural state classifications along with calibrated class probabilities.

NOTE: Output probabilities are model-estimated multi-class posterior probabilities
on this synthetic classification task, NOT structural failure or collapse probabilities.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from evaluation.metrics import CLASS_ORDER
from ml.model_io import load_model_bundle

_CACHED_BUNDLE: Optional[tuple] = None


def get_inference_bundle(
    models_dir: Path = Path("models/step4"),
    prefix: str = "step4_v1",
) -> tuple:
    """Retrieve or load cached model, preprocessor, and metadata bundle."""
    global _CACHED_BUNDLE
    if _CACHED_BUNDLE is None:
        _CACHED_BUNDLE = load_model_bundle(models_dir, prefix=prefix)
    return _CACHED_BUNDLE


def predict(
    features: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    models_dir: Path = Path("models/step4"),
    prefix: str = "step4_v1",
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Perform structural condition prediction on input feature(s).

    Args:
        features: Single feature dictionary, Series, or multi-row DataFrame.
        models_dir: Artifacts directory containing serialized model.
        prefix: Version prefix.

    Returns:
        For a single sample:
            {
                "predicted_class": "NORMAL" | "WARNING" | "CRITICAL",
                "probabilities": {
                    "NORMAL": float,
                    "WARNING": float,
                    "CRITICAL": float
                }
            }
        For a DataFrame: list of such dictionaries.
    """
    model, preprocessor, _ = get_inference_bundle(models_dir=models_dir, prefix=prefix)

    is_single = False
    if isinstance(features, dict):
        df = pd.DataFrame([features])
        is_single = True
    elif isinstance(features, pd.Series):
        df = pd.DataFrame([features.to_dict()])
        is_single = True
    elif isinstance(features, pd.DataFrame):
        df = features.copy()
    else:
        raise TypeError(f"Unsupported features input type: {type(features)}")

    # Preprocess using fitted preprocessor
    X_proc = preprocessor.transform(df)

    # Predict class
    preds = model.predict(X_proc)

    # Predict probabilities if supported
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_proc)
        classes = list(model.classes_)
    elif hasattr(model, "decision_function"):
        # Softmax over decision function as fallback
        dfunc = model.decision_function(X_proc)
        exp_df = np.exp(dfunc - np.max(dfunc, axis=1, keepdims=True))
        probs = exp_df / np.sum(exp_df, axis=1, keepdims=True)
        classes = list(model.classes_)
    else:
        classes = CLASS_ORDER

    results = []
    for idx, pred_cls in enumerate(preds):
        prob_dict = {}
        if probs is not None:
            sample_probs = probs[idx]
            for c_idx, c_name in enumerate(classes):
                prob_dict[str(c_name)] = float(sample_probs[c_idx])
            # Ensure all canonical classes exist in dictionary
            for c_name in CLASS_ORDER:
                if c_name not in prob_dict:
                    prob_dict[c_name] = 0.0
        else:
            prob_dict = {c: (1.0 if c == pred_cls else 0.0) for c in CLASS_ORDER}

        results.append(
            {
                "predicted_class": str(pred_cls),
                "probabilities": prob_dict,
            }
        )

    if is_single:
        return results[0]
    return results
