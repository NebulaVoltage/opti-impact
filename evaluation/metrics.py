"""Comprehensive evaluation metrics for structural state classification.

Calculates multi-class classification metrics with strict class ordering
(['NORMAL', 'WARNING', 'CRITICAL']) and isolated focus on the CRITICAL class.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

CLASS_ORDER: List[str] = ["NORMAL", "WARNING", "CRITICAL"]


def compute_classification_metrics(
    y_true: Any,
    y_pred: Any,
    class_order: List[str] = CLASS_ORDER,
) -> Dict[str, float]:
    """Compute overall, macro, weighted, and per-class classification metrics.

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted target labels.
        class_order: Canonical list of class labels in deterministic order.

    Returns:
        Dictionary of computed metric names and floating point values.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_prec = float(precision_score(y_true_arr, y_pred_arr, labels=class_order, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true_arr, y_pred_arr, labels=class_order, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, labels=class_order, average="macro", zero_division=0))

    weighted_prec = float(precision_score(y_true_arr, y_pred_arr, labels=class_order, average="weighted", zero_division=0))
    weighted_rec = float(recall_score(y_true_arr, y_pred_arr, labels=class_order, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, labels=class_order, average="weighted", zero_division=0))

    # Per-class metrics
    precisions = precision_score(y_true_arr, y_pred_arr, labels=class_order, average=None, zero_division=0)
    recalls = recall_score(y_true_arr, y_pred_arr, labels=class_order, average=None, zero_division=0)
    f1s = f1_score(y_true_arr, y_pred_arr, labels=class_order, average=None, zero_division=0)

    per_class: Dict[str, float] = {}
    for idx, cls_name in enumerate(class_order):
        c_lower = cls_name.lower()
        per_class[f"{c_lower}_precision"] = float(precisions[idx])
        per_class[f"{c_lower}_recall"] = float(recalls[idx])
        per_class[f"{c_lower}_f1"] = float(f1s[idx])

    metrics: Dict[str, float] = {
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_prec,
        "weighted_recall": weighted_rec,
        "weighted_f1": weighted_f1,
        **per_class,
        # Secondary optimization target alias
        "critical_recall": per_class.get("critical_recall", 0.0),
        "critical_f1": per_class.get("critical_f1", 0.0),
        "critical_precision": per_class.get("critical_precision", 0.0),
    }

    return metrics


def compute_confusion_matrices(
    y_true: Any,
    y_pred: Any,
    class_order: List[str] = CLASS_ORDER,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute raw counts and row-normalized (true class) confusion matrices.

    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.
        class_order: Ordered list of labels.

    Returns:
        Tuple of (raw_cm_df, normalized_cm_df).
    """
    raw_cm = confusion_matrix(y_true, y_pred, labels=class_order)
    raw_df = pd.DataFrame(raw_cm, index=class_order, columns=class_order)

    # Row-normalized: each row sums to 1.0 (or 0 if row sum is 0)
    row_sums = raw_cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(
        raw_cm.astype(float),
        row_sums,
        out=np.zeros_like(raw_cm, dtype=float),
        where=(row_sums > 0),
    )
    norm_df = pd.DataFrame(norm_cm, index=class_order, columns=class_order)

    return raw_df, norm_df
