"""Model definitions and factories for structural response classification.

Provides standardized estimators for four distinct model families:
  1. Logistic Regression (Linear baseline with l2 regularization)
  2. Support Vector Machine (RBF kernel with calibrated probabilities)
  3. Random Forest (Non-linear ensemble with balanced weights)
  4. HistGradientBoosting (Histogram-based gradient boosted trees)
  5. XGBoost (Extreme gradient boosted decision trees)
"""

from typing import Any, Dict, List, Optional
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb

RANDOM_STATE: int = 42


def get_model_zoo(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """Get the dictionary of default candidate model instances for benchmarking.

    Args:
        random_state: Seed for reproducible initialization.

    Returns:
        Dictionary mapping model key to instantiated estimator.
    """
    zoo: Dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            solver="lbfgs",
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state,
        ),
    }
    return zoo


def get_hyperparameter_grids() -> Dict[str, Dict[str, List[Any]]]:
    """Get candidate hyperparameter grids for tuning on training cross-validation."""
    grids: Dict[str, Dict[str, List[Any]]] = {
        "logistic_regression": {
            "C": [0.01, 0.1, 1.0, 10.0],
        },
        "svm_rbf": {
            "C": [0.1, 1.0, 10.0],
            "gamma": ["scale", 0.01, 0.1],
        },
        "random_forest": {
            "max_depth": [None, 5, 10, 20],
            "min_samples_leaf": [1, 2, 5],
        },
        "hist_gradient_boosting": {
            "learning_rate": [0.05, 0.1],
            "max_depth": [4, 6, 10],
        },
        "xgboost": {
            "max_depth": [3, 5, 8],
            "learning_rate": [0.05, 0.1],
        },
    }
    return grids
