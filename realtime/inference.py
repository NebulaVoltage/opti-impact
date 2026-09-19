"""Real-time single-window inference module reusing Step 2 FeaturePipeline and Step 4 model bundle."""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from dataset.schema import FEATURE_COLUMNS, VALID_SCENARIOS
from ml.model_io import load_model_bundle
from realtime.config import InferenceConfig
from realtime.windowing import SensorWindow
from signal_processing.feature_pipeline import FeaturePipeline, StructuralFeatures


@dataclass
class WindowInferenceResult:
    """Encapsulates the single-window inference result, probabilities, and latencies.

    NOTE: Probabilities represent model-estimated class probabilities within the
    synthetic machine learning task, NOT structural failure or bridge collapse probabilities.
    """

    window_id: int
    start_time: float
    end_time: float
    instant_prediction: str
    probabilities: Dict[str, float]
    extracted_features: Dict[str, float]
    ground_truth_scenario: str
    is_transition_window: bool
    feature_extraction_latency_ms: float
    preprocessing_latency_ms: float
    model_inference_latency_ms: float
    total_latency_ms: float


class RealtimeInferenceEngine:
    """Executes single-window engineering feature extraction and ML classification."""

    def __init__(
        self,
        config: Optional[InferenceConfig] = None,
        feature_pipeline: Optional[FeaturePipeline] = None,
    ):
        """Initialize inference engine and load Step 4 serialized artifacts.

        Args:
            config: Optional InferenceConfig with paths and model prefix.
            feature_pipeline: Optional Step 2 FeaturePipeline instance.
        """
        self.config = config or InferenceConfig()
        self.feature_pipeline = feature_pipeline or FeaturePipeline()

        # Load Step 4 artifacts
        self.model, self.preprocessor, self.metadata = load_model_bundle(
            artifact_dir=self.config.models_dir,
            prefix=self.config.model_prefix,
        )

        # Validate loaded artifacts
        self._validate_artifacts()

    def _validate_artifacts(self) -> None:
        """Validate feature count, names, ordering, and target classes against schema."""
        if not hasattr(self.model, "predict"):
            raise ValueError("Loaded model does not have a predict() method.")

        expected_features = list(FEATURE_COLUMNS)
        prep_features = list(self.preprocessor.feature_columns)

        if len(prep_features) != len(expected_features):
            raise ValueError(
                f"Preprocessor feature count ({len(prep_features)}) != expected schema ({len(expected_features)})."
            )

        if prep_features != expected_features:
            raise ValueError("Preprocessor feature columns or ordering do not match schema FEATURE_COLUMNS.")

        model_classes = set(str(c) for c in getattr(self.model, "classes_", VALID_SCENARIOS))
        expected_classes = set(VALID_SCENARIOS)
        if not expected_classes.issubset(model_classes):
            raise ValueError(f"Model classes {model_classes} do not contain all expected classes {expected_classes}.")

    def predict_window(
        self,
        window: SensorWindow,
        sampling_rate: float = 100.0,
    ) -> WindowInferenceResult:
        """Extract engineering features and compute instantaneous ML classification.

        Args:
            window: SensorWindow of multi-channel sensor measurements.
            sampling_rate: Sampling frequency in Hz.

        Returns:
            WindowInferenceResult instance.
        """
        t_total_start = time.perf_counter()

        # 1. Feature Extraction (reusing Step 2 FeaturePipeline)
        t_feat_start = time.perf_counter()
        features: StructuralFeatures = self.feature_pipeline.extract_from_arrays(
            vibration=window.vibration,
            optical_x=window.optical_x,
            optical_y=window.optical_y,
            optical_rotation=window.optical_rotation,
            sampling_rate=sampling_rate,
            scenario=window.ground_truth_scenario,
        )
        t_feat_end = time.perf_counter()
        feat_latency_ms = (t_feat_end - t_feat_start) * 1000.0

        feat_dict = features.to_dict()
        if "scenario" in feat_dict:
            del feat_dict["scenario"]

        # Format into 1-row DataFrame strictly ordered by FEATURE_COLUMNS
        feat_df = pd.DataFrame([feat_dict])[FEATURE_COLUMNS]

        # 2. Preprocessing (reusing Step 4 preprocessor)
        t_prep_start = time.perf_counter()
        X_proc = self.preprocessor.transform(feat_df)
        t_prep_end = time.perf_counter()
        prep_latency_ms = (t_prep_end - t_prep_start) * 1000.0

        # 3. Model Inference (reusing Step 4 model)
        t_infer_start = time.perf_counter()
        instant_pred = str(self.model.predict(X_proc)[0])

        if hasattr(self.model, "predict_proba"):
            raw_probs = self.model.predict_proba(X_proc)[0]
            classes = [str(c) for c in self.model.classes_]
            prob_dict = {c: float(raw_probs[idx]) for idx, c in enumerate(classes)}
            for c_name in VALID_SCENARIOS:
                if c_name not in prob_dict:
                    prob_dict[c_name] = 0.0
        else:
            prob_dict = {c: (1.0 if c == instant_pred else 0.0) for c in VALID_SCENARIOS}
        t_infer_end = time.perf_counter()
        infer_latency_ms = (t_infer_end - t_infer_start) * 1000.0

        t_total_end = time.perf_counter()
        total_latency_ms = (t_total_end - t_total_start) * 1000.0

        return WindowInferenceResult(
            window_id=window.window_id,
            start_time=window.start_time,
            end_time=window.end_time,
            instant_prediction=instant_pred,
            probabilities=prob_dict,
            extracted_features=feat_dict,
            ground_truth_scenario=window.ground_truth_scenario,
            is_transition_window=window.is_transition_window,
            feature_extraction_latency_ms=feat_latency_ms,
            preprocessing_latency_ms=prep_latency_ms,
            model_inference_latency_ms=infer_latency_ms,
            total_latency_ms=total_latency_ms,
        )
