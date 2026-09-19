"""Structured telemetry records for real-time inference windows."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class InferenceTelemetry:
    """Telemetry record captured for each streaming inference window.

    Encapsulates predictions, probabilities, temporal state, key physical engineering
    metrics, and sub-millisecond component profiling latencies.
    """

    timestamp: float
    window_id: int
    instant_prediction: str
    stable_state: str
    probability_normal: float
    probability_warning: float
    probability_critical: float
    vibration_rms: float
    vibration_peak: float
    dominant_frequency: float
    optical_displacement: float  # optical_displacement_max
    optical_velocity: float  # optical_velocity_max
    optical_acceleration: float  # optical_acceleration_max
    optical_dominant_frequency: float
    vibration_optical_correlation: float
    cross_correlation_max: float
    cross_correlation_lag: float
    state_duration: float
    ground_truth_scenario: str
    is_transition_window: bool
    candidate_state: Optional[str] = None
    candidate_count: int = 0
    feature_extraction_latency_ms: float = 0.0
    preprocessing_latency_ms: float = 0.0
    model_inference_latency_ms: float = 0.0
    temporal_decision_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert telemetry record to serializable dictionary."""
        return asdict(self)
