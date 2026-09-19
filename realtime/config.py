"""Configuration definitions for real-time multimodal inference and temporal monitoring."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class StreamConfig:
    """Sensor stream and windowing configuration."""

    sampling_rate: float = 100.0  # Hz
    window_duration: float = 5.0  # seconds
    inference_step: float = 1.0  # seconds

    @property
    def window_size(self) -> int:
        """Total number of samples per window."""
        return int(round(self.sampling_rate * self.window_duration))

    @property
    def step_size(self) -> int:
        """Number of samples between successive inference windows."""
        return int(round(self.sampling_rate * self.inference_step))


@dataclass
class TemporalEngineConfig:
    """Temporal state decision engine and persistence configuration."""

    warning_persistence: int = 3  # Consecutive WARNING predictions required for NORMAL -> WARNING
    critical_persistence: int = 3  # Consecutive CRITICAL predictions required for WARNING -> CRITICAL
    recovery_persistence: int = 3  # Consecutive NORMAL predictions required to restore NORMAL state
    valid_states: List[str] = field(default_factory=lambda: ["NORMAL", "WARNING", "CRITICAL"])


@dataclass
class InferenceConfig:
    """Model loading and feature extraction configuration."""

    models_dir: Path = Path("models/step4")
    model_prefix: str = "step4_v1"
    min_feature_count: int = 32


@dataclass
class RealtimePipelineConfig:
    """Master configuration for real-time monitoring runtime."""

    stream: StreamConfig = field(default_factory=StreamConfig)
    temporal: TemporalEngineConfig = field(default_factory=TemporalEngineConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    output_dir: Path = Path("outputs/step5")
    inference_log_file: str = "inference_log.csv"
    state_events_file: str = "state_events.jsonl"
    summary_file: str = "realtime_summary.json"
