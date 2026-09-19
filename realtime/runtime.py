"""Master runtime coordinator for real-time streaming multimodal monitoring."""

from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from realtime.config import RealtimePipelineConfig
from realtime.data_stream import SensorStream, SyntheticSensorStream
from realtime.event_logger import EventLogger
from realtime.inference import RealtimeInferenceEngine, WindowInferenceResult
from realtime.state_machine import StateTransitionEvent
from realtime.telemetry import InferenceTelemetry
from realtime.temporal_engine import TemporalDecisionEngine
from realtime.windowing import SensorWindow, StreamingWindowManager


class RealtimeMonitoringRuntime:
    """End-to-end runtime executing streaming windowing, feature extraction, ML, and temporal decisions."""

    def __init__(
        self,
        config: Optional[RealtimePipelineConfig] = None,
        stream: Optional[SensorStream] = None,
        inference_engine: Optional[RealtimeInferenceEngine] = None,
        temporal_engine: Optional[TemporalDecisionEngine] = None,
        event_logger: Optional[EventLogger] = None,
    ):
        """Initialize the real-time monitoring runtime.

        Args:
            config: Optional RealtimePipelineConfig.
            stream: SensorStream source (synthetic or physical).
            inference_engine: RealtimeInferenceEngine instance.
            temporal_engine: TemporalDecisionEngine instance.
            event_logger: EventLogger instance.
        """
        self.config = config or RealtimePipelineConfig()
        self.stream = stream
        self.window_manager = StreamingWindowManager(config=self.config.stream)
        self.inference_engine = inference_engine or RealtimeInferenceEngine(config=self.config.inference)
        self.temporal_engine = temporal_engine or TemporalDecisionEngine(config=self.config.temporal)
        self.event_logger = event_logger or EventLogger(output_dir=self.config.output_dir)

        self._start_wall_time: Optional[float] = None
        self._total_wall_time: float = 0.0

    def process_window(self, window: SensorWindow) -> Tuple[InferenceTelemetry, Optional[StateTransitionEvent]]:
        """Process a single sensor window through feature extraction, ML, and temporal hysteresis."""
        # 1. Single-window ML inference
        infer_res: WindowInferenceResult = self.inference_engine.predict_window(
            window=window,
            sampling_rate=self.config.stream.sampling_rate,
        )

        # 2. Temporal decision engine
        t_temp_start = time.perf_counter()
        stable_state, transition_event = self.temporal_engine.process_prediction(
            instant_pred=infer_res.instant_prediction,
            timestamp=infer_res.end_time,
            window_id=infer_res.window_id,
        )
        t_temp_end = time.perf_counter()
        temp_latency_ms = (t_temp_end - t_temp_start) * 1000.0

        if transition_event is not None:
            self.event_logger.log_transition(transition_event)

        # 3. Extract key telemetry features
        feats = infer_res.extracted_features
        telemetry = InferenceTelemetry(
            timestamp=infer_res.end_time,
            window_id=infer_res.window_id,
            instant_prediction=infer_res.instant_prediction,
            stable_state=stable_state,
            probability_normal=infer_res.probabilities.get("NORMAL", 0.0),
            probability_warning=infer_res.probabilities.get("WARNING", 0.0),
            probability_critical=infer_res.probabilities.get("CRITICAL", 0.0),
            vibration_rms=feats.get("vibration_rms", 0.0),
            vibration_peak=feats.get("vibration_peak", 0.0),
            dominant_frequency=feats.get("dominant_frequency", 0.0),
            optical_displacement=feats.get("optical_displacement_max", 0.0),
            optical_velocity=feats.get("optical_velocity_max", 0.0),
            optical_acceleration=feats.get("optical_acceleration_max", 0.0),
            optical_dominant_frequency=feats.get("optical_dominant_frequency", 0.0),
            vibration_optical_correlation=feats.get("vibration_optical_correlation", 0.0),
            cross_correlation_max=feats.get("cross_correlation_max", 0.0),
            cross_correlation_lag=feats.get("cross_correlation_lag_seconds", 0.0),
            state_duration=self.temporal_engine.state_duration(infer_res.end_time),
            ground_truth_scenario=infer_res.ground_truth_scenario,
            is_transition_window=infer_res.is_transition_window,
            candidate_state=self.temporal_engine.candidate_state,
            candidate_count=self.temporal_engine.candidate_count,
            feature_extraction_latency_ms=infer_res.feature_extraction_latency_ms,
            preprocessing_latency_ms=infer_res.preprocessing_latency_ms,
            model_inference_latency_ms=infer_res.model_inference_latency_ms,
            temporal_decision_latency_ms=temp_latency_ms,
            total_latency_ms=infer_res.total_latency_ms + temp_latency_ms,
        )

        self.event_logger.log_telemetry(telemetry)
        return telemetry, transition_event

    def step(self) -> Optional[Tuple[InferenceTelemetry, Optional[StateTransitionEvent]]]:
        """Perform one step: ingest samples into window manager and evaluate next window if ready."""
        # Read from stream if more samples needed
        if self.stream is not None and not self.window_manager.has_window() and self.stream.has_more():
            chunk = self.stream.read_chunk(self.config.stream.window_size)
            if chunk is not None:
                self.window_manager.add_chunk(chunk)

        # Ingest incremental chunks if stream has more
        while self.stream is not None and not self.window_manager.has_window() and self.stream.has_more():
            chunk = self.stream.read_chunk(self.config.stream.step_size)
            if chunk is not None:
                self.window_manager.add_chunk(chunk)
            else:
                break

        window = self.window_manager.get_next_window()
        if window is None:
            return None

        return self.process_window(window)

    def run_all(
        self,
        callback: Optional[Callable[[InferenceTelemetry, Optional[StateTransitionEvent]], None]] = None,
    ) -> Dict[str, Any]:
        """Run the streaming pipeline to completion across the entire input stream.

        Args:
            callback: Optional callback invoked on each window telemetry and transition event.

        Returns:
            Dictionary containing session summary and latency metrics.
        """
        if self.stream is None:
            raise ValueError("Cannot run pipeline without an attached SensorStream.")

        t_wall_start = time.perf_counter()

        # Ingest stream and evaluate all windows
        while self.stream.has_more():
            chunk = self.stream.read_chunk(self.config.stream.window_size)
            if chunk is not None:
                self.window_manager.add_chunk(chunk)

            while self.window_manager.has_window():
                window = self.window_manager.get_next_window()
                if window is None:
                    break
                telemetry, transition_event = self.process_window(window)
                if callback is not None:
                    callback(telemetry, transition_event)

        # Process any remaining full windows in buffer
        while self.window_manager.has_window():
            window = self.window_manager.get_next_window()
            if window is None:
                break
            telemetry, transition_event = self.process_window(window)
            if callback is not None:
                callback(telemetry, transition_event)

        t_wall_end = time.perf_counter()
        total_wall_time = t_wall_end - t_wall_start

        # Save artifacts
        self.event_logger.save_inference_csv()
        summary = self.event_logger.save_summary(
            total_runtime=total_wall_time,
            final_state=self.temporal_engine.current_state,
        )

        return summary
