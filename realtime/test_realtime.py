"""Comprehensive unit and integration test suite for Step 5 real-time monitoring."""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from dataset.schema import FEATURE_COLUMNS, VALID_SCENARIOS
from ml.model_io import load_model_bundle
from realtime.config import (
    InferenceConfig,
    RealtimePipelineConfig,
    StreamConfig,
    TemporalEngineConfig,
)
from realtime.data_stream import SensorChunk, SyntheticSensorStream
from realtime.event_logger import EventLogger
from realtime.inference import RealtimeInferenceEngine, WindowInferenceResult
from realtime.runtime import RealtimeMonitoringRuntime
from realtime.state_machine import StateTransitionEvent
from realtime.telemetry import InferenceTelemetry
from realtime.temporal_engine import TemporalDecisionEngine
from realtime.windowing import SensorWindow, StreamingWindowManager


# ---------------------------------------------------------------------------
# 1. Model Loading Tests
# ---------------------------------------------------------------------------


def test_model_and_preprocessor_load():
    """Verify that Step 4 model, preprocessor, and metadata load cleanly."""
    model, preprocessor, metadata = load_model_bundle(Path("models/step4"), prefix="step4_v1")
    assert model is not None
    assert preprocessor is not None
    assert metadata is not None
    assert metadata["model_version"] == "step4_v1"


def test_class_labels_match():
    """Verify that model classes match canonical VALID_SCENARIOS."""
    model, _, _ = load_model_bundle(Path("models/step4"), prefix="step4_v1")
    classes = list(model.classes_)
    assert set(classes) == set(VALID_SCENARIOS)


def test_feature_dimensions_and_ordering():
    """Verify that preprocessor feature names strictly match schema FEATURE_COLUMNS."""
    _, preprocessor, _ = load_model_bundle(Path("models/step4"), prefix="step4_v1")
    assert len(preprocessor.feature_columns) == len(FEATURE_COLUMNS)
    assert list(preprocessor.feature_columns) == list(FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# 2. Sensor Stream Abstraction Tests
# ---------------------------------------------------------------------------


def test_synthetic_stream_creation():
    """Verify SyntheticSensorStream generates synchronized continuous signals."""
    schedule = [("NORMAL", 5.0), ("WARNING", 5.0)]
    stream = SyntheticSensorStream(scenario_schedule=schedule, sampling_rate=100.0, base_seed=42)

    assert stream.total_samples == 1000
    assert stream.has_more()


def test_stream_timestamps_are_monotonic():
    """Verify stream timestamps are strictly monotonically increasing with correct dt."""
    schedule = [("NORMAL", 2.0), ("CRITICAL", 2.0)]
    stream = SyntheticSensorStream(scenario_schedule=schedule, sampling_rate=100.0, base_seed=42)

    chunk = stream.read_chunk(400)
    assert chunk is not None
    assert len(chunk) == 400
    diffs = np.diff(chunk.timestamp)
    assert np.all(diffs > 0)
    np.testing.assert_allclose(diffs, 0.01, atol=1e-4)


def test_stream_channels_exist_and_synchronized():
    """Verify that vibration, optical_x, optical_y, and optical_rotation are present and equal length."""
    schedule = [("WARNING", 3.0)]
    stream = SyntheticSensorStream(scenario_schedule=schedule, sampling_rate=100.0, base_seed=42)
    chunk = stream.read_chunk(300)
    assert chunk is not None

    assert chunk.vibration is not None
    assert chunk.optical_x is not None
    assert chunk.optical_y is not None
    assert chunk.optical_rotation is not None
    assert len(chunk.vibration) == 300
    assert len(chunk.optical_x) == 300
    assert len(chunk.optical_y) == 300
    assert len(chunk.optical_rotation) == 300


def test_sensor_chunk_validation():
    """Verify validation detects channel length mismatch and non-finite values."""
    # Length mismatch
    with pytest.raises(ValueError, match="does not match timestamp length"):
        bad_chunk = SensorChunk(
            timestamp=np.array([0.0, 0.01, 0.02]),
            vibration=np.array([0.1, 0.2]),
            optical_x=np.array([0.1, 0.2, 0.3]),
            optical_y=np.array([0.1, 0.2, 0.3]),
            optical_rotation=np.array([0.0, 0.0, 0.0]),
        )
        bad_chunk.validate()

    # NaN in vibration
    with pytest.raises(ValueError, match="contains NaN"):
        nan_chunk = SensorChunk(
            timestamp=np.array([0.0, 0.01, 0.02]),
            vibration=np.array([0.1, np.nan, 0.3]),
            optical_x=np.array([0.1, 0.2, 0.3]),
            optical_y=np.array([0.1, 0.2, 0.3]),
            optical_rotation=np.array([0.0, 0.0, 0.0]),
        )
        nan_chunk.validate()


# ---------------------------------------------------------------------------
# 3. Windowing Buffer Tests
# ---------------------------------------------------------------------------


def test_window_manager_correct_window_and_step_size():
    """Verify StreamingWindowManager extracts correct window size and advances by step size."""
    cfg = StreamConfig(sampling_rate=100.0, window_duration=5.0, inference_step=1.0)
    wm = StreamingWindowManager(config=cfg)

    schedule = [("NORMAL", 10.0)]
    stream = SyntheticSensorStream(scenario_schedule=schedule, sampling_rate=100.0, base_seed=42)
    chunk = stream.read_chunk(1000)
    assert chunk is not None

    wm.add_chunk(chunk)
    assert wm.has_window()

    win1 = wm.get_next_window()
    assert win1 is not None
    assert len(win1) == 500  # 5 seconds at 100 Hz
    assert win1.window_id == 0
    assert pytest.approx(win1.start_time, 0.01) == 0.0
    assert pytest.approx(win1.end_time, 0.01) == 4.99

    # Second window should start at 1.0s (step size 100 samples)
    win2 = wm.get_next_window()
    assert win2 is not None
    assert len(win2) == 500
    assert win2.window_id == 1
    assert pytest.approx(win2.start_time, 0.01) == 1.0
    assert pytest.approx(win2.end_time, 0.01) == 5.99


def test_window_manager_insufficient_samples():
    """Verify window manager returns None when fewer than window_size samples are buffered."""
    wm = StreamingWindowManager(window_size=500, step_size=100)
    chunk = SensorChunk(
        timestamp=np.linspace(0, 2.99, 300),
        vibration=np.zeros(300),
        optical_x=np.zeros(300),
        optical_y=np.zeros(300),
        optical_rotation=np.zeros(300),
    )
    wm.add_chunk(chunk)
    assert not wm.has_window()
    assert wm.get_next_window() is None


def test_window_manager_transition_detection():
    """Verify that windows spanning multiple scenarios are flagged as transition windows."""
    schedule = [("NORMAL", 4.0), ("WARNING", 4.0)]
    stream = SyntheticSensorStream(scenario_schedule=schedule, sampling_rate=100.0, base_seed=42)
    chunk = stream.read_chunk(800)
    assert chunk is not None

    wm = StreamingWindowManager(window_size=500, step_size=100)
    wm.add_chunk(chunk)

    # Window 0: 0-5s contains 0-4s NORMAL and 4-5s WARNING -> transition!
    win0 = wm.get_next_window()
    assert win0 is not None
    assert win0.is_transition_window is True
    assert "NORMAL" in win0.scenario_proportions
    assert "WARNING" in win0.scenario_proportions


# ---------------------------------------------------------------------------
# 4. Single-Window Inference Tests
# ---------------------------------------------------------------------------


def test_inference_engine_predict_window():
    """Verify RealtimeInferenceEngine extracts features and outputs valid prediction and probabilities."""
    engine = RealtimeInferenceEngine()
    schedule = [("NORMAL", 6.0)]
    stream = SyntheticSensorStream(scenario_schedule=schedule, sampling_rate=100.0, base_seed=42)
    wm = StreamingWindowManager(window_size=500, step_size=100)
    chunk = stream.read_chunk(600)
    assert chunk is not None
    wm.add_chunk(chunk)

    win = wm.get_next_window()
    assert win is not None

    res: WindowInferenceResult = engine.predict_window(win)
    assert res.instant_prediction in VALID_SCENARIOS
    assert len(res.probabilities) == 3
    for p in res.probabilities.values():
        assert 0.0 <= p <= 1.0
    assert pytest.approx(sum(res.probabilities.values()), rel=1e-3) == 1.0
    assert len(res.extracted_features) == 32
    assert res.total_latency_ms > 0.0


# ---------------------------------------------------------------------------
# 5. Temporal Decision Engine & Hysteresis Tests
# ---------------------------------------------------------------------------


def test_single_warning_does_not_trigger_transition():
    """A single isolated WARNING prediction must not change stable state from NORMAL."""
    engine = TemporalDecisionEngine(TemporalEngineConfig(warning_persistence=3))
    assert engine.current_state == "NORMAL"

    state, event = engine.process_prediction("WARNING", timestamp=1.0, window_id=1)
    assert state == "NORMAL"
    assert event is None
    assert engine.candidate_state == "WARNING"
    assert engine.candidate_count == 1


def test_two_warnings_do_not_trigger_transition():
    """Two consecutive WARNING predictions must not trigger transition when threshold is 3."""
    engine = TemporalDecisionEngine(TemporalEngineConfig(warning_persistence=3))
    engine.process_prediction("WARNING", timestamp=1.0, window_id=1)
    state, event = engine.process_prediction("WARNING", timestamp=2.0, window_id=2)

    assert state == "NORMAL"
    assert event is None
    assert engine.candidate_count == 2


def test_three_warnings_trigger_transition():
    """Three consecutive WARNING predictions must confirm transition to WARNING."""
    engine = TemporalDecisionEngine(TemporalEngineConfig(warning_persistence=3))
    engine.process_prediction("WARNING", timestamp=1.0, window_id=1)
    engine.process_prediction("WARNING", timestamp=2.0, window_id=2)
    state, event = engine.process_prediction("WARNING", timestamp=3.0, window_id=3)

    assert state == "WARNING"
    assert event is not None
    assert event.previous_state == "NORMAL"
    assert event.new_state == "WARNING"
    assert event.consecutive_windows == 3
    assert len(engine.transition_history) == 1


def test_one_critical_does_not_trigger_critical():
    """An isolated CRITICAL prediction from WARNING state must not immediately trigger CRITICAL."""
    engine = TemporalDecisionEngine(TemporalEngineConfig(critical_persistence=3), initial_state="WARNING")
    state, event = engine.process_prediction("CRITICAL", timestamp=10.0, window_id=10)

    assert state == "WARNING"
    assert event is None
    assert engine.candidate_state == "CRITICAL"
    assert engine.candidate_count == 1


def test_three_criticals_trigger_critical():
    """Three consecutive CRITICAL predictions must transition WARNING -> CRITICAL."""
    engine = TemporalDecisionEngine(TemporalEngineConfig(critical_persistence=3), initial_state="WARNING")
    engine.process_prediction("CRITICAL", timestamp=10.0, window_id=10)
    engine.process_prediction("CRITICAL", timestamp=11.0, window_id=11)
    state, event = engine.process_prediction("CRITICAL", timestamp=12.0, window_id=12)

    assert state == "CRITICAL"
    assert event is not None
    assert event.previous_state == "WARNING"
    assert event.new_state == "CRITICAL"


def test_recovery_requires_persistence():
    """Recovering from CRITICAL to NORMAL requires 3 consecutive NORMAL predictions."""
    engine = TemporalDecisionEngine(TemporalEngineConfig(recovery_persistence=3), initial_state="CRITICAL")

    # 1st NORMAL
    state, _ = engine.process_prediction("NORMAL", timestamp=20.0, window_id=20)
    assert state == "CRITICAL"

    # 2nd NORMAL
    state, _ = engine.process_prediction("NORMAL", timestamp=21.0, window_id=21)
    assert state == "CRITICAL"

    # 3rd NORMAL
    state, event = engine.process_prediction("NORMAL", timestamp=22.0, window_id=22)
    assert state == "NORMAL"
    assert event is not None
    assert event.previous_state == "CRITICAL"
    assert event.new_state == "NORMAL"


def test_candidate_counter_resets_on_interruption():
    """If a sequence of candidate predictions is interrupted by current state, counter resets."""
    engine = TemporalDecisionEngine(TemporalEngineConfig(warning_persistence=3), initial_state="NORMAL")

    engine.process_prediction("WARNING", timestamp=1.0, window_id=1)
    engine.process_prediction("WARNING", timestamp=2.0, window_id=2)
    assert engine.candidate_count == 2

    # Interrupted by NORMAL
    state, event = engine.process_prediction("NORMAL", timestamp=3.0, window_id=3)
    assert state == "NORMAL"
    assert event is None
    assert engine.candidate_state is None
    assert engine.candidate_count == 0


def test_hysteresis_prevents_rapid_oscillation():
    """Alternating predictions (WARNING, NORMAL, WARNING, NORMAL) must never change stable state."""
    engine = TemporalDecisionEngine(TemporalEngineConfig(warning_persistence=3), initial_state="NORMAL")

    sequence = ["WARNING", "NORMAL", "WARNING", "NORMAL", "WARNING", "NORMAL"]
    for idx, pred in enumerate(sequence):
        state, event = engine.process_prediction(pred, timestamp=float(idx), window_id=idx)
        assert state == "NORMAL"
        assert event is None

    assert len(engine.transition_history) == 0


# ---------------------------------------------------------------------------
# 6. Robustness & Error Handling Tests
# ---------------------------------------------------------------------------


def test_invalid_prediction_raises_error():
    """Passing an unknown scenario string to temporal engine must raise ValueError."""
    engine = TemporalDecisionEngine()
    with pytest.raises(ValueError, match="Invalid prediction"):
        engine.process_prediction("CATASTROPHIC_FAILURE", timestamp=1.0, window_id=1)


def test_model_loading_failure():
    """Attempting to load from an invalid directory must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        RealtimeInferenceEngine(config=InferenceConfig(models_dir=Path("non_existent_dir_12345")))


# ---------------------------------------------------------------------------
# 7. Serialization & Logging Tests
# ---------------------------------------------------------------------------


def test_telemetry_and_event_logger():
    """Verify EventLogger records telemetry CSV, transition JSONL, and summary JSON."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = EventLogger(output_dir=Path(tmp_dir))

        telemetry = InferenceTelemetry(
            timestamp=5.0,
            window_id=0,
            instant_prediction="NORMAL",
            stable_state="NORMAL",
            probability_normal=0.99,
            probability_warning=0.01,
            probability_critical=0.0,
            vibration_rms=0.25,
            vibration_peak=0.8,
            dominant_frequency=5.1,
            optical_displacement=0.6,
            optical_velocity=15.0,
            optical_acceleration=300.0,
            optical_dominant_frequency=5.1,
            vibration_optical_correlation=-0.25,
            cross_correlation_max=0.5,
            cross_correlation_lag=0.0,
            state_duration=5.0,
            ground_truth_scenario="NORMAL",
            is_transition_window=False,
            total_latency_ms=12.5,
        )
        logger.log_telemetry(telemetry)

        trans = StateTransitionEvent(
            timestamp=10.0,
            previous_state="NORMAL",
            new_state="WARNING",
            reason="persistent_model_prediction",
            consecutive_windows=3,
            window_id=5,
        )
        logger.log_transition(trans)

        csv_path = logger.save_inference_csv()
        summary = logger.save_summary(total_runtime=1.5, final_state="WARNING")

        assert csv_path.exists()
        assert logger.jsonl_path.exists()
        assert logger.summary_path.exists()
        assert summary["number_of_windows"] == 1
        assert summary["number_of_state_transitions"] == 1
        assert summary["mean_latency"] == 12.5

        # Read back CSV
        df_read = pd.read_csv(csv_path)
        assert len(df_read) == 1
        assert df_read.iloc[0]["instant_prediction"] == "NORMAL"


# ---------------------------------------------------------------------------
# 8. End-to-End Runtime Integration Test
# ---------------------------------------------------------------------------


def test_optical_data_is_independent_sensing_channel():
    """Verify that optical measurements remain an independent sensing channel and are extracted as expected."""
    engine = RealtimeInferenceEngine()
    # Create window with zero vibration but real optical displacement
    n = 500
    win = SensorWindow(
        window_id=99,
        start_time=0.0,
        end_time=4.99,
        vibration=np.zeros(n),
        optical_x=np.sin(np.linspace(0, 10 * np.pi, n)) * 2.5,
        optical_y=np.cos(np.linspace(0, 10 * np.pi, n)) * 2.5,
        optical_rotation=np.zeros(n),
        timestamps=np.linspace(0, 4.99, n),
        ground_truth_scenario="NORMAL",
        is_transition_window=False,
        scenario_proportions={"NORMAL": 1.0},
    )
    res = engine.predict_window(win)
    assert res.extracted_features["optical_displacement_max"] > 2.0
    assert res.extracted_features["vibration_rms"] == 0.0


def test_state_machine_transition_event_dataclass():
    """Verify StateTransitionEvent creation and serialization."""
    event = StateTransitionEvent(
        timestamp=12.34,
        previous_state="NORMAL",
        new_state="WARNING",
        reason="persistent_model_prediction",
        consecutive_windows=3,
        window_id=14,
    )
    d = event.to_dict()
    assert d["timestamp"] == 12.34
    assert d["previous_state"] == "NORMAL"
    assert d["new_state"] == "WARNING"
    assert d["consecutive_windows"] == 3
    assert d["window_id"] == 14


def test_window_manager_clear():
    """Verify StreamingWindowManager clear() empties all buffers and resets window counter."""
    wm = StreamingWindowManager(window_size=500, step_size=100)
    chunk = SensorChunk(
        timestamp=np.linspace(0, 1.99, 200),
        vibration=np.zeros(200),
        optical_x=np.zeros(200),
        optical_y=np.zeros(200),
        optical_rotation=np.zeros(200),
    )
    wm.add_chunk(chunk)
    assert wm.buffered_samples == 200

    wm.clear()
    assert wm.buffered_samples == 0
    assert not wm.has_window()


def test_runtime_end_to_end():
    """Verify RealtimeMonitoringRuntime executes full multi-window pipeline to completion."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = RealtimePipelineConfig(output_dir=Path(tmp_dir))
        # Run a short 12-second sequence (NORMAL -> WARNING)
        schedule = [("NORMAL", 6.0), ("WARNING", 6.0)]
        stream = SyntheticSensorStream(schedule, sampling_rate=100.0, base_seed=42)

        runtime = RealtimeMonitoringRuntime(config=cfg, stream=stream)
        summary = runtime.run_all()

        assert summary["number_of_windows"] > 0
        assert summary["mean_latency"] > 0.0
        assert (Path(tmp_dir) / cfg.inference_log_file).exists()
        assert (Path(tmp_dir) / cfg.summary_file).exists()
