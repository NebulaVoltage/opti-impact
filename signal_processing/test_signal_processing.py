"""Unit and integration tests for signal processing and feature extraction.

Tests:
1. Filter returns correct signal length.
2. No NaN values are produced across features and filtered signals.
3. No infinite values are produced.
4. RMS is mathematically reasonable: RMS = sqrt(mean(x^2)).
5. Peak-to-peak calculation is correct: max(x) - min(x).
6. FFT frequency axis is correct: starts at 0, uniform step fs/N, length N//2 + 1.
7. Known synthetic sine wave produces approximately the expected dominant frequency.
8. DC component is not incorrectly selected as dominant structural frequency.
9. Optical displacement magnitude is calculated correctly: R(t) = sqrt(x_disp^2 + y_disp^2).
10. Velocity and acceleration arrays have correct lengths.
11. FeaturePipeline returns all expected features.
12. Feature conversion to dictionary works.
13. Feature conversion to DataFrame works.
14. Same simulated event produces reproducible features.
15. NORMAL/WARNING/CRITICAL events can all pass through the pipeline.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pytest

from simulation.simulator import StructuralSimulator
from signal_processing.filters import (
    bandpass_filter,
    highpass_filter,
    lowpass_filter,
    preprocess_signal,
    remove_dc,
)
from signal_processing.time_features import extract_time_features
from signal_processing.frequency_features import (
    compute_spectrum,
    get_dominant_frequency,
    calculate_frequency_shift,
    extract_frequency_features,
)
from signal_processing.optical_features import (
    calculate_relative_displacement,
    compute_optical_kinematics,
    compute_vibration_optical_correlation,
    extract_optical_features,
)
from signal_processing.feature_pipeline import FeaturePipeline, StructuralFeatures


# 1. Filter returns correct signal length
def test_filter_returns_correct_signal_length():
    fs = 100.0
    sig = np.random.default_rng(42).normal(0, 1, 1000)
    filtered = bandpass_filter(sig, fs, lowcut=1.0, highcut=20.0, order=4)
    assert len(filtered) == len(sig)

    raw, proc = preprocess_signal(sig, fs, remove_mean=True, filter_type="bandpass", lowcut=1.0, highcut=20.0)
    assert len(raw) == len(sig)
    assert len(proc) == len(sig)


# 2. No NaN values are produced
def test_no_nan_values_produced():
    sim = StructuralSimulator(seed=42)
    event = sim.generate("NORMAL")
    pipeline = FeaturePipeline()
    features = pipeline.extract(event)
    feat_dict = features.to_dict()

    for k, v in feat_dict.items():
        if isinstance(v, (int, float)):
            assert not np.isnan(v), f"NaN found in feature '{k}'"


# 3. No infinite values are produced
def test_no_infinite_values_produced():
    sim = StructuralSimulator(seed=42)
    event = sim.generate("CRITICAL")
    pipeline = FeaturePipeline()
    features = pipeline.extract(event)
    feat_dict = features.to_dict()

    for k, v in feat_dict.items():
        if isinstance(v, (int, float)):
            assert not np.isinf(v), f"Inf found in feature '{k}'"


# 4. RMS is mathematically reasonable: RMS = sqrt(mean(x^2))
def test_rms_calculation():
    # Test on known analytical signal: x = A * sin(2*pi*f*t)
    # Theoretical RMS for sine wave is A / sqrt(2)
    fs = 1000.0
    t = np.arange(10000) / fs
    amplitude = 4.0
    x = amplitude * np.sin(2 * np.pi * 5.0 * t)

    feats = extract_time_features(x, sampling_rate=fs)
    expected_rms = amplitude / np.sqrt(2.0)
    assert np.isclose(feats["rms"], expected_rms, rtol=1e-3)
    # Verify strict formula
    assert np.isclose(feats["rms"], np.sqrt(np.mean(x**2)))


# 5. Peak-to-peak calculation is correct
def test_peak_to_peak_calculation():
    x = np.array([-3.5, 1.2, 0.0, 7.8, -2.1, 4.4])
    feats = extract_time_features(x)
    assert np.isclose(feats["peak_to_peak"], 7.8 - (-3.5))
    assert np.isclose(feats["peak"], 7.8)


# 6. FFT frequency axis is correct
def test_fft_frequency_axis():
    n = 1000
    fs = 100.0
    x = np.random.default_rng(42).normal(0, 1, n)
    freqs, amp = compute_spectrum(x, fs, window=None)

    expected_len = n // 2 + 1
    assert len(freqs) == expected_len
    assert len(amp) == expected_len
    assert freqs[0] == 0.0
    assert np.isclose(freqs[1] - freqs[0], fs / n)
    assert np.isclose(freqs[-1], fs / 2.0)


# 7. Known synthetic sine wave produces approximately the expected dominant frequency
def test_known_sine_wave_dominant_frequency():
    fs = 100.0
    duration = 5.0
    t = np.arange(int(fs * duration)) / fs
    target_freq = 10.0
    # Clean sine wave at 10 Hz
    x = 2.5 * np.sin(2 * np.pi * target_freq * t)

    dom_freq, dom_amp = get_dominant_frequency(x, fs, min_freq=0.2)
    # Bin resolution is 1 / 5.0 = 0.2 Hz
    assert abs(dom_freq - target_freq) <= 0.25, f"Expected ~{target_freq} Hz, got {dom_freq} Hz"
    assert dom_amp > 1.5


# 8. DC component is not incorrectly selected as dominant structural frequency
def test_dc_component_not_selected_as_dominant_frequency():
    fs = 100.0
    t = np.arange(1000) / fs
    osc_freq = 6.0
    # Add large static DC offset (+50.0) to an oscillating mode (amplitude 2.0)
    x = 50.0 + 2.0 * np.sin(2 * np.pi * osc_freq * t)

    dom_freq, _ = get_dominant_frequency(x, fs, min_freq=0.5)
    assert dom_freq != 0.0, "DC component 0 Hz was incorrectly selected as dominant frequency"
    assert abs(dom_freq - osc_freq) <= 0.2, f"Expected dominant freq ~{osc_freq} Hz, got {dom_freq} Hz"


# 9. Optical displacement magnitude calculated correctly
def test_optical_displacement_magnitude_calculation():
    # 3-4-5 right triangle: x = 3, y = 4 -> R = 5
    ox = np.array([3.0, 6.0, 0.0])
    oy = np.array([4.0, 8.0, 0.0])

    # Using explicit reference at (0, 0)
    x_disp, y_disp, r_disp = calculate_relative_displacement(ox, oy, reference_x=0.0, reference_y=0.0)
    assert np.allclose(x_disp, ox)
    assert np.allclose(y_disp, oy)
    assert np.allclose(r_disp, np.array([5.0, 10.0, 0.0]))


# 10. Velocity and acceleration arrays have correct lengths
def test_kinematics_lengths():
    n = 500
    fs = 100.0
    x_disp = np.random.default_rng(42).normal(0, 1, n)
    y_disp = np.random.default_rng(43).normal(0, 1, n)

    vel_mag, accel_mag = compute_optical_kinematics(x_disp, y_disp, sampling_rate=fs)
    assert len(vel_mag) == n
    assert len(accel_mag) == n


# 11. FeaturePipeline returns all expected features
def test_feature_pipeline_returns_all_expected_features():
    sim = StructuralSimulator(seed=42)
    event = sim.generate("NORMAL")
    pipeline = FeaturePipeline()
    features = pipeline.extract(event)

    assert isinstance(features, StructuralFeatures)
    assert features.vibration_rms > 0.0
    assert features.dominant_frequency > 0.0
    assert features.optical_displacement_max > 0.0
    assert features.optical_velocity_max > 0.0
    assert features.optical_acceleration_max > 0.0
    assert features.scenario == "NORMAL"


# 12. Feature conversion to dictionary works
def test_feature_to_dict():
    sim = StructuralSimulator(seed=42)
    event = sim.generate("WARNING")
    features = FeaturePipeline().extract(event)
    d = features.to_dict()

    assert isinstance(d, dict)
    assert "vibration_rms" in d
    assert "dominant_frequency" in d
    assert "optical_displacement_max" in d
    assert "vibration_optical_correlation" in d
    assert d["scenario"] == "WARNING"


# 13. Feature conversion to DataFrame works
def test_feature_to_dataframe():
    sim = StructuralSimulator(seed=42)
    event = sim.generate("CRITICAL")
    features = FeaturePipeline().extract(event)
    df = features.to_dataframe_row()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "dominant_frequency" in df.columns
    assert df["scenario"].iloc[0] == "CRITICAL"


# 14. Same simulated event produces reproducible features
def test_reproducible_features():
    sim = StructuralSimulator(seed=123)
    event1 = sim.generate("NORMAL", seed=42)
    event2 = sim.generate("NORMAL", seed=42)

    pipeline = FeaturePipeline()
    f1 = pipeline.extract(event1)
    f2 = pipeline.extract(event2)

    assert f1.to_dict() == f2.to_dict()


# 15. NORMAL/WARNING/CRITICAL events can all pass through pipeline
@pytest.mark.parametrize("scn", ["NORMAL", "WARNING", "CRITICAL"])
def test_all_scenarios_pass_through_pipeline(scn: str):
    sim = StructuralSimulator(seed=42)
    event = sim.generate(scn, seed=100)
    pipeline = FeaturePipeline()
    feats = pipeline.extract(event)

    assert feats.scenario == scn
    assert feats.vibration_rms > 0.0
    assert feats.dominant_frequency > 0.0
    assert -1.0 <= feats.vibration_optical_correlation <= 1.0


def test_frequency_shift_calculation():
    baseline_freq = 5.0
    warning_freq = 4.3
    shift, abs_shift = calculate_frequency_shift(warning_freq, baseline_freq)
    assert np.isclose(shift, -0.7)
    assert np.isclose(abs_shift, 0.7)


@pytest.mark.parametrize("scenario_name", ["NORMAL", "WARNING", "CRITICAL"])
def test_physical_consistency_and_multimodal_coherence(scenario_name: str):
    """Verify that vibration and optical signals remain physically coherent."""
    sim = StructuralSimulator(seed=42)
    event = sim.generate(scenario_name, seed=42)
    pipeline = FeaturePipeline()
    features = pipeline.extract(event)

    # 1. Natural frequency agreement between optical and vibration
    freq_diff = abs(features.dominant_frequency - features.optical_dominant_frequency)
    assert freq_diff <= 0.5, (
        f"{scenario_name}: Frequency mismatch between vibration ({features.dominant_frequency:.2f} Hz) "
        f"and optical ({features.optical_dominant_frequency:.2f} Hz), diff = {freq_diff:.2f} Hz"
    )

    # 2. Acceleration and displacement are in phase opposition: negative correlation
    assert features.vibration_optical_correlation < 0.0, (
        f"{scenario_name}: Expected negative correlation between acceleration and displacement, "
        f"got {features.vibration_optical_correlation:.4f}"
    )

    # 3. Peak cross-correlation occurs near zero lag (|lag| <= 0.15 s)
    assert abs(features.cross_correlation_lag_seconds) <= 0.15, (
        f"{scenario_name}: Peak cross-correlation lag ({features.cross_correlation_lag_seconds:.3f} s) "
        f"exceeds allowable tolerance"
    )

    # 4. Optical acceleration with Savitzky-Golay smoothing is physically realistic (not noise-dominated)
    # Theoretical peak acceleration ~ omega^2 * disp
    w = 2.0 * np.pi * features.dominant_frequency
    theoretical_accel = (w**2) * features.optical_displacement_max
    # Savitzky-Golay smoothed acceleration should be within a factor of 1.5 of theoretical peak
    assert features.optical_acceleration_max <= 1.5 * theoretical_accel, (
        f"{scenario_name}: Smoothed optical acceleration ({features.optical_acceleration_max:.1f} mm/s^2) "
        f"exceeds theoretical bound (1.5 * {theoretical_accel:.1f} mm/s^2)"
    )


def test_kinematics_smoothing_configurability():
    """Verify that smoothing can be toggled and that it suppresses differentiation noise."""
    sim = StructuralSimulator(seed=42)
    event = sim.generate("WARNING", seed=42)
    x_disp, y_disp, _ = calculate_relative_displacement(event.optical_x, event.optical_y)
    fs = float(event.config.sampling_rate)

    # Raw differentiation
    _, accel_raw = compute_optical_kinematics(x_disp, y_disp, fs, smoothing_method=None)
    # Smoothed differentiation
    _, accel_smooth = compute_optical_kinematics(x_disp, y_disp, fs, smoothing_method="savgol", smoothing_window=11)

    max_raw = np.max(accel_raw)
    max_smooth = np.max(accel_smooth)

    # Savitzky-Golay should significantly reduce noise amplification
    assert max_smooth < max_raw
    assert max_smooth < 0.7 * max_raw, f"Smoothing should reduce noise spikes: raw={max_raw:.1f}, smooth={max_smooth:.1f}"
