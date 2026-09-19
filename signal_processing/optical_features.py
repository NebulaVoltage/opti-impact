"""Optical motion feature extraction and multimodal correlation module.

Reference Position Convention:
    Optical camera measurements report coordinates in camera pixel/millimeter space.
    To calculate true structural deflection, relative displacements are explicitly
    referenced to an equilibrium baseline:
        x_displacement(t) = optical_x(t) - reference_x
        y_displacement(t) = optical_y(t) - reference_y
    If reference_x and reference_y are not explicitly specified by the caller, they
    default to the signal mean (reference_x = mean(optical_x), reference_y = mean(optical_y)).
    
    Total displacement magnitude is strictly computed on these relative displacements:
        R(t) = sqrt(x_displacement(t)^2 + y_displacement(t)^2)
"""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy import signal

from signal_processing.frequency_features import get_dominant_frequency


def calculate_relative_displacement(
    optical_x: np.ndarray,
    optical_y: np.ndarray,
    reference_x: Optional[float] = None,
    reference_y: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute centered relative displacements and 2D displacement magnitude.

    Args:
        optical_x: 1D NumPy array of optical X coordinates (mm).
        optical_y: 1D NumPy array of optical Y coordinates (mm).
        reference_x: Optional baseline reference X coordinate. If None, uses mean(optical_x).
        reference_y: Optional baseline reference Y coordinate. If None, uses mean(optical_y).

    Returns:
        Tuple of (x_displacement, y_displacement, displacement_magnitude) as 1D NumPy arrays.
    """
    ox = np.asarray(optical_x, dtype=np.float64)
    oy = np.asarray(optical_y, dtype=np.float64)

    ref_x = float(np.mean(ox)) if reference_x is None else float(reference_x)
    ref_y = float(np.mean(oy)) if reference_y is None else float(reference_y)

    x_disp = ox - ref_x
    y_disp = oy - ref_y
    r_disp = np.sqrt(x_disp**2 + y_disp**2)

    return x_disp, y_disp, r_disp


def compute_optical_kinematics(
    x_displacement: np.ndarray,
    y_displacement: np.ndarray,
    sampling_rate: float,
    smoothing_method: Optional[str] = "savgol",
    smoothing_window: int = 11,
    polynomial_order: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute optical velocity and acceleration magnitude profiles.

    Addresses differentiation noise amplification:
        Direct numerical differentiation of discrete optical measurements heavily
        amplifies high-frequency sub-pixel noise and quantization jumps (scaling with
        f^2 for acceleration). To mitigate this, an optional zero-phase Savitzky-Golay
        filter is applied to the displacement before differentiation.

    Rationale for Savitzky-Golay default:
        - Preserves peak amplitudes, widths, and structural transient dynamics.
        - Zero phase delay (symmetric polynomial least-squares smoothing).
        - At fs = 100 Hz, an 11-point window (0.11 s) attenuates high-frequency noise
          while faithfully preserving structural oscillations up to ~10 Hz.
        - The centered displacement arrays remain unmodified.

    Args:
        x_displacement: 1D array of relative displacement X (mm).
        y_displacement: 1D array of relative displacement Y (mm).
        sampling_rate: Sampling frequency in Hz.
        smoothing_method: Smoothing algorithm ('savgol', 'none', or None). Default is 'savgol'.
        smoothing_window: Filter window length for Savitzky-Golay (must be odd positive int).
        polynomial_order: Polynomial order for Savitzky-Golay (default is 3).

    Returns:
        Tuple of (velocity_magnitude, acceleration_magnitude) in mm/s and mm/s^2.
    """
    n = len(x_displacement)
    if n < 2 or sampling_rate <= 0:
        return np.zeros_like(x_displacement), np.zeros_like(x_displacement)

    dt = 1.0 / sampling_rate
    x_work = np.asarray(x_displacement, dtype=np.float64)
    y_work = np.asarray(y_displacement, dtype=np.float64)

    # Optional smoothing stage
    if smoothing_method is not None and smoothing_method.strip().lower() == "savgol":
        # Validate window parameters
        win = int(smoothing_window)
        poly = int(polynomial_order)
        if win % 2 == 0:
            win += 1  # Window must be odd
        if win > poly and win <= n:
            x_work = signal.savgol_filter(x_work, window_length=win, polyorder=poly)
            y_work = signal.savgol_filter(y_work, window_length=win, polyorder=poly)

    # Velocity components (mm/s)
    vx = np.gradient(x_work, dt)
    vy = np.gradient(y_work, dt)
    velocity_mag = np.sqrt(vx**2 + vy**2)

    # Acceleration components (mm/s^2)
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    accel_mag = np.sqrt(ax**2 + ay**2)

    return velocity_mag, accel_mag


def compute_vibration_optical_correlation(
    vibration_signal: np.ndarray,
    optical_displacement: np.ndarray,
    sampling_rate: float = 100.0,
) -> Dict[str, float]:
    """Perform multimodal correlation analysis between vibration and optical displacement.

    Computes:
        - pearson_correlation: Zero-lag Pearson r correlation coefficient.
        - max_cross_correlation: Peak absolute normalized cross-correlation across all lags.
        - cross_correlation_lag_samples: Integer lag (in samples) where max correlation occurs.
        - cross_correlation_lag_seconds: Time delay (in seconds) corresponding to optimal alignment.

    Args:
        vibration_signal: 1D NumPy array of vibration/acceleration data.
        optical_displacement: 1D NumPy array of optical displacement data.
        sampling_rate: Sampling frequency in Hz.

    Returns:
        Dictionary of correlation and lag metrics.
    """
    v = np.asarray(vibration_signal, dtype=np.float64)
    o = np.asarray(optical_displacement, dtype=np.float64)

    n = min(len(v), len(o))
    if n < 2:
        return {
            "pearson_correlation": 0.0,
            "max_cross_correlation": 0.0,
            "cross_correlation_lag_samples": 0.0,
            "cross_correlation_lag_seconds": 0.0,
        }

    v = v[:n]
    o = o[:n]

    # 1. Pearson correlation at zero lag
    v_detrend = v - np.mean(v)
    o_detrend = o - np.mean(o)
    v_norm = np.linalg.norm(v_detrend)
    o_norm = np.linalg.norm(o_detrend)

    if v_norm > 1e-12 and o_norm > 1e-12:
        pearson_r = float(np.dot(v_detrend, o_detrend) / (v_norm * o_norm))
    else:
        pearson_r = 0.0

    # 2. Normalized cross-correlation across lags
    # Using scipy.signal.correlate
    if v_norm > 1e-12 and o_norm > 1e-12:
        raw_xcorr = signal.correlate(v_detrend, o_detrend, mode="full", method="auto")
        norm_factor = v_norm * o_norm
        norm_xcorr = raw_xcorr / norm_factor
        lags = signal.correlation_lags(len(v_detrend), len(o_detrend), mode="full")

        max_idx = int(np.argmax(np.abs(norm_xcorr)))
        max_xcorr = float(norm_xcorr[max_idx])
        optimal_lag_samples = int(lags[max_idx])
        dt = 1.0 / sampling_rate if sampling_rate > 0 else 0.0
        optimal_lag_seconds = float(optimal_lag_samples * dt)
    else:
        max_xcorr = 0.0
        optimal_lag_samples = 0
        optimal_lag_seconds = 0.0

    result = {
        "pearson_correlation": float(np.clip(pearson_r, -1.0, 1.0)),
        "max_cross_correlation": float(np.clip(max_xcorr, -1.0, 1.0)),
        "cross_correlation_lag_samples": float(optimal_lag_samples),
        "cross_correlation_lag_seconds": float(optimal_lag_seconds),
    }

    # Guard against NaN/inf
    for k, val in result.items():
        if np.isnan(val) or np.isinf(val):
            result[k] = 0.0

    return result


def extract_optical_features(
    optical_x: np.ndarray,
    optical_y: np.ndarray,
    optical_rotation: np.ndarray,
    sampling_rate: float,
    reference_x: Optional[float] = None,
    reference_y: Optional[float] = None,
    smoothing_method: Optional[str] = "savgol",
    smoothing_window: int = 11,
    polynomial_order: int = 3,
    min_freq: float = 0.5,
) -> Dict[str, float]:
    """Extract comprehensive engineering features from optical camera tracking signals.

    Args:
        optical_x: 1D array of measured optical X positions (mm).
        optical_y: 1D array of measured optical Y positions (mm).
        optical_rotation: 1D array of measured optical rotation (degrees).
        sampling_rate: Sampling frequency in Hz.
        reference_x: Optional equilibrium baseline X coordinate.
        reference_y: Optional equilibrium baseline Y coordinate.
        smoothing_method: Optional smoothing method for kinematics ('savgol', None).
        smoothing_window: Window length for Savitzky-Golay filter.
        polynomial_order: Polynomial order for Savitzky-Golay filter.
        min_freq: Minimum frequency in Hz for dominant frequency search (default 0.5 Hz).

    Returns:
        Dictionary of optical displacement, velocity, acceleration, and rotation features.
    """
    x_disp, y_disp, r_disp = calculate_relative_displacement(
        optical_x, optical_y, reference_x=reference_x, reference_y=reference_y
    )

    vel_mag, accel_mag = compute_optical_kinematics(
        x_disp,
        y_disp,
        sampling_rate,
        smoothing_method=smoothing_method,
        smoothing_window=smoothing_window,
        polynomial_order=polynomial_order,
    )

    rot = np.asarray(optical_rotation, dtype=np.float64)

    # 1. Optical displacement magnitude statistics (centered R(t) = sqrt(x_disp^2 + y_disp^2))
    # RMS explicitly defined as sqrt(mean(R(t)^2))
    disp_max = float(np.max(r_disp)) if len(r_disp) > 0 else 0.0
    disp_rms = float(np.sqrt(np.mean(r_disp**2))) if len(r_disp) > 0 else 0.0
    disp_mean = float(np.mean(r_disp)) if len(r_disp) > 0 else 0.0
    disp_std = float(np.std(r_disp)) if len(r_disp) > 0 else 0.0

    # 2. X and Y displacement statistics
    x_max = float(np.max(np.abs(x_disp))) if len(x_disp) > 0 else 0.0
    x_rms = float(np.sqrt(np.mean(x_disp**2))) if len(x_disp) > 0 else 0.0
    x_p2p = float(np.max(x_disp) - np.min(x_disp)) if len(x_disp) > 0 else 0.0

    y_max = float(np.max(np.abs(y_disp))) if len(y_disp) > 0 else 0.0
    y_rms = float(np.sqrt(np.mean(y_disp**2))) if len(y_disp) > 0 else 0.0
    y_p2p = float(np.max(y_disp) - np.min(y_disp)) if len(y_disp) > 0 else 0.0

    # 3. Kinematics (Velocity and Acceleration)
    vel_max = float(np.max(vel_mag)) if len(vel_mag) > 0 else 0.0
    vel_rms = float(np.sqrt(np.mean(vel_mag**2))) if len(vel_mag) > 0 else 0.0
    accel_max = float(np.max(accel_mag)) if len(accel_mag) > 0 else 0.0
    accel_rms = float(np.sqrt(np.mean(accel_mag**2))) if len(accel_mag) > 0 else 0.0

    # 4. Dominant frequency of primary optical displacement
    opt_dom_freq, opt_dom_amp = get_dominant_frequency(x_disp, sampling_rate, min_freq=min_freq)

    # 5. Rotation statistics
    rot_max = float(np.max(np.abs(rot))) if len(rot) > 0 else 0.0
    rot_rms = float(np.sqrt(np.mean(rot**2))) if len(rot) > 0 else 0.0
    rot_std = float(np.std(rot)) if len(rot) > 0 else 0.0

    features = {
        "optical_displacement_max": disp_max,
        "optical_displacement_rms": disp_rms,
        "optical_displacement_mean": disp_mean,
        "optical_displacement_std": disp_std,
        "optical_x_max": x_max,
        "optical_x_rms": x_rms,
        "optical_x_peak_to_peak": x_p2p,
        "optical_y_max": y_max,
        "optical_y_rms": y_rms,
        "optical_y_peak_to_peak": y_p2p,
        "optical_velocity_max": vel_max,
        "optical_velocity_rms": vel_rms,
        "optical_acceleration_max": accel_max,
        "optical_acceleration_rms": accel_rms,
        "optical_dominant_frequency": opt_dom_freq,
        "optical_frequency_amplitude": opt_dom_amp,
        "optical_rotation_max": rot_max,
        "optical_rotation_rms": rot_rms,
        "optical_rotation_std": rot_std,
    }

    # Guard against NaN/inf
    for k, v in features.items():
        if np.isnan(v) or np.isinf(v):
            features[k] = 0.0

    return features
