"""Frequency-domain analysis and spectral feature extraction module.

Normalization Convention:
    Uses standard single-sided amplitude normalization for real-valued FFT (rfft).
    For a windowed signal x[n] * w[n] of length N, with window weight sum W = sum(w[n]):
        A[0] = |X[0]| / W                 (DC component)
        A[k] = 2 * |X[k]| / W  (for k > 0, single-sided AC amplitude conservation)
    
    Spectral energy is defined consistently across the project as:
        spectral_energy = sum(A[k]^2)
    This represents the total sum of squared physical harmonic amplitudes (e.g. in (m/s^2)^2
    or mm^2) and is invariant to window normalization choices for signals of equal duration
    and sampling rate.

Dominant Frequency Decision:
    The zero-frequency DC component (0.0 Hz) and quasi-static environmental drift
    (< min_freq, default 0.5 Hz) are explicitly excluded from structural resonance
    identification. In structural health monitoring, thermal expansion, tidal movement,
    and slow wind sway occur at ultra-low frequencies (< 0.5 Hz), while dynamic structural
    modes of interest occur at higher frequencies. Excluding frequencies below 0.5 Hz prevents
    quasi-static drift and window leakage from being falsely identified as structural vibration modes.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy import signal


def compute_spectrum(
    signal_data: np.ndarray,
    sampling_rate: float,
    window: Optional[str] = "hann",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact single-sided frequency bins and calibrated amplitude spectrum.

    Args:
        signal_data: 1D NumPy array representing the physical signal.
        sampling_rate: Sampling frequency in Hz.
        window: Window function name ('hann', 'hamming', 'boxcar'/None).

    Returns:
        Tuple of (frequency_bins, amplitude_spectrum) as 1D NumPy arrays.
    """
    x = np.asarray(signal_data, dtype=np.float64)
    n = len(x)

    if n == 0 or sampling_rate <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Detrend by removing arithmetic mean before FFT
    x_detrended = x - np.mean(x)

    # Apply window
    if window == "hann" or window == "hanning":
        win = np.hanning(n)
    elif window == "hamming":
        win = np.hamming(n)
    elif window is None or window in ("none", "boxcar", "rect"):
        win = np.ones(n, dtype=np.float64)
    else:
        win = signal.get_window(window, n)

    w_sum = np.sum(win)
    if w_sum <= 0:
        w_sum = float(n)

    # Compute real FFT
    fft_complex = np.fft.rfft(x_detrended * win)
    freq_bins = np.fft.rfftfreq(n, d=1.0 / sampling_rate)

    # Single-sided amplitude spectrum calibration
    # DC component: |X[0]| / W
    # AC components (k > 0): 2 * |X[k]| / W
    amp_spectrum = np.abs(fft_complex) * (2.0 / w_sum)
    if len(amp_spectrum) > 0:
        amp_spectrum[0] = np.abs(fft_complex[0]) / w_sum

    return freq_bins, amp_spectrum


def get_dominant_frequency(
    signal_data: np.ndarray,
    sampling_rate: float,
    min_freq: float = 0.5,
    window: Optional[str] = "hann",
) -> Tuple[float, float]:
    """Identify the dominant oscillatory structural resonance frequency.

    Excludes DC (0 Hz) and quasi-static drift below `min_freq` (default 0.5 Hz)
    to ensure sensor offset or slow sway is not incorrectly reported as a
    structural resonance mode.

    Args:
        signal_data: 1D NumPy array of signal samples.
        sampling_rate: Sampling rate in Hz.
        min_freq: Minimum frequency threshold in Hz (default 0.5 Hz).
        window: Window name for spectral calculation.

    Returns:
        Tuple of (dominant_frequency, dominant_amplitude).
    """
    freq_bins, amp_spectrum = compute_spectrum(signal_data, sampling_rate, window=window)

    if len(freq_bins) == 0:
        return 0.0, 0.0

    # Mask frequencies above min_freq to exclude DC / slow drift
    valid_mask = freq_bins >= min_freq
    if not np.any(valid_mask):
        # Fallback to entire spectrum if all bins are below min_freq
        valid_mask = np.ones_like(freq_bins, dtype=bool)

    valid_freqs = freq_bins[valid_mask]
    valid_amps = amp_spectrum[valid_mask]

    peak_idx = int(np.argmax(valid_amps))
    dominant_freq = float(valid_freqs[peak_idx])
    dominant_amp = float(valid_amps[peak_idx])

    return dominant_freq, dominant_amp


def calculate_frequency_shift(
    current_dominant_frequency: float,
    baseline_dominant_frequency: float,
) -> Tuple[float, float]:
    """Calculate signed and absolute frequency shift relative to a baseline.

    Args:
        current_dominant_frequency: Current measured dominant frequency in Hz.
        baseline_dominant_frequency: Healthy/baseline reference frequency in Hz.

    Returns:
        Tuple of (frequency_shift, abs_frequency_shift) in Hz.
    """
    shift = float(current_dominant_frequency - baseline_dominant_frequency)
    abs_shift = float(abs(shift))
    return shift, abs_shift


def extract_frequency_features(
    signal_data: np.ndarray,
    sampling_rate: float,
    min_freq: float = 0.5,
    window: Optional[str] = "hann",
) -> Dict[str, float]:
    """Extract comprehensive frequency-domain spectral features.

    Features calculated:
        - dominant_frequency: Frequency corresponding to largest structural spectral peak (Hz).
        - dominant_frequency_amplitude: Calibrated single-sided amplitude at dominant peak.
        - spectral_energy: Total sum of squared spectral amplitudes: sum(A_k^2).
        - spectral_centroid: Spectral center of mass in Hz: sum(f_k * A_k) / sum(A_k).
        - spectral_bandwidth: Spectral spread/deviation around the centroid in Hz.
        - spectral_peak_count: Number of significant spectral peaks detected.

    Args:
        signal_data: 1D NumPy array representing the physical signal.
        sampling_rate: Sampling frequency in Hz.
        min_freq: Minimum frequency (Hz) for structural mode analysis (default 0.5 Hz).
        window: Windowing function (default 'hann').

    Returns:
        Dictionary mapping spectral feature names to float values.
    """
    freq_bins, amp_spectrum = compute_spectrum(signal_data, sampling_rate, window=window)

    if len(freq_bins) == 0:
        return {
            "dominant_frequency": 0.0,
            "dominant_frequency_amplitude": 0.0,
            "spectral_energy": 0.0,
            "spectral_centroid": 0.0,
            "spectral_bandwidth": 0.0,
            "spectral_peak_count": 0,
        }

    # Filter frequencies for structural mode analysis
    valid_mask = freq_bins >= min_freq
    if not np.any(valid_mask):
        valid_mask = np.ones_like(freq_bins, dtype=bool)

    valid_f = freq_bins[valid_mask]
    valid_a = amp_spectrum[valid_mask]

    # 1. Dominant frequency and amplitude
    peak_idx = int(np.argmax(valid_a))
    dom_freq = float(valid_f[peak_idx])
    dom_amp = float(valid_a[peak_idx])

    # 2. Consistent spectral energy: sum(A_k^2)
    spectral_energy = float(np.sum(amp_spectrum**2))

    # 3. Spectral Centroid
    total_amp = np.sum(valid_a)
    if total_amp > 1e-12:
        spectral_centroid = float(np.sum(valid_f * valid_a) / total_amp)
    else:
        spectral_centroid = 0.0

    # 4. Spectral Bandwidth (spectral spread around centroid)
    if total_amp > 1e-12:
        spectral_variance = np.sum(((valid_f - spectral_centroid) ** 2) * valid_a) / total_amp
        spectral_bandwidth = float(np.sqrt(max(0.0, spectral_variance)))
    else:
        spectral_bandwidth = 0.0

    # 5. Spectral Peak Count
    # Prominence threshold set to 10% of dominant amplitude to filter minor noise ripples
    prominence_threshold = max(1e-6, 0.10 * dom_amp)
    peaks, _ = signal.find_peaks(valid_a, prominence=prominence_threshold)
    peak_count = int(len(peaks))

    features = {
        "dominant_frequency": dom_freq,
        "dominant_frequency_amplitude": dom_amp,
        "spectral_energy": spectral_energy,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_peak_count": float(peak_count),
    }

    # Guard against NaN/inf
    for k, v in features.items():
        if np.isnan(v) or np.isinf(v):
            features[k] = 0.0

    return features
