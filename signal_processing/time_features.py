"""Time-domain feature extraction module.

Extracts statistical and temporal vibration features from generic 1D NumPy arrays.
All features are protected against NaN and infinite values.
"""

from typing import Dict, Optional
import numpy as np
from scipy import stats


def extract_time_features(
    signal_data: np.ndarray,
    sampling_rate: Optional[float] = None,
) -> Dict[str, float]:
    """Extract time-domain statistical and kinematic features from a 1D signal.

    Features calculated:
        - mean: Arithmetic mean of the signal.
        - std: Standard deviation.
        - rms: Root-mean-square amplitude, calculated strictly as sqrt(mean(x^2)).
        - variance: Sample variance (sigma^2).
        - min: Minimum sample amplitude.
        - max: Maximum sample amplitude.
        - peak_to_peak: Total dynamic span (max - min).
        - peak: Peak absolute magnitude max(|x|).
        - crest_factor: Ratio of peak amplitude to RMS (peak / rms). Protected against zero division.
        - kurtosis: Fisher excess kurtosis (0.0 for ideal Gaussian noise).
        - skewness: Third standardized moment measuring signal asymmetry.
        - zero_crossing_rate: Normalized rate of sign changes across adjacent samples.
        - energy: Total signal energy, sum(x^2).

    Args:
        signal_data: 1D NumPy array representing the physical signal.
        sampling_rate: Optional sampling frequency in Hz. If provided, also computes
                       zero_crossings_per_second.

    Returns:
        Dictionary mapping feature names to float values.
    """
    x = np.asarray(signal_data, dtype=np.float64)

    if x.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "rms": 0.0,
            "variance": 0.0,
            "min": 0.0,
            "max": 0.0,
            "peak_to_peak": 0.0,
            "peak": 0.0,
            "crest_factor": 0.0,
            "kurtosis": 0.0,
            "skewness": 0.0,
            "zero_crossing_rate": 0.0,
            "energy": 0.0,
        }

    # Basic statistics
    sig_mean = float(np.mean(x))
    sig_std = float(np.std(x))
    sig_var = float(np.var(x))
    sig_min = float(np.min(x))
    sig_max = float(np.max(x))
    peak_to_peak = float(sig_max - sig_min)
    peak = float(np.max(np.abs(x)))

    # RMS: strictly defined as sqrt(mean(x^2))
    rms = float(np.sqrt(np.mean(x**2)))

    # Crest Factor: peak / RMS
    if rms > 1e-12:
        crest_factor = float(peak / rms)
    else:
        crest_factor = 0.0

    # Kurtosis and Skewness (with protection against constant signals)
    if sig_std > 1e-12:
        sig_kurtosis = float(stats.kurtosis(x, fisher=True, bias=False))
        sig_skewness = float(stats.skew(x, bias=False))
    else:
        sig_kurtosis = 0.0
        sig_skewness = 0.0

    # Zero-crossing rate: fraction of consecutive samples with differing signs
    if len(x) > 1:
        # Sign of samples (-1, 0, 1)
        signs = np.sign(x)
        # Count non-zero sign changes
        zero_crossings = np.sum(np.diff(signs) != 0)
        zero_crossing_rate = float(zero_crossings / (len(x) - 1))
    else:
        zero_crossing_rate = 0.0

    # Total signal energy: sum(x^2)
    energy = float(np.sum(x**2))

    features = {
        "mean": sig_mean,
        "std": sig_std,
        "rms": rms,
        "variance": sig_var,
        "min": sig_min,
        "max": sig_max,
        "peak_to_peak": peak_to_peak,
        "peak": peak,
        "crest_factor": crest_factor,
        "kurtosis": sig_kurtosis,
        "skewness": sig_skewness,
        "zero_crossing_rate": zero_crossing_rate,
        "energy": energy,
    }

    if sampling_rate is not None and sampling_rate > 0:
        duration = len(x) / sampling_rate
        features["zero_crossings_per_second"] = (
            float(zero_crossings / duration) if duration > 0 else 0.0
        )

    # Sanitize any unexpected non-finite values
    for k, v in features.items():
        if np.isnan(v) or np.isinf(v):
            features[k] = 0.0

    return features
