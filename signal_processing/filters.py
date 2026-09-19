"""Signal preprocessing and filtering module.

Provides configurable zero-phase filtering and DC offset removal using
numerically stable second-order sections (SOS) IIR Butterworth filters.
All functions operate on generic 1D NumPy arrays and sampling rates.
"""

from typing import Optional, Tuple
import numpy as np
from scipy import signal


def remove_dc(signal_data: np.ndarray) -> np.ndarray:
    """Remove DC offset / arithmetic mean from a signal.

    Args:
        signal_data: 1D NumPy array representing input signal.

    Returns:
        Zero-mean 1D NumPy array.
    """
    if signal_data.size == 0:
        return signal_data.copy()
    return signal_data - np.mean(signal_data)


def bandpass_filter(
    signal_data: np.ndarray,
    sampling_rate: float,
    lowcut: float,
    highcut: float,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth band-pass filter using second-order sections (SOS).

    Args:
        signal_data: 1D NumPy array representing input signal.
        sampling_rate: Sampling frequency in Hz. Must be > 0.
        lowcut: Lower cutoff frequency in Hz. Must be > 0.
        highcut: Upper cutoff frequency in Hz. Must be < sampling_rate / 2.
        order: Filter order (default is 4).

    Returns:
        Filtered 1D NumPy array of the same length.
    """
    if signal_data.size == 0:
        return signal_data.copy()

    nyquist = 0.5 * sampling_rate
    if lowcut <= 0.0:
        raise ValueError(f"lowcut must be positive, got {lowcut} Hz")
    if highcut >= nyquist:
        raise ValueError(
            f"highcut ({highcut} Hz) must be less than the Nyquist frequency ({nyquist} Hz)"
        )
    if lowcut >= highcut:
        raise ValueError(f"lowcut ({lowcut} Hz) must be strictly less than highcut ({highcut} Hz)")

    # Normalize cutoffs to Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist

    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, signal_data)


def highpass_filter(
    signal_data: np.ndarray,
    sampling_rate: float,
    cutoff: float,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth high-pass filter using second-order sections (SOS).

    Args:
        signal_data: 1D NumPy array representing input signal.
        sampling_rate: Sampling frequency in Hz.
        cutoff: Cutoff frequency in Hz. Must be in (0, sampling_rate / 2).
        order: Filter order (default is 4).

    Returns:
        Filtered 1D NumPy array of the same length.
    """
    if signal_data.size == 0:
        return signal_data.copy()

    nyquist = 0.5 * sampling_rate
    if not (0.0 < cutoff < nyquist):
        raise ValueError(f"cutoff must be between 0 and Nyquist ({nyquist} Hz), got {cutoff} Hz")

    sos = signal.butter(order, cutoff / nyquist, btype="highpass", output="sos")
    return signal.sosfiltfilt(sos, signal_data)


def lowpass_filter(
    signal_data: np.ndarray,
    sampling_rate: float,
    cutoff: float,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth low-pass filter using second-order sections (SOS).

    Args:
        signal_data: 1D NumPy array representing input signal.
        sampling_rate: Sampling frequency in Hz.
        cutoff: Cutoff frequency in Hz. Must be in (0, sampling_rate / 2).
        order: Filter order (default is 4).

    Returns:
        Filtered 1D NumPy array of the same length.
    """
    if signal_data.size == 0:
        return signal_data.copy()

    nyquist = 0.5 * sampling_rate
    if not (0.0 < cutoff < nyquist):
        raise ValueError(f"cutoff must be between 0 and Nyquist ({nyquist} Hz), got {cutoff} Hz")

    sos = signal.butter(order, cutoff / nyquist, btype="lowpass", output="sos")
    return signal.sosfiltfilt(sos, signal_data)


def preprocess_signal(
    signal_data: np.ndarray,
    sampling_rate: float,
    remove_mean: bool = True,
    filter_type: Optional[str] = None,
    lowcut: Optional[float] = None,
    highcut: Optional[float] = None,
    order: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute configured signal preprocessing pipeline.

    Preserves both the original raw signal and the processed signal.

    Pipeline:
        raw signal
        ↓
        remove mean (optional, default True)
        ↓
        optional filtering (None, 'bandpass', 'highpass', 'lowpass')
        ↓
        processed signal

    Args:
        signal_data: 1D NumPy array of raw sensor or simulation samples.
        sampling_rate: Sampling frequency in Hz.
        remove_mean: Whether to subtract the arithmetic mean.
        filter_type: Optional filter type: None, 'bandpass', 'highpass', 'lowpass'.
        lowcut: Lower cutoff frequency (Hz) for bandpass or cutoff for highpass.
        highcut: Upper cutoff frequency (Hz) for bandpass or cutoff for lowpass.
        order: Filter order (default 4).

    Returns:
        Tuple of (raw_signal, processed_signal) as 1D NumPy arrays.
    """
    raw = np.asarray(signal_data, dtype=np.float64)
    processed = raw.copy()

    if remove_mean:
        processed = remove_dc(processed)

    if filter_type is not None:
        ftype = filter_type.strip().lower()
        if ftype == "bandpass":
            if lowcut is None or highcut is None:
                raise ValueError("Both lowcut and highcut must be provided for bandpass filter")
            processed = bandpass_filter(processed, sampling_rate, lowcut, highcut, order=order)
        elif ftype == "highpass":
            cutoff = lowcut if lowcut is not None else highcut
            if cutoff is None:
                raise ValueError("cutoff frequency must be provided for highpass filter")
            processed = highpass_filter(processed, sampling_rate, cutoff, order=order)
        elif ftype == "lowpass":
            cutoff = highcut if highcut is not None else lowcut
            if cutoff is None:
                raise ValueError("cutoff frequency must be provided for lowpass filter")
            processed = lowpass_filter(processed, sampling_rate, cutoff, order=order)
        elif ftype in ("none", ""):
            pass
        else:
            raise ValueError(
                f"Unknown filter_type '{filter_type}'. Choose from 'bandpass', 'highpass', 'lowpass', None"
            )

    return raw, processed
