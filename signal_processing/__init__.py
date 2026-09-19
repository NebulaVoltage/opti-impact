"""Signal processing and engineering feature extraction package.

Provides modular signal preprocessing, time-domain and frequency-domain
feature extraction, optical kinematics analysis, and end-to-end feature pipelines.
"""

from typing import Any

__all__ = [
    "remove_dc",
    "bandpass_filter",
    "highpass_filter",
    "lowpass_filter",
    "preprocess_signal",
    "extract_time_features",
    "compute_spectrum",
    "get_dominant_frequency",
    "calculate_frequency_shift",
    "extract_frequency_features",
    "calculate_relative_displacement",
    "compute_optical_kinematics",
    "compute_vibration_optical_correlation",
    "extract_optical_features",
    "StructuralFeatures",
    "FeaturePipeline",
]


def __getattr__(name: str) -> Any:
    """Lazy-load package symbols to prevent circular import warnings with runpy."""
    if name in (
        "remove_dc",
        "bandpass_filter",
        "highpass_filter",
        "lowpass_filter",
        "preprocess_signal",
    ):
        import signal_processing.filters as m

        return getattr(m, name)
    elif name == "extract_time_features":
        import signal_processing.time_features as m

        return getattr(m, name)
    elif name in (
        "compute_spectrum",
        "get_dominant_frequency",
        "calculate_frequency_shift",
        "extract_frequency_features",
    ):
        import signal_processing.frequency_features as m

        return getattr(m, name)
    elif name in (
        "calculate_relative_displacement",
        "compute_optical_kinematics",
        "compute_vibration_optical_correlation",
        "extract_optical_features",
    ):
        import signal_processing.optical_features as m

        return getattr(m, name)
    elif name in ("StructuralFeatures", "FeaturePipeline"):
        import signal_processing.feature_pipeline as m

        return getattr(m, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
