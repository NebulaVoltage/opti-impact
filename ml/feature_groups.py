"""Feature groups and modalities taxonomy for ML models and ablation studies.

Categorizes measurable engineering features into functional domains and
defines specific feature subsets for incremental information studies:
  - Group A: Vibration Only (Time + Frequency)
  - Group B: Optical Only (Displacement, Kinematics, Frequency, Rotation)
  - Group C: Vibration + Optical (without explicit cross-correlation)
  - Group D: Full Multimodal (All features including cross-correlation)
"""

from typing import Dict, List
from dataset.schema import FEATURE_COLUMNS

# 1. Functional / Domain Subsets
VIBRATION_TIME_FEATURES: List[str] = [
    "vibration_mean",
    "vibration_std",
    "vibration_rms",
    "vibration_variance",
    "vibration_peak",
    "vibration_peak_to_peak",
    "vibration_crest_factor",
    "vibration_kurtosis",
    "vibration_skewness",
    "vibration_zero_crossing_rate",
    "vibration_energy",
]

VIBRATION_FREQ_FEATURES: List[str] = [
    "dominant_frequency",
    "dominant_frequency_amplitude",
    "spectral_energy",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_peak_count",
]

OPTICAL_FEATURES: List[str] = [
    "optical_displacement_max",
    "optical_displacement_rms",
    "optical_displacement_mean",
    "optical_displacement_std",
    "optical_velocity_max",
    "optical_velocity_rms",
    "optical_acceleration_max",
    "optical_acceleration_rms",
    "optical_dominant_frequency",
    "optical_frequency_amplitude",
    "optical_rotation_max",
    "optical_rotation_rms",
]

MULTIMODAL_FEATURES: List[str] = [
    "vibration_optical_correlation",
    "cross_correlation_max",
    "cross_correlation_lag_seconds",
]

# High-level domain mapping for interpretability and feature importance grouping
FEATURE_DOMAIN_MAP: Dict[str, str] = {}
for f in VIBRATION_TIME_FEATURES:
    FEATURE_DOMAIN_MAP[f] = "VIBRATION"
for f in VIBRATION_FREQ_FEATURES:
    FEATURE_DOMAIN_MAP[f] = "FREQUENCY"
for f in OPTICAL_FEATURES:
    FEATURE_DOMAIN_MAP[f] = "OPTICAL"
for f in MULTIMODAL_FEATURES:
    FEATURE_DOMAIN_MAP[f] = "MULTIMODAL"

# 2. Ablation Study Groups
# Group A — Vibration Only: vibration time-domain + frequency features
GROUP_A_VIBRATION_ONLY: List[str] = VIBRATION_TIME_FEATURES + VIBRATION_FREQ_FEATURES

# Group B — Optical Only: optical displacement, velocity, acceleration, frequency, rotation
GROUP_B_OPTICAL_ONLY: List[str] = OPTICAL_FEATURES.copy()

# Group C — Vibration + Optical: both modalities but excluding cross-sensor correlations
GROUP_C_VIBRATION_AND_OPTICAL: List[str] = (
    VIBRATION_TIME_FEATURES + VIBRATION_FREQ_FEATURES + OPTICAL_FEATURES
)

# Group D — Full Multimodal: all measurable features including cross-sensor correlations
GROUP_D_FULL_MULTIMODAL: List[str] = FEATURE_COLUMNS.copy()

ABLATION_GROUPS: Dict[str, List[str]] = {
    "group_a_vibration_only": GROUP_A_VIBRATION_ONLY,
    "group_b_optical_only": GROUP_B_OPTICAL_ONLY,
    "group_c_vibration_and_optical": GROUP_C_VIBRATION_AND_OPTICAL,
    "group_d_full_multimodal": GROUP_D_FULL_MULTIMODAL,
}
