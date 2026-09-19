"""Feature extraction pipeline module.

Converts raw simulated or physical vibration and optical displacement time-series
into structured engineering features for machine learning and condition assessment.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Ensure repo root is available for direct execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from signal_processing.filters import preprocess_signal
from signal_processing.time_features import extract_time_features
from signal_processing.frequency_features import extract_frequency_features
from signal_processing.optical_features import (
    extract_optical_features,
    compute_vibration_optical_correlation,
    calculate_relative_displacement,
)


@dataclass
class StructuralFeatures:
    """Dataclass encapsulating extracted engineering features for a structural event.

    Features are categorized into:
    1. Vibration Time-Domain Features (RMS = sqrt(mean(x^2)), energy = sum(x^2))
    2. Vibration Frequency-Domain Features (spectral energy = sum(A_k^2))
    3. Optical Kinematics & Deflection Features (RMS displacement = sqrt(mean(R^2)))
    4. Multimodal Optical-Vibration Correlation
    5. Event Scenario Metadata
    
    Note:
        RMS is strictly calculated as sqrt(mean(x^2)) and never confused with mean-square energy.
    """

    # 1. Vibration Time-Domain Features
    vibration_mean: float
    vibration_std: float
    vibration_rms: float
    vibration_variance: float
    vibration_peak: float
    vibration_peak_to_peak: float
    vibration_crest_factor: float
    vibration_kurtosis: float
    vibration_skewness: float
    vibration_zero_crossing_rate: float
    vibration_energy: float

    # 2. Vibration Frequency-Domain Features
    dominant_frequency: float
    dominant_frequency_amplitude: float
    spectral_energy: float
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_peak_count: float

    # 3. Optical Features
    optical_displacement_max: float
    optical_displacement_rms: float
    optical_displacement_mean: float
    optical_displacement_std: float
    optical_velocity_max: float
    optical_velocity_rms: float
    optical_acceleration_max: float
    optical_acceleration_rms: float
    optical_dominant_frequency: float
    optical_frequency_amplitude: float
    optical_rotation_max: float
    optical_rotation_rms: float

    # 4. Multimodal Correlation Features
    vibration_optical_correlation: float
    cross_correlation_max: float
    cross_correlation_lag_seconds: float

    # 5. Metadata
    scenario: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert features dataclass to a standard Python dictionary."""
        return asdict(self)

    def to_dataframe_row(self) -> pd.DataFrame:
        """Convert features to a single-row pandas DataFrame."""
        return pd.DataFrame([self.to_dict()])

    def to_series(self) -> pd.Series:
        """Convert features to a pandas Series."""
        return pd.Series(self.to_dict())


class FeaturePipeline:
    """End-to-end signal preprocessing and engineering feature extraction pipeline.

    Operates on both generic NumPy arrays and SimulationEvent objects.
    Maintains raw and processed signals separately.
    """

    def __init__(
        self,
        remove_mean: bool = True,
        filter_type: Optional[str] = None,
        lowcut: Optional[float] = None,
        highcut: Optional[float] = None,
        filter_order: int = 4,
        min_freq: float = 0.5,
        smoothing_method: Optional[str] = "savgol",
        smoothing_window: int = 11,
        polynomial_order: int = 3,
    ):
        """Initialize pipeline with preprocessing parameters.

        Args:
            remove_mean: Whether to remove DC offset / mean from raw vibration (default True).
            filter_type: Optional filter type ('bandpass', 'highpass', 'lowpass', None).
            lowcut: Lower cutoff frequency (Hz).
            highcut: Upper cutoff frequency (Hz).
            filter_order: Butterworth filter order (default 4).
            min_freq: Minimum frequency (Hz) for structural resonance mode search (default 0.5 Hz).
            smoothing_method: Optical kinematics differentiation smoothing ('savgol', None). Default 'savgol'.
            smoothing_window: Savitzky-Golay filter window length (default 11 samples).
            polynomial_order: Savitzky-Golay polynomial order (default 3).
        """
        self.remove_mean = remove_mean
        self.filter_type = filter_type
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.min_freq = min_freq
        self.smoothing_method = smoothing_method
        self.smoothing_window = smoothing_window
        self.polynomial_order = polynomial_order

        # Cache last processed signals for diagnostic inspection
        self.last_raw_vibration: Optional[np.ndarray] = None
        self.last_processed_vibration: Optional[np.ndarray] = None

    def extract_from_arrays(
        self,
        vibration: np.ndarray,
        optical_x: np.ndarray,
        optical_y: np.ndarray,
        optical_rotation: np.ndarray,
        sampling_rate: float,
        scenario: Optional[str] = None,
        reference_x: Optional[float] = None,
        reference_y: Optional[float] = None,
    ) -> StructuralFeatures:
        """Extract engineering features from generic numerical arrays.

        This decoupled method supports real sensor data (e.g. from accelerometers
        and optical tracking cameras) without depending on the simulation module.

        Args:
            vibration: 1D array of vibration/acceleration samples (m/s^2).
            optical_x: 1D array of optical tracking X coordinates (mm).
            optical_y: 1D array of optical tracking Y coordinates (mm).
            optical_rotation: 1D array of optical rotation values (degrees).
            sampling_rate: Sampling frequency in Hz.
            scenario: Optional scenario or condition label.
            reference_x: Optional baseline equilibrium X coordinate.
            reference_y: Optional baseline equilibrium Y coordinate.

        Returns:
            StructuralFeatures instance.
        """
        # 1. Preprocessing: preserve raw and processed vibration separately
        raw_vib, proc_vib = preprocess_signal(
            vibration,
            sampling_rate=sampling_rate,
            remove_mean=self.remove_mean,
            filter_type=self.filter_type,
            lowcut=self.lowcut,
            highcut=self.highcut,
            order=self.filter_order,
        )
        self.last_raw_vibration = raw_vib
        self.last_processed_vibration = proc_vib

        # 2. Time-domain vibration features
        time_feats = extract_time_features(proc_vib, sampling_rate=sampling_rate)

        # 3. Frequency-domain vibration features
        freq_feats = extract_frequency_features(
            proc_vib, sampling_rate=sampling_rate, min_freq=self.min_freq
        )

        # 4. Optical features (with centered relative displacements and optional kinematics smoothing)
        opt_feats = extract_optical_features(
            optical_x=optical_x,
            optical_y=optical_y,
            optical_rotation=optical_rotation,
            sampling_rate=sampling_rate,
            reference_x=reference_x,
            reference_y=reference_y,
            smoothing_method=self.smoothing_method,
            smoothing_window=self.smoothing_window,
            polynomial_order=self.polynomial_order,
            min_freq=self.min_freq,
        )

        # 5. Multimodal correlation between vibration and relative optical displacement
        x_disp, _, _ = calculate_relative_displacement(
            optical_x, optical_y, reference_x=reference_x, reference_y=reference_y
        )
        corr_feats = compute_vibration_optical_correlation(
            vibration_signal=proc_vib,
            optical_displacement=x_disp,
            sampling_rate=sampling_rate,
        )

        return StructuralFeatures(
            # Time-domain
            vibration_mean=time_feats["mean"],
            vibration_std=time_feats["std"],
            vibration_rms=time_feats["rms"],
            vibration_variance=time_feats["variance"],
            vibration_peak=time_feats["peak"],
            vibration_peak_to_peak=time_feats["peak_to_peak"],
            vibration_crest_factor=time_feats["crest_factor"],
            vibration_kurtosis=time_feats["kurtosis"],
            vibration_skewness=time_feats["skewness"],
            vibration_zero_crossing_rate=time_feats["zero_crossing_rate"],
            vibration_energy=time_feats["energy"],
            # Frequency-domain
            dominant_frequency=freq_feats["dominant_frequency"],
            dominant_frequency_amplitude=freq_feats["dominant_frequency_amplitude"],
            spectral_energy=freq_feats["spectral_energy"],
            spectral_centroid=freq_feats["spectral_centroid"],
            spectral_bandwidth=freq_feats["spectral_bandwidth"],
            spectral_peak_count=freq_feats["spectral_peak_count"],
            # Optical
            optical_displacement_max=opt_feats["optical_displacement_max"],
            optical_displacement_rms=opt_feats["optical_displacement_rms"],
            optical_displacement_mean=opt_feats["optical_displacement_mean"],
            optical_displacement_std=opt_feats["optical_displacement_std"],
            optical_velocity_max=opt_feats["optical_velocity_max"],
            optical_velocity_rms=opt_feats["optical_velocity_rms"],
            optical_acceleration_max=opt_feats["optical_acceleration_max"],
            optical_acceleration_rms=opt_feats["optical_acceleration_rms"],
            optical_dominant_frequency=opt_feats["optical_dominant_frequency"],
            optical_frequency_amplitude=opt_feats["optical_frequency_amplitude"],
            optical_rotation_max=opt_feats["optical_rotation_max"],
            optical_rotation_rms=opt_feats["optical_rotation_rms"],
            # Correlation
            vibration_optical_correlation=corr_feats["pearson_correlation"],
            cross_correlation_max=corr_feats["max_cross_correlation"],
            cross_correlation_lag_seconds=corr_feats["cross_correlation_lag_seconds"],
            # Metadata
            scenario=scenario,
        )

    def extract(self, event: Any) -> StructuralFeatures:
        """Extract features directly from a SimulationEvent.

        Args:
            event: SimulationEvent instance containing time, vibration, optical_x,
                   optical_y, optical_rotation, and config.

        Returns:
            StructuralFeatures instance.
        """
        sampling_rate = float(event.config.sampling_rate)
        return self.extract_from_arrays(
            vibration=event.vibration,
            optical_x=event.optical_x,
            optical_y=event.optical_y,
            optical_rotation=event.optical_rotation,
            sampling_rate=sampling_rate,
            scenario=event.scenario,
        )


def main():
    """Demonstration entry point for feature pipeline across NORMAL, WARNING, CRITICAL."""
    from simulation.simulator import StructuralSimulator

    sim = StructuralSimulator(seed=42)
    pipeline = FeaturePipeline(remove_mean=True)

    print("=" * 70)
    print("STEP 2: FEATURE EXTRACTION PIPELINE DEMONSTRATION")
    print("=" * 70)

    for scn in ["NORMAL", "WARNING", "CRITICAL"]:
        event = sim.generate(scn, seed=42)
        features = pipeline.extract(event)

        print(f"\nScenario: {scn}")
        print(f"  Vibration RMS:                  {features.vibration_rms:.4f} m/s²")
        print(f"  Peak:                           {features.vibration_peak:.4f} m/s²")
        print(f"  Dominant Frequency:             {features.dominant_frequency:.2f} Hz")
        print(f"  Spectral Energy:                {features.spectral_energy:.4f} (m/s²)²")
        print(f"  Optical Displacement:           {features.optical_displacement_max:.4f} mm")
        print(f"  Optical Velocity:               {features.optical_velocity_max:.4f} mm/s")
        print(f"  Optical Acceleration:           {features.optical_acceleration_max:.4f} mm/s²")
        print(f"  Optical Dominant Frequency:     {features.optical_dominant_frequency:.2f} Hz")
        print(f"  Vibration/Optical Correlation:  {features.vibration_optical_correlation:.4f}")
        print(f"  Max Cross-Correlation:          {features.cross_correlation_max:.4f} (lag: {features.cross_correlation_lag_seconds:.3f} s)")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
