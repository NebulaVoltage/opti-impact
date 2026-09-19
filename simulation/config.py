"""Configuration module for structural simulation scenarios.

Note:
    The numerical parameters defined herein represent synthetic prototype
    configurations for developing and validating signal processing and
    machine learning pipelines. They do NOT represent real-world civil
    engineering safety or certified bridge structural health thresholds.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScenarioConfig:
    """Configuration dataclass for structural response simulations.

    Attributes:
        name: Name of the scenario ('NORMAL', 'WARNING', 'CRITICAL', or custom).
        natural_frequency: Baseline natural resonance frequency of the structure in Hz.
        amplitude: Baseline displacement amplitude scale in millimeters (mm).
        damping_ratio: Dimensionless viscous damping ratio (zeta, where 0 < zeta < 1).
        impact_strength: Amplitude scaling factor for impact impulse excitation.
        noise_level: Root-mean-square amplitude of broadband environmental ambient excitation.
        frequency_shift: Frequency delta (in Hz) applied to natural frequency to represent degradation.
        optical_noise: Standard deviation of optical camera pixel measurement Gaussian noise.
        duration: Total duration of the simulation event in seconds. Default is 10.0 s.
        sampling_rate: Simulation sampling frequency in Hz (samples per second). Default is 100.0 Hz.
        impact_time: Time in seconds when the impact event occurs. Default is 2.0 s.
        drift_amplitude: Amplitude of slow environmental/thermal structural drift in mm.
        drift_frequency: Frequency of slow environmental drift in Hz (typically < 0.5 Hz).
        camera_jitter: Standard deviation of low-frequency camera fixture vibration/jitter.
        optical_quantization: Discretization step size for optical sensor sub-pixel quantization in mm.
        rotation_coupling: Angular rotation conversion factor in degrees per mm of primary displacement.
        vibration_noise: Standard deviation of accelerometer measurement electronic noise.
    """

    name: str
    natural_frequency: float
    amplitude: float
    damping_ratio: float
    impact_strength: float
    noise_level: float
    frequency_shift: float = 0.0
    optical_noise: float = 0.02
    duration: float = 10.0
    sampling_rate: float = 100.0
    impact_time: float = 2.0
    drift_amplitude: float = 0.15
    drift_frequency: float = 0.1
    camera_jitter: float = 0.015
    optical_quantization: float = 0.005
    rotation_coupling: float = 0.08
    vibration_noise: float = 0.03

    @property
    def effective_frequency(self) -> float:
        """Calculate the effective natural frequency taking frequency shift into account."""
        effective_fn = self.natural_frequency + self.frequency_shift
        if effective_fn <= 0:
            raise ValueError(
                f"Effective natural frequency must be positive, got {effective_fn} Hz "
                f"(base: {self.natural_frequency} Hz, shift: {self.frequency_shift} Hz)"
            )
        return effective_fn

    @property
    def damped_frequency(self) -> float:
        """Calculate the damped natural frequency omega_d = omega_n * sqrt(1 - zeta^2)."""
        if not (0.0 <= self.damping_ratio < 1.0):
            raise ValueError(
                f"Damping ratio must be in [0, 1) for underdamped oscillation, got {self.damping_ratio}"
            )
        import numpy as np

        fn = self.effective_frequency
        return fn * np.sqrt(1.0 - self.damping_ratio**2)

    @property
    def total_samples(self) -> int:
        """Total number of discrete time samples in the simulation."""
        return int(round(self.duration * self.sampling_rate))
