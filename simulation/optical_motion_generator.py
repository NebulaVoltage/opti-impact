"""Optical motion generator.

Synthesizes optical displacement measurements from the underlying physical
structural motion, accounting for camera mounting jitter, sub-pixel measurement
noise, and sensor quantization.
"""

from typing import Tuple
import numpy as np
from simulation.config import ScenarioConfig


class OpticalMotionGenerator:
    """Simulates an optical camera sensor observing the target structure."""

    def __init__(self, config: ScenarioConfig, rng: np.random.Generator):
        """Initialize generator with scenario configuration and random generator.

        Args:
            config: ScenarioConfig containing optical and noise parameters.
            rng: NumPy random Generator instance.
        """
        self.config = config
        self.rng = rng

    def _quantize(self, signal_data: np.ndarray, step_size: float) -> np.ndarray:
        """Apply measurement quantization (discretization step size).

        Args:
            signal_data: Continuous physical or sensor signal array.
            step_size: Discretization step size (e.g. sub-pixel resolution limit).

        Returns:
            Discretized signal array.
        """
        if step_size <= 0.0:
            return signal_data
        return np.round(signal_data / step_size) * step_size

    def generate(
        self, time: np.ndarray, true_displacement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate optical_x, optical_y, and optical_rotation from true displacement.

        Args:
            time: 1D NumPy array of time samples.
            true_displacement: 1D NumPy array of underlying physical structural displacement (mm).

        Returns:
            optical_x: 1D array of optical displacement along primary X-axis (mm).
            optical_y: 1D array of optical displacement along transverse Y-axis (mm).
            optical_rotation: 1D array of optical structural rotation (degrees).
        """
        cfg = self.config
        n_samples = len(time)

        # 1. Camera mounting jitter (low-frequency drift and subtle camera mast vibration)
        phi_jx1, phi_jx2 = self.rng.uniform(0, 2 * np.pi, size=2)
        phi_jy1, phi_jy2 = self.rng.uniform(0, 2 * np.pi, size=2)

        camera_jitter_x = cfg.camera_jitter * (
            0.7 * np.sin(2 * np.pi * 0.25 * time + phi_jx1)
            + 0.3 * np.cos(2 * np.pi * 0.85 * time + phi_jx2)
        )
        camera_jitter_y = cfg.camera_jitter * (
            0.7 * np.sin(2 * np.pi * 0.20 * time + phi_jy1)
            + 0.3 * np.cos(2 * np.pi * 0.75 * time + phi_jy2)
        )

        # 2. Pixel measurement noise (Gaussian white noise representing sub-pixel localization error)
        pixel_noise_x = self.rng.normal(0.0, cfg.optical_noise, size=n_samples)
        pixel_noise_y = self.rng.normal(0.0, cfg.optical_noise, size=n_samples)

        # 3. Transverse structural motion (cross-axis dynamic coupling + Poisson expansion)
        # Transverse displacement is coupled to primary displacement plus transverse drift
        transverse_coupling = 0.12
        transverse_drift = 0.05 * np.cos(2 * np.pi * 0.12 * time + phi_jy1)
        true_displacement_y = (transverse_coupling * true_displacement) + transverse_drift

        # 4. Structural rotation (bending slope theta proportional to primary displacement)
        # Unit: degrees
        true_rotation = cfg.rotation_coupling * true_displacement
        rot_noise = self.rng.normal(0.0, cfg.optical_noise * 0.5, size=n_samples)
        rot_jitter = (cfg.camera_jitter * 0.4) * np.sin(2 * np.pi * 0.3 * time + phi_jx2)

        # Combine physical motion + camera artifacts
        raw_optical_x = true_displacement + camera_jitter_x + pixel_noise_x
        raw_optical_y = true_displacement_y + camera_jitter_y + pixel_noise_y
        raw_optical_rot = true_rotation + rot_jitter + rot_noise

        # 5. Apply sensor quantization (sub-pixel grid quantization)
        optical_x = self._quantize(raw_optical_x, cfg.optical_quantization)
        optical_y = self._quantize(raw_optical_y, cfg.optical_quantization)
        optical_rotation = self._quantize(raw_optical_rot, cfg.optical_quantization * 0.5)

        return optical_x, optical_y, optical_rotation
