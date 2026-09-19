"""Vibration and underlying structural motion generator.

Generates physically coherent structural response models:
- Underlying structural displacement: damped transient response + ambient vibration + low-frequency drift
- Synchronized vibration (acceleration) signal: dynamic acceleration + impact spike + environmental & sensor noise
"""

from typing import Tuple
import numpy as np
from scipy import signal
from simulation.config import ScenarioConfig


class StructuralVibrationGenerator:
    """Generates underlying physical structural displacement and coupled vibration signals."""

    def __init__(self, config: ScenarioConfig, rng: np.random.Generator):
        """Initialize generator with configuration and reproducible random generator.

        Args:
            config: ScenarioConfig defining physical and sensor parameters.
            rng: NumPy random Generator instance.
        """
        self.config = config
        self.rng = rng

    def generate_time_vector(self) -> np.ndarray:
        """Generate uniformly spaced time vector [0, duration) with total_samples points."""
        dt = 1.0 / self.config.sampling_rate
        return np.arange(self.config.total_samples, dtype=np.float64) * dt

    def generate_ambient_displacement(self, time: np.ndarray) -> np.ndarray:
        """Generate ambient continuous structural displacement.

        Combines:
        1. Low-frequency structural movement (slow thermal/wind drift).
        2. Resonant ambient structural oscillation (bandpass-filtered random excitation
           around the effective natural frequency).
        """
        cfg = self.config
        dt = 1.0 / cfg.sampling_rate

        # 1. Slow environmental drift (e.g. 0.1 Hz primary + minor harmonic)
        phase_drift1 = self.rng.uniform(0, 2 * np.pi)
        phase_drift2 = self.rng.uniform(0, 2 * np.pi)
        drift = (
            cfg.drift_amplitude * np.sin(2 * np.pi * cfg.drift_frequency * time + phase_drift1)
            + 0.3 * cfg.drift_amplitude * np.cos(2 * np.pi * 1.7 * cfg.drift_frequency * time + phase_drift2)
        )

        # 2. Continuous ambient structural resonance
        # Model background force excitation filtered by a 2nd-order resonator
        fn = cfg.effective_frequency
        wn = 2.0 * np.pi * fn
        zeta = cfg.damping_ratio

        # Continuous SDOF system: H(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
        # Discretize via bilinear transform to get stable digital filter
        b, a = signal.bilinear([wn**2], [1.0, 2.0 * zeta * wn, wn**2], fs=cfg.sampling_rate)

        # Broadband Gaussian ambient driving force
        raw_ambient_force = self.rng.normal(0.0, cfg.noise_level * cfg.amplitude, size=len(time))
        # Initial transient warm-up to avoid zero-start filter artifact
        warmup_samples = min(200, len(time))
        warmup_force = self.rng.normal(0.0, cfg.noise_level * cfg.amplitude, size=warmup_samples)
        full_force = np.concatenate([warmup_force, raw_ambient_force])
        filtered_resp = signal.lfilter(b, a, full_force)[warmup_samples:]

        return drift + filtered_resp

    def generate_impact_displacement(self, time: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate damped structural oscillatory response to impact impulse.

        The transient displacement follows:
            x_impact(t) = A_impact * exp(-zeta * omega_n * t_rel) * sin(omega_d * t_rel)
        for t >= t_impact, and 0 for t < t_impact.

        Also computes the corresponding analytical dynamic acceleration:
            a_impact(t) = d^2 x_impact / dt^2

        Returns:
            Tuple of (displacement_transient, acceleration_transient) in mm and m/s^2 respectively.
        """
        cfg = self.config
        fn = cfg.effective_frequency
        wn = 2.0 * np.pi * fn
        zeta = cfg.damping_ratio
        wd = 2.0 * np.pi * cfg.damped_frequency
        gamma = zeta * wn

        # Transient amplitude scaled by baseline amplitude and impact strength
        # Unit: mm
        a_impact = cfg.amplitude * cfg.impact_strength

        # Random phase jitter on impact response
        phase = self.rng.uniform(-0.1, 0.1)

        disp_transient = np.zeros_like(time)
        accel_transient = np.zeros_like(time)

        # Impact index
        impact_idx = np.searchsorted(time, cfg.impact_time)
        if impact_idx < len(time):
            t_rel = time[impact_idx:] - cfg.impact_time
            envelope = np.exp(-gamma * t_rel)
            sin_term = np.sin(wd * t_rel + phase)
            cos_term = np.cos(wd * t_rel + phase)

            # True displacement (mm)
            disp_transient[impact_idx:] = a_impact * envelope * sin_term

            # Second derivative d^2(x)/dt^2
            # d/dt [exp(-gamma*t)*sin(wd*t+phi)] = exp(-gamma*t)*[-gamma*sin + wd*cos]
            # d^2/dt^2 [...] = exp(-gamma*t)*[(gamma^2 - wd^2)*sin - 2*gamma*wd*cos]
            # Convert mm to m for acceleration: 1 mm = 1e-3 m
            scale_to_m = 1e-3
            accel_transient[impact_idx:] = (
                a_impact
                * scale_to_m
                * envelope
                * ((gamma**2 - wd**2) * sin_term - 2.0 * gamma * wd * cos_term)
            )

            # Impact contact impulse spike at impact instant (short mechanical contact pulse)
            # Duration ~ 2-3 samples representing the immediate hammer/collision shock
            impulse_width_samples = max(2, int(0.03 * cfg.sampling_rate))
            idx_end = min(len(time), impact_idx + impulse_width_samples)
            impulse_shape = np.hanning(2 * (idx_end - impact_idx))[-(idx_end - impact_idx):]
            # Shock acceleration spike in m/s^2
            shock_magnitude = cfg.impact_strength * (wn / (2 * np.pi)) * 1.5
            accel_transient[impact_idx:idx_end] += shock_magnitude * impulse_shape

        return disp_transient, accel_transient

    def generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synchronized time, true displacement, and vibration signals.

        Returns:
            time: 1D NumPy array of timestamps (s).
            true_displacement: 1D NumPy array of primary structural displacement (mm).
            vibration: 1D NumPy array of structural vibration / acceleration (m/s^2).
        """
        cfg = self.config
        time = self.generate_time_vector()

        # Generate ambient motion
        ambient_disp = self.generate_ambient_displacement(time)

        # Generate impact transient
        impact_disp, impact_accel = self.generate_impact_displacement(time)

        # True structural displacement is the combined physical motion
        true_displacement = ambient_disp + impact_disp

        # Physical acceleration of ambient component: numerical 2nd derivative with smoothing
        # Using central differences for ambient acceleration
        dt = 1.0 / cfg.sampling_rate
        ambient_disp_m = ambient_disp * 1e-3
        ambient_accel = np.gradient(np.gradient(ambient_disp_m, dt), dt)

        # Total dynamic acceleration
        total_accel = ambient_accel + impact_accel

        # Environmental high-frequency vibration (ambient machinery / wind buffeting)
        env_freq1 = 2.4 * cfg.effective_frequency
        env_freq2 = 4.1 * cfg.effective_frequency
        phase1, phase2 = self.rng.uniform(0, 2 * np.pi, size=2)
        env_vibration = (
            0.15 * cfg.noise_level * np.sin(2 * np.pi * env_freq1 * time + phase1)
            + 0.10 * cfg.noise_level * np.cos(2 * np.pi * env_freq2 * time + phase2)
        )

        # Measurement sensor electronic noise (accelerometer noise)
        sensor_noise = self.rng.normal(0.0, cfg.vibration_noise, size=len(time))

        # Vibration signal (m/s^2)
        vibration = total_accel + env_vibration + sensor_noise

        return time, true_displacement, vibration
