"""Controlled physical parameter variation system for synthetic dataset generation.

Defines realistic parameter distribution intervals around nominal scenario configurations.
Deliberately introduces physical boundary overlap between classes (e.g. strong normal
impacts overlapping with mild warning events, and frequency distribution transitions)
so that classification requires holistic multimodal feature integration rather than
any single trivial feature threshold.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np

from simulation.config import ScenarioConfig


@dataclass
class ParameterRange:
    """Represents a continuous uniform parameter sampling interval [low, high]."""

    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        """Sample a single value from the uniform distribution."""
        return float(rng.uniform(self.low, self.high))


@dataclass
class ScenarioVariationConfig:
    """Configurable parameter variation ranges for a specific structural scenario."""

    scenario_name: str
    frequency_range: ParameterRange
    amplitude_range: ParameterRange
    damping_range: ParameterRange
    impact_strength_range: ParameterRange
    impact_time_range: ParameterRange
    noise_level_range: ParameterRange
    optical_noise_range: ParameterRange
    camera_jitter_range: ParameterRange
    vibration_noise_range: ParameterRange
    drift_amplitude_range: ParameterRange
    duration: float = 10.0
    sampling_rate: float = 100.0

    def sample_config(self, rng: np.random.Generator) -> ScenarioConfig:
        """Sample an independent, physically coherent ScenarioConfig instance."""
        fn = self.frequency_range.sample(rng)
        amp = self.amplitude_range.sample(rng)
        zeta = self.damping_range.sample(rng)
        imp_str = self.impact_strength_range.sample(rng)
        imp_time = self.impact_time_range.sample(rng)
        noise = self.noise_level_range.sample(rng)
        opt_noise = self.optical_noise_range.sample(rng)
        cam_jitter = self.camera_jitter_range.sample(rng)
        vib_noise = self.vibration_noise_range.sample(rng)
        drift_amp = self.drift_amplitude_range.sample(rng)

        # Baseline reference is 5.0 Hz; frequency shift is fn - 5.0
        base_fn = 5.0
        freq_shift = fn - base_fn

        return ScenarioConfig(
            name=self.scenario_name,
            natural_frequency=base_fn,
            amplitude=amp,
            damping_ratio=zeta,
            impact_strength=imp_str,
            noise_level=noise,
            frequency_shift=freq_shift,
            optical_noise=opt_noise,
            duration=self.duration,
            sampling_rate=self.sampling_rate,
            impact_time=imp_time,
            drift_amplitude=drift_amp,
            camera_jitter=cam_jitter,
            vibration_noise=vib_noise,
        )


# Default parameter variation ranges designed with controlled physical boundary overlap
DEFAULT_VARIATION_PRESETS: Dict[str, ScenarioVariationConfig] = {
    "NORMAL": ScenarioVariationConfig(
        scenario_name="NORMAL",
        frequency_range=ParameterRange(4.65, 5.35),       # Nominal ~5.0 Hz
        amplitude_range=ParameterRange(0.35, 0.95),       # Moderate overlap on high end
        damping_range=ParameterRange(0.045, 0.075),       # High damping (fast decay)
        impact_strength_range=ParameterRange(0.5, 1.8),   # Occasionally heavier normal impact
        impact_time_range=ParameterRange(1.5, 3.5),
        noise_level_range=ParameterRange(0.025, 0.065),
        optical_noise_range=ParameterRange(0.010, 0.022),
        camera_jitter_range=ParameterRange(0.008, 0.016),
        vibration_noise_range=ParameterRange(0.015, 0.030),
        drift_amplitude_range=ParameterRange(0.05, 0.15),
    ),
    "WARNING": ScenarioVariationConfig(
        scenario_name="WARNING",
        frequency_range=ParameterRange(3.95, 4.75),       # Overlaps NORMAL [4.65 - 4.75]
        amplitude_range=ParameterRange(1.0, 2.7),         # Overlaps strong NORMAL and mild CRITICAL
        damping_range=ParameterRange(0.022, 0.048),       # Moderate damping
        impact_strength_range=ParameterRange(1.6, 3.8),
        impact_time_range=ParameterRange(1.5, 3.5),
        noise_level_range=ParameterRange(0.045, 0.100),
        optical_noise_range=ParameterRange(0.014, 0.026),
        camera_jitter_range=ParameterRange(0.010, 0.020),
        vibration_noise_range=ParameterRange(0.020, 0.040),
        drift_amplitude_range=ParameterRange(0.12, 0.28),
    ),
    "CRITICAL": ScenarioVariationConfig(
        scenario_name="CRITICAL",
        frequency_range=ParameterRange(3.05, 4.05),       # Overlaps WARNING [3.95 - 4.05]
        amplitude_range=ParameterRange(2.8, 5.8),         # High deflection
        damping_range=ParameterRange(0.007, 0.022),       # Very low damping (lingering ringing)
        impact_strength_range=ParameterRange(3.8, 7.2),
        impact_time_range=ParameterRange(1.5, 3.5),
        noise_level_range=ParameterRange(0.08, 0.160),
        optical_noise_range=ParameterRange(0.018, 0.032),
        camera_jitter_range=ParameterRange(0.015, 0.028),
        vibration_noise_range=ParameterRange(0.025, 0.055),
        drift_amplitude_range=ParameterRange(0.20, 0.45),
    ),
}


def get_variation_config(
    scenario_name: str,
    presets: Dict[str, ScenarioVariationConfig] = DEFAULT_VARIATION_PRESETS,
) -> ScenarioVariationConfig:
    """Retrieve the variation configuration for a given scenario.

    Args:
        scenario_name: 'NORMAL', 'WARNING', or 'CRITICAL'.
        presets: Optional dictionary of presets.

    Returns:
        ScenarioVariationConfig instance.
    """
    key = scenario_name.strip().upper()
    if key not in presets:
        raise ValueError(
            f"Unknown scenario '{scenario_name}'. Available presets: {list(presets.keys())}"
        )
    return presets[key]
