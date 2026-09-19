"""Predefined structural condition scenarios and scenario factory.

Defines the physical configurations for NORMAL, WARNING, and CRITICAL
scenarios representing progressive structural degradation.
"""

from typing import Dict
from simulation.config import ScenarioConfig


# Base structural natural frequency reference (synthetic prototype)
BASE_NATURAL_FREQUENCY: float = 5.0  # Hz


def create_normal_config(**overrides) -> ScenarioConfig:
    """Create configuration for healthy structure in normal operation."""
    defaults = dict(
        name="NORMAL",
        natural_frequency=BASE_NATURAL_FREQUENCY,
        amplitude=0.6,
        damping_ratio=0.06,
        impact_strength=1.0,
        noise_level=0.04,
        frequency_shift=0.0,  # 5.0 Hz effective
        optical_noise=0.015,
        duration=10.0,
        sampling_rate=100.0,
        impact_time=2.0,
        drift_amplitude=0.10,
        drift_frequency=0.1,
        camera_jitter=0.01,
        optical_quantization=0.005,
        rotation_coupling=0.06,
        vibration_noise=0.02,
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


def create_warning_config(**overrides) -> ScenarioConfig:
    """Create configuration for structure with moderate stiffness degradation."""
    defaults = dict(
        name="WARNING",
        natural_frequency=BASE_NATURAL_FREQUENCY,
        amplitude=1.8,
        damping_ratio=0.035,
        impact_strength=2.6,
        noise_level=0.07,
        frequency_shift=-0.7,  # 4.3 Hz effective (-14% shift)
        optical_noise=0.02,
        duration=10.0,
        sampling_rate=100.0,
        impact_time=2.0,
        drift_amplitude=0.20,
        drift_frequency=0.1,
        camera_jitter=0.015,
        optical_quantization=0.005,
        rotation_coupling=0.09,
        vibration_noise=0.03,
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


def create_critical_config(**overrides) -> ScenarioConfig:
    """Create configuration for structure with severe degradation and low damping."""
    defaults = dict(
        name="CRITICAL",
        natural_frequency=BASE_NATURAL_FREQUENCY,
        amplitude=4.2,
        damping_ratio=0.012,
        impact_strength=5.5,
        noise_level=0.12,
        frequency_shift=-1.6,  # 3.4 Hz effective (-32% shift)
        optical_noise=0.025,
        duration=10.0,
        sampling_rate=100.0,
        impact_time=2.0,
        drift_amplitude=0.35,
        drift_frequency=0.1,
        camera_jitter=0.02,
        optical_quantization=0.005,
        rotation_coupling=0.14,
        vibration_noise=0.04,
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


SCENARIO_BUILDERS = {
    "NORMAL": create_normal_config,
    "WARNING": create_warning_config,
    "CRITICAL": create_critical_config,
}


def get_scenario_config(scenario_name: str, **overrides) -> ScenarioConfig:
    """Retrieve and customize a scenario configuration by name.

    Args:
        scenario_name: One of 'NORMAL', 'WARNING', 'CRITICAL' (case-insensitive).
        **overrides: Optional attribute overrides for ScenarioConfig.

    Returns:
        Configured ScenarioConfig instance.

    Raises:
        ValueError: If scenario_name is not recognized.
    """
    key = scenario_name.strip().upper()
    if key not in SCENARIO_BUILDERS:
        valid_keys = ", ".join(SCENARIO_BUILDERS.keys())
        raise ValueError(f"Unknown scenario '{scenario_name}'. Valid scenarios are: {valid_keys}")
    return SCENARIO_BUILDERS[key](**overrides)
