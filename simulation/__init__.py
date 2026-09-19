"""Optical Structural Impact Monitoring - Synthetic Structural Response Simulator.

This module provides physically coherent synthetic generation of synchronized
structural vibration and optical displacement signals under NORMAL, WARNING,
and CRITICAL structural conditions.
"""

from simulation.config import ScenarioConfig
from simulation.scenarios import (
    get_scenario_config,
    create_normal_config,
    create_warning_config,
    create_critical_config,
)
from simulation.vibration_generator import StructuralVibrationGenerator
from simulation.optical_motion_generator import OpticalMotionGenerator
from simulation.simulator import (
    SimulationEvent,
    StructuralSimulator,
    generate_dataset,
)

__all__ = [
    "ScenarioConfig",
    "get_scenario_config",
    "create_normal_config",
    "create_warning_config",
    "create_critical_config",
    "StructuralVibrationGenerator",
    "OpticalMotionGenerator",
    "SimulationEvent",
    "StructuralSimulator",
    "generate_dataset",
]
