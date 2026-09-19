"""Main simulator module orchestrating vibration and optical motion generators.

Provides:
- SimulationEvent dataclass representing synchronized simulation results.
- StructuralSimulator class providing the public generation and dataset export API.
- generate_dataset utility function.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd

from simulation.config import ScenarioConfig
from simulation.scenarios import get_scenario_config
from simulation.vibration_generator import StructuralVibrationGenerator
from simulation.optical_motion_generator import OpticalMotionGenerator


@dataclass
class SimulationEvent:
    """Represents a synchronized structural response simulation event.

    Attributes:
        time: Timestamp array in seconds (float64).
        true_displacement: Underlying primary structural displacement in mm (float64).
        vibration: Measured structural vibration / acceleration in m/s^2 (float64).
        optical_x: Optical sensor measured displacement along X-axis in mm (float64).
        optical_y: Optical sensor measured displacement along Y-axis in mm (float64).
        optical_rotation: Optical sensor measured rotation in degrees (float64).
        scenario: Scenario name string (e.g. 'NORMAL', 'WARNING', 'CRITICAL').
        config: The ScenarioConfig instance used to produce this event.
    """

    time: np.ndarray
    true_displacement: np.ndarray
    vibration: np.ndarray
    optical_x: np.ndarray
    optical_y: np.ndarray
    optical_rotation: np.ndarray
    scenario: str
    config: ScenarioConfig

    def to_dataframe(self) -> pd.DataFrame:
        """Convert simulation signals into a standardized pandas DataFrame.

        Columns:
            timestamp, true_displacement, vibration, optical_x, optical_y,
            optical_rotation, scenario
        """
        return pd.DataFrame(
            {
                "timestamp": self.time,
                "true_displacement": self.true_displacement,
                "vibration": self.vibration,
                "optical_x": self.optical_x,
                "optical_y": self.optical_y,
                "optical_rotation": self.optical_rotation,
                "scenario": self.scenario,
            }
        )

    def to_csv(self, filepath: Union[str, Path]) -> Path:
        """Save event data to a CSV file.

        Args:
            filepath: Destination file path.

        Returns:
            Resolved Path of the written file.
        """
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(out_path, index=False, float_format="%.6f")
        return out_path


class StructuralSimulator:
    """Synchronized structural vibration and optical displacement simulator."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize simulator.

        Args:
            seed: Optional integer random seed for reproducibility.
        """
        self.default_seed = seed
        self._rng = np.random.default_rng(seed)

    def generate(
        self,
        scenario: Union[str, ScenarioConfig],
        seed: Optional[int] = None,
        **overrides,
    ) -> SimulationEvent:
        """Generate a synchronized simulation event for the given scenario.

        Args:
            scenario: Either a scenario name ('NORMAL', 'WARNING', 'CRITICAL')
                      or a custom ScenarioConfig instance.
            seed: Optional seed for this specific run. If None, uses internal RNG.
            **overrides: Optional overrides if scenario is specified by name.

        Returns:
            SimulationEvent containing synchronized signals.
        """
        if isinstance(scenario, ScenarioConfig):
            config = scenario
        elif isinstance(scenario, str):
            config = get_scenario_config(scenario, **overrides)
        else:
            raise TypeError(
                f"scenario must be str or ScenarioConfig, got {type(scenario).__name__}"
            )

        rng = np.random.default_rng(seed) if seed is not None else self._rng

        # 1. Generate underlying physical motion and vibration
        vib_gen = StructuralVibrationGenerator(config, rng)
        time, true_displacement, vibration = vib_gen.generate()

        # 2. Generate synchronized optical measurements
        opt_gen = OpticalMotionGenerator(config, rng)
        optical_x, optical_y, optical_rotation = opt_gen.generate(time, true_displacement)

        return SimulationEvent(
            time=time,
            true_displacement=true_displacement,
            vibration=vibration,
            optical_x=optical_x,
            optical_y=optical_y,
            optical_rotation=optical_rotation,
            scenario=config.name,
            config=config,
        )

    def generate_dataset(
        self,
        scenario: str,
        count: int,
        output_dir: Union[str, Path] = "dataset/raw",
        base_seed: int = 42,
    ) -> List[Path]:
        """Generate multiple simulation events and save them as CSV files.

        Args:
            scenario: Scenario name ('NORMAL', 'WARNING', 'CRITICAL').
            count: Number of events to generate.
            output_dir: Directory where CSV files will be stored.
            base_seed: Base integer seed for reproducibility across the batch.

        Returns:
            List of generated CSV file paths.
        """
        return generate_dataset(
            scenario=scenario,
            count=count,
            output_dir=output_dir,
            base_seed=base_seed,
            simulator=self,
        )


def generate_dataset(
    scenario: str,
    count: int,
    output_dir: Union[str, Path] = "dataset/raw",
    base_seed: int = 42,
    simulator: Optional[StructuralSimulator] = None,
) -> List[Path]:
    """Generate a batch of simulation events and save as CSV files.

    Example filenames:
        normal_event_001.csv
        warning_event_001.csv
        critical_event_001.csv

    Args:
        scenario: Scenario name ('NORMAL', 'WARNING', 'CRITICAL').
        count: Number of events to generate.
        output_dir: Output directory path.
        base_seed: Base random seed.
        simulator: Optional existing StructuralSimulator instance.

    Returns:
        List of generated CSV file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = simulator or StructuralSimulator()
    prefix = scenario.strip().lower()
    created_files: List[Path] = []

    for i in range(1, count + 1):
        # Derive distinct reproducible seed per run
        run_seed = base_seed + i if base_seed is not None else None
        event = sim.generate(scenario, seed=run_seed)
        filename = f"{prefix}_event_{i:03d}.csv"
        file_path = out_dir / filename
        event.to_csv(file_path)
        created_files.append(file_path)

    return created_files
