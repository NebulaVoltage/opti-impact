"""Hardware-ready sensor stream abstraction and synthetic stream implementation.

Establishes a decoupled multi-channel sensor interface capable of ingesting
synchronized vibration and optical measurements from either synthetic simulation
sources or future physical hardware (Arduino / ESP32 + OpenCV optical tracking).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from simulation.config import ScenarioConfig
from simulation.scenarios import get_scenario_config
from simulation.simulator import SimulationEvent, StructuralSimulator


@dataclass
class SensorChunk:
    """A synchronized slice of multi-channel sensor data.

    Attributes:
        timestamp: 1D array of monotonic timestamp seconds (float64).
        vibration: 1D array of structural vibration / acceleration in m/s^2.
        optical_x: 1D array of optical sensor X displacement in mm.
        optical_y: 1D array of optical sensor Y displacement in mm.
        optical_rotation: 1D array of optical sensor rotation in degrees.
        scenario: Optional 1D array or list of scenario labels (ground truth).
    """

    timestamp: np.ndarray
    vibration: np.ndarray
    optical_x: np.ndarray
    optical_y: np.ndarray
    optical_rotation: np.ndarray
    scenario: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.timestamp)

    def validate(self, sampling_rate: Optional[float] = None) -> None:
        """Validate multi-channel alignment, finite values, and timestamp consistency."""
        n = len(self.timestamp)
        if n == 0:
            raise ValueError("SensorChunk is empty.")

        channels = [
            ("vibration", self.vibration),
            ("optical_x", self.optical_x),
            ("optical_y", self.optical_y),
            ("optical_rotation", self.optical_rotation),
        ]
        for name, arr in channels:
            if len(arr) != n:
                raise ValueError(
                    f"Channel '{name}' length ({len(arr)}) does not match timestamp length ({n})."
                )
            if np.isnan(arr).any():
                raise ValueError(f"Channel '{name}' contains NaN values.")
            if np.isinf(arr).any():
                raise ValueError(f"Channel '{name}' contains Inf values.")

        if self.scenario is not None and len(self.scenario) != n:
            raise ValueError(
                f"Scenario length ({len(self.scenario)}) does not match timestamp length ({n})."
            )

        # Validate monotonic timestamps
        if n > 1:
            diffs = np.diff(self.timestamp)
            if np.any(diffs <= 0):
                raise ValueError("Timestamps are not strictly monotonically increasing.")
            if sampling_rate is not None and sampling_rate > 0:
                expected_dt = 1.0 / sampling_rate
                if not np.allclose(diffs, expected_dt, atol=1e-3, rtol=1e-2):
                    raise ValueError(
                        f"Timestamps do not match expected dt={expected_dt:.4f}s for rate={sampling_rate}Hz."
                    )

    def slice(self, start_idx: int, end_idx: int) -> "SensorChunk":
        """Return a sliced sub-chunk."""
        return SensorChunk(
            timestamp=self.timestamp[start_idx:end_idx],
            vibration=self.vibration[start_idx:end_idx],
            optical_x=self.optical_x[start_idx:end_idx],
            optical_y=self.optical_y[start_idx:end_idx],
            optical_rotation=self.optical_rotation[start_idx:end_idx],
            scenario=self.scenario[start_idx:end_idx] if self.scenario is not None else None,
        )


class SensorStream(ABC):
    """Abstract base class for streaming synchronized sensor data."""

    @abstractmethod
    def read_chunk(self, num_samples: int) -> Optional[SensorChunk]:
        """Read the next chunk of samples from the stream.

        Args:
            num_samples: Number of samples to read.

        Returns:
            SensorChunk if samples available, None if stream exhausted.
        """
        pass

    @abstractmethod
    def has_more(self) -> bool:
        """Check if more samples are available in the stream."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the stream back to the beginning."""
        pass


class SyntheticSensorStream(SensorStream):
    """Concrete sensor stream orchestrating the Step 1 physics-inspired simulator.

    Chains multiple simulated structural condition segments into a continuous,
    seamless multi-channel stream with continuous monotonically increasing timestamps.
    """

    def __init__(
        self,
        scenario_schedule: Sequence[Tuple[str, float]],
        sampling_rate: float = 100.0,
        base_seed: int = 42,
        simulator: Optional[StructuralSimulator] = None,
    ):
        """Initialize synthetic stream.

        Args:
            scenario_schedule: Sequence of (scenario_name, duration_seconds) tuples.
                Example: [("NORMAL", 20.0), ("WARNING", 20.0), ("CRITICAL", 20.0)]
            sampling_rate: Sampling frequency in Hz.
            base_seed: Deterministic seed for reproducible simulation generation.
            simulator: Optional existing StructuralSimulator instance.
        """
        self.scenario_schedule = list(scenario_schedule)
        self.sampling_rate = sampling_rate
        self.base_seed = base_seed
        self.simulator = simulator or StructuralSimulator(seed=base_seed)

        self._full_chunk: Optional[SensorChunk] = None
        self._cursor: int = 0

        self._generate_stream()

    def _generate_stream(self) -> None:
        """Generate and concatenate synchronized simulator segments with continuous time."""
        all_timestamps = []
        all_vibrations = []
        all_opt_x = []
        all_opt_y = []
        all_opt_rot = []
        all_scenarios = []

        current_start_time = 0.0
        dt = 1.0 / self.sampling_rate

        event_duration = 5.0

        for seg_idx, (scen_name, duration) in enumerate(self.scenario_schedule):
            remaining = duration
            sub_idx = 0
            while remaining > 1e-5:
                cur_dur = min(remaining, 5.0)
                sub_seed = self.base_seed + seg_idx * 10007 + sub_idx * 101
                seg_config = get_scenario_config(
                    scen_name,
                    sampling_rate=self.sampling_rate,
                    duration=cur_dur,
                )

                # Generate synchronized event using existing Step 1 simulator
                event: SimulationEvent = self.simulator.generate(seg_config, seed=sub_seed)

                # Offset time to maintain continuous monotonic time axis
                seg_time = current_start_time + np.arange(len(event.time)) * dt

                all_timestamps.append(seg_time)
                all_vibrations.append(event.vibration)
                all_opt_x.append(event.optical_x)
                all_opt_y.append(event.optical_y)
                all_opt_rot.append(event.optical_rotation)
                all_scenarios.append(np.array([scen_name] * len(seg_time)))

                current_start_time = seg_time[-1] + dt
                remaining -= cur_dur
                sub_idx += 1

        self._full_chunk = SensorChunk(
            timestamp=np.concatenate(all_timestamps),
            vibration=np.concatenate(all_vibrations),
            optical_x=np.concatenate(all_opt_x),
            optical_y=np.concatenate(all_opt_y),
            optical_rotation=np.concatenate(all_opt_rot),
            scenario=np.concatenate(all_scenarios),
        )
        self._full_chunk.validate(sampling_rate=self.sampling_rate)
        self._cursor = 0

    def read_chunk(self, num_samples: int) -> Optional[SensorChunk]:
        """Read a slice of up to num_samples from the pre-generated stream."""
        if self._full_chunk is None or self._cursor >= len(self._full_chunk):
            return None

        end_idx = min(self._cursor + num_samples, len(self._full_chunk))
        chunk = self._full_chunk.slice(self._cursor, end_idx)
        self._cursor = end_idx
        return chunk

    def has_more(self) -> bool:
        """Return True if unread samples remain."""
        if self._full_chunk is None:
            return False
        return self._cursor < len(self._full_chunk)

    def reset(self) -> None:
        """Reset cursor to stream start."""
        self._cursor = 0

    @property
    def total_samples(self) -> int:
        """Total number of samples in the stream."""
        return len(self._full_chunk) if self._full_chunk is not None else 0

    @property
    def total_duration(self) -> float:
        """Total simulated duration in seconds."""
        if self._full_chunk is None or len(self._full_chunk) == 0:
            return 0.0
        return float(self._full_chunk.timestamp[-1] - self._full_chunk.timestamp[0])
