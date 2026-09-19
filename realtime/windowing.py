"""Streaming window buffer manager for real-time multimodal time-series processing."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from realtime.config import StreamConfig
from realtime.data_stream import SensorChunk


@dataclass
class SensorWindow:
    """A fixed-duration synchronized window of multi-channel sensor measurements.

    Attributes:
        window_id: Integer index of this window in the session.
        start_time: Starting timestamp of the window (seconds).
        end_time: Ending timestamp of the window (seconds).
        vibration: 1D array of vibration/acceleration samples (m/s^2).
        optical_x: 1D array of optical X coordinates (mm).
        optical_y: 1D array of optical Y coordinates (mm).
        optical_rotation: 1D array of optical rotation values (degrees).
        timestamps: 1D array of timestamp points.
        ground_truth_scenario: Majority/dominant scenario within this window.
        is_transition_window: True if window spans multiple scenarios across boundary.
        scenario_proportions: Proportions of each scenario present in window samples.
    """

    window_id: int
    start_time: float
    end_time: float
    vibration: np.ndarray
    optical_x: np.ndarray
    optical_y: np.ndarray
    optical_rotation: np.ndarray
    timestamps: np.ndarray
    ground_truth_scenario: str
    is_transition_window: bool
    scenario_proportions: Dict[str, float]

    def __len__(self) -> int:
        return len(self.timestamps)


class StreamingWindowManager:
    """Sliding-window buffer managing multi-channel sensor streaming samples."""

    def __init__(
        self,
        config: Optional[StreamConfig] = None,
        window_size: Optional[int] = None,
        step_size: Optional[int] = None,
    ):
        """Initialize window manager.

        Args:
            config: StreamConfig instance defining window_size and step_size.
            window_size: Optional direct window size override.
            step_size: Optional direct step size override.
        """
        cfg = config or StreamConfig()
        self.window_size = window_size or cfg.window_size
        self.step_size = step_size or cfg.step_size

        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if self.step_size <= 0:
            raise ValueError(f"step_size must be positive, got {self.step_size}")

        # Internal buffers
        self._timestamps: List[float] = []
        self._vibration: List[float] = []
        self._optical_x: List[float] = []
        self._optical_y: List[float] = []
        self._optical_rotation: List[float] = []
        self._scenario: List[str] = []

        self._window_counter: int = 0

    def add_chunk(self, chunk: SensorChunk) -> None:
        """Append a new sensor chunk to the sliding buffer.

        Args:
            chunk: SensorChunk containing new synchronized samples.
        """
        chunk.validate()
        self._timestamps.extend(chunk.timestamp.tolist())
        self._vibration.extend(chunk.vibration.tolist())
        self._optical_x.extend(chunk.optical_x.tolist())
        self._optical_y.extend(chunk.optical_y.tolist())
        self._optical_rotation.extend(chunk.optical_rotation.tolist())
        if chunk.scenario is not None:
            self._scenario.extend(chunk.scenario.tolist())
        else:
            self._scenario.extend(["UNKNOWN"] * len(chunk.timestamp))

    def has_window(self) -> bool:
        """Return True if enough samples are buffered to form a full window."""
        return len(self._timestamps) >= self.window_size

    def get_next_window(self) -> Optional[SensorWindow]:
        """Extract the next complete window and advance the buffer by step_size.

        Returns:
            SensorWindow if a full window is available, otherwise None.
        """
        if not self.has_window():
            return None

        w_size = self.window_size

        t_slice = np.array(self._timestamps[:w_size], dtype=np.float64)
        v_slice = np.array(self._vibration[:w_size], dtype=np.float64)
        ox_slice = np.array(self._optical_x[:w_size], dtype=np.float64)
        oy_slice = np.array(self._optical_y[:w_size], dtype=np.float64)
        or_slice = np.array(self._optical_rotation[:w_size], dtype=np.float64)
        scen_slice = self._scenario[:w_size]

        # Analyze scenario composition
        unique, counts = np.unique(scen_slice, return_counts=True)
        props = {str(u): float(c / w_size) for u, c in zip(unique, counts)}
        dominant_scenario = str(unique[np.argmax(counts)])
        is_transition = len(unique) > 1

        window = SensorWindow(
            window_id=self._window_counter,
            start_time=float(t_slice[0]),
            end_time=float(t_slice[-1]),
            vibration=v_slice,
            optical_x=ox_slice,
            optical_y=oy_slice,
            optical_rotation=or_slice,
            timestamps=t_slice,
            ground_truth_scenario=dominant_scenario,
            is_transition_window=is_transition,
            scenario_proportions=props,
        )

        self._window_counter += 1

        # Advance buffer by step_size
        step = self.step_size
        del self._timestamps[:step]
        del self._vibration[:step]
        del self._optical_x[:step]
        del self._optical_y[:step]
        del self._optical_rotation[:step]
        del self._scenario[:step]

        return window

    def clear(self) -> None:
        """Reset and empty all internal buffers."""
        self._timestamps.clear()
        self._vibration.clear()
        self._optical_x.clear()
        self._optical_y.clear()
        self._optical_rotation.clear()
        self._scenario.clear()
        self._window_counter = 0

    @property
    def buffered_samples(self) -> int:
        """Number of samples currently buffered."""
        return len(self._timestamps)
