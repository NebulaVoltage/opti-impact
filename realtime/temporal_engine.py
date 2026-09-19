"""Temporal decision engine with persistence and hysteresis filtering.

Converts instantaneous single-window ML classifications into a stable structural
state. Rejects isolated transient noisy predictions and requires configurable
consecutive persistent evidence before state transitions are confirmed.

NOTE: Persistence thresholds (e.g. 3 consecutive windows) are software demonstration
parameters designed to show state stabilization, NOT certified civil engineering safety criteria.
"""

from typing import List, Optional, Tuple

from realtime.config import TemporalEngineConfig
from realtime.state_machine import StateTransitionEvent


class TemporalDecisionEngine:
    """Stabilizes instantaneous predictions using temporal hysteresis and persistence."""

    def __init__(self, config: Optional[TemporalEngineConfig] = None, initial_state: str = "NORMAL"):
        """Initialize temporal engine.

        Args:
            config: TemporalEngineConfig instance.
            initial_state: Default starting state.
        """
        self.config = config or TemporalEngineConfig()
        if initial_state not in self.config.valid_states:
            raise ValueError(f"Invalid initial state '{initial_state}'. Expected one of {self.config.valid_states}")

        self._current_state: str = initial_state
        self._candidate_state: Optional[str] = None
        self._candidate_count: int = 0
        self._state_start_time: float = 0.0
        self._transitions: List[StateTransitionEvent] = []

    @property
    def current_state(self) -> str:
        """Current stable structural state."""
        return self._current_state

    @property
    def candidate_state(self) -> Optional[str]:
        """Active candidate state awaiting persistence confirmation."""
        return self._candidate_state

    @property
    def candidate_count(self) -> int:
        """Number of consecutive windows observing candidate_state."""
        return self._candidate_count

    @property
    def transition_history(self) -> List[StateTransitionEvent]:
        """List of all confirmed state transition events."""
        return list(self._transitions)

    def state_duration(self, current_time: float) -> float:
        """Elapsed duration in seconds in the current stable state."""
        return max(0.0, current_time - self._state_start_time)

    def get_persistence_threshold(self, target_state: str) -> int:
        """Retrieve required consecutive window threshold for target state."""
        if target_state == "WARNING":
            return self.config.warning_persistence
        elif target_state == "CRITICAL":
            return self.config.critical_persistence
        elif target_state == "NORMAL":
            return self.config.recovery_persistence
        return 3

    def process_prediction(
        self,
        instant_pred: str,
        timestamp: float,
        window_id: int,
    ) -> Tuple[str, Optional[StateTransitionEvent]]:
        """Evaluate an instantaneous window prediction and update stable system state.

        Args:
            instant_pred: Instantaneous prediction string ('NORMAL', 'WARNING', 'CRITICAL').
            timestamp: Current window timestamp.
            window_id: Current window index.

        Returns:
            Tuple of (current_stable_state, Optional[StateTransitionEvent]).
        """
        if instant_pred not in self.config.valid_states:
            raise ValueError(f"Invalid prediction '{instant_pred}'. Expected one of {self.config.valid_states}")

        transition_event: Optional[StateTransitionEvent] = None

        # Case 1: Prediction agrees with current stable state
        if instant_pred == self._current_state:
            # Cancel any pending candidate state (hysteresis recovery)
            self._candidate_state = None
            self._candidate_count = 0
            return self._current_state, None

        # Case 2: Prediction differs from current stable state
        if instant_pred == self._candidate_state:
            self._candidate_count += 1
        else:
            self._candidate_state = instant_pred
            self._candidate_count = 1

        required_count = self.get_persistence_threshold(self._candidate_state)

        # Confirm state transition upon reaching persistence requirement
        if self._candidate_count >= required_count:
            prev_state = self._current_state
            new_state = self._candidate_state

            transition_event = StateTransitionEvent(
                timestamp=timestamp,
                previous_state=prev_state,
                new_state=new_state,
                reason="persistent_model_prediction",
                consecutive_windows=self._candidate_count,
                window_id=window_id,
            )
            self._transitions.append(transition_event)

            self._current_state = new_state
            self._state_start_time = timestamp
            self._candidate_state = None
            self._candidate_count = 0

        return self._current_state, transition_event

    def reset(self, initial_state: str = "NORMAL") -> None:
        """Reset temporal state machine to clean starting state."""
        self._current_state = initial_state
        self._candidate_state = None
        self._candidate_count = 0
        self._state_start_time = 0.0
        self._transitions.clear()
