"""State definitions, transition events, and structural condition state machine."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class StateTransitionEvent:
    """Represents a validated structural state transition event.

    Attributes:
        timestamp: Timestamp of the transition in seconds.
        previous_state: Previous stable structural state ('NORMAL', 'WARNING', 'CRITICAL').
        new_state: New stable structural state ('NORMAL', 'WARNING', 'CRITICAL').
        reason: Cause of transition (e.g. 'persistent_model_prediction').
        consecutive_windows: Number of consecutive agreeing windows that confirmed the transition.
        window_id: Index of the window that triggered the transition.
    """

    timestamp: float
    previous_state: str
    new_state: str
    reason: str
    consecutive_windows: int
    window_id: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert transition event to standard dictionary."""
        return asdict(self)
