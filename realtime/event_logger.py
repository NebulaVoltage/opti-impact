"""Persistent logging for inference windows, state transitions, and performance summaries."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from realtime.state_machine import StateTransitionEvent
from realtime.telemetry import InferenceTelemetry


class EventLogger:
    """Manages writing telemetry to CSV, transition events to JSONL, and session summaries."""

    def __init__(
        self,
        output_dir: Path = Path("outputs/step5"),
        csv_filename: str = "inference_log.csv",
        jsonl_filename: str = "state_events.jsonl",
        summary_filename: str = "realtime_summary.json",
    ):
        """Initialize logger with output directory and file paths."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.output_dir / csv_filename
        self.jsonl_path = self.output_dir / jsonl_filename
        self.summary_path = self.output_dir / summary_filename

        self._telemetry_records: List[InferenceTelemetry] = []
        self._transition_events: List[StateTransitionEvent] = []

        # Clear/initialize files
        if self.jsonl_path.exists():
            self.jsonl_path.unlink()

    def log_telemetry(self, telemetry: InferenceTelemetry) -> None:
        """Store in-memory telemetry record."""
        self._telemetry_records.append(telemetry)

    def log_transition(self, event: StateTransitionEvent) -> None:
        """Append a state transition event as a JSON line."""
        self._transition_events.append(event)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    def save_inference_csv(self) -> Path:
        """Export all recorded telemetry rows to CSV."""
        if not self._telemetry_records:
            # Write empty dataframe with columns
            df = pd.DataFrame()
        else:
            rows = [t.to_dict() for t in self._telemetry_records]
            df = pd.DataFrame(rows)

        df.to_csv(self.csv_path, index=False, float_format="%.4f")
        return self.csv_path

    def save_summary(self, total_runtime: float, final_state: str) -> Dict[str, Any]:
        """Compute latency percentiles and serialize summary JSON."""
        latencies = [t.total_latency_ms for t in self._telemetry_records]
        if latencies:
            mean_lat = float(np.mean(latencies))
            median_lat = float(np.median(latencies))
            p95_lat = float(np.percentile(latencies, 95))
            max_lat = float(np.max(latencies))
        else:
            mean_lat = median_lat = p95_lat = max_lat = 0.0

        summary: Dict[str, Any] = {
            "number_of_windows": len(self._telemetry_records),
            "number_of_state_transitions": len(self._transition_events),
            "total_runtime": float(total_runtime),
            "mean_latency": mean_lat,
            "median_latency": median_lat,
            "p95_latency": p95_lat,
            "max_latency": max_lat,
            "final_state": str(final_state),
        }

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
