"""Session recording module exporting frame-by-frame optical telemetry to CSV."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from optical.optical_motion import OpticalKinematics


class OpticalSessionRecorder:
    """Records real-time optical tracking telemetry to structured CSV logs."""

    COLUMNS: List[str] = [
        "timestamp",
        "frame_index",
        "structural_dx_pixels",
        "structural_dy_pixels",
        "structural_displacement_pixels",
        "velocity_x_pixels_s",
        "velocity_y_pixels_s",
        "velocity_magnitude_pixels_s",
        "acceleration_x_pixels_s2",
        "acceleration_y_pixels_s2",
        "acceleration_magnitude_pixels_s2",
        "optical_dominant_frequency_hz",
        "valid_feature_count",
        "tracking_quality",
        "camera_fps",
    ]

    def __init__(
        self,
        output_dir: Path = Path("outputs/optical"),
        session_name: Optional[str] = None,
    ):
        """Initialize session recorder.

        Args:
            output_dir: Directory where session CSV will be stored.
            session_name: Optional explicit file basename without extension.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if session_name is None:
            ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            session_name = f"optical_session_{ts_str}"

        self.session_name = session_name
        self.filepath = self.output_dir / f"{session_name}.csv"
        self._records: List[Dict[str, Any]] = []

    def record_frame(
        self,
        kinematics: OpticalKinematics,
        camera_fps: float,
    ) -> None:
        """Record an instantaneous frame measurement."""
        row = {
            "timestamp": float(kinematics.timestamp),
            "frame_index": int(kinematics.frame_index),
            "structural_dx_pixels": float(kinematics.dx_pixels),
            "structural_dy_pixels": float(kinematics.dy_pixels),
            "structural_displacement_pixels": float(kinematics.cum_displacement_pixels),
            "velocity_x_pixels_s": float(kinematics.velocity_x_pixels_s),
            "velocity_y_pixels_s": float(kinematics.velocity_y_pixels_s),
            "velocity_magnitude_pixels_s": float(kinematics.velocity_magnitude_pixels_s),
            "acceleration_x_pixels_s2": float(kinematics.acceleration_x_pixels_s2),
            "acceleration_y_pixels_s2": float(kinematics.acceleration_y_pixels_s2),
            "acceleration_magnitude_pixels_s2": float(kinematics.acceleration_magnitude_pixels_s2),
            "optical_dominant_frequency_hz": (
                float(kinematics.dominant_frequency_hz)
                if not np.isnan(kinematics.dominant_frequency_hz)
                else np.nan
            ),
            "valid_feature_count": int(kinematics.valid_feature_count),
            "tracking_quality": str(kinematics.tracking_quality),
            "camera_fps": float(camera_fps),
        }
        self._records.append(row)

    def save(self) -> Path:
        """Flush and save recorded rows to CSV."""
        if not self._records:
            df = pd.DataFrame(columns=self.COLUMNS)
        else:
            df = pd.DataFrame(self._records)[self.COLUMNS]

        df.to_csv(self.filepath, index=False, float_format="%.4f")
        return self.filepath

    @property
    def record_count(self) -> int:
        """Number of recorded frames in current session."""
        return len(self._records)
