"""Planar optical distance calibration module.

Enables approximate planar conversion from image pixels to physical millimeters:
    scale_mm_per_pixel = known_physical_distance_mm / measured_image_distance_pixels

NOTE: When uncalibrated, the system operates purely in pixel space and explicitly
reports 'METRIC CALIBRATION: NOT ACTIVE' to avoid fabricating physical accuracy.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple


class PlanarCalibration:
    """Manages planar distance scaling calibration between pixels and millimeters."""

    def __init__(self, scale_mm_per_pixel: Optional[float] = None):
        """Initialize calibration.

        Args:
            scale_mm_per_pixel: Ratio of millimeters per image pixel.
        """
        self.scale_mm_per_pixel = scale_mm_per_pixel
        self.is_calibrated: bool = (
            scale_mm_per_pixel is not None and scale_mm_per_pixel > 0.0
        )

    def calibrate_from_distance(
        self,
        known_distance_mm: float,
        measured_distance_pixels: float,
    ) -> float:
        """Calibrate planar scale from a known physical reference dimension.

        Args:
            known_distance_mm: Ground truth physical distance in mm (e.g. 50 mm).
            measured_distance_pixels: Corresponding pixel span measured on image plane.

        Returns:
            Computed scale factor in mm/pixel.
        """
        if known_distance_mm <= 0:
            raise ValueError(f"known_distance_mm must be positive, got {known_distance_mm}")
        if measured_distance_pixels <= 0:
            raise ValueError(f"measured_distance_pixels must be positive, got {measured_distance_pixels}")

        self.scale_mm_per_pixel = float(known_distance_mm / measured_distance_pixels)
        self.is_calibrated = True
        return self.scale_mm_per_pixel

    def pixels_to_mm(self, pixels: float) -> Optional[float]:
        """Convert pixel measurement to physical millimeters if calibrated."""
        if not self.is_calibrated or self.scale_mm_per_pixel is None:
            return None
        return float(pixels * self.scale_mm_per_pixel)

    def mm_to_pixels(self, mm: float) -> Optional[float]:
        """Convert millimeters to image pixels if calibrated."""
        if not self.is_calibrated or self.scale_mm_per_pixel is None or self.scale_mm_per_pixel == 0:
            return None
        return float(mm / self.scale_mm_per_pixel)

    def save(self, filepath: Path) -> None:
        """Save calibration scale to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "is_calibrated": self.is_calibrated,
            "scale_mm_per_pixel": self.scale_mm_per_pixel,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "PlanarCalibration":
        """Load calibration scale from a JSON file."""
        filepath = Path(filepath)
        if not filepath.exists():
            return cls(scale_mm_per_pixel=None)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        scale = data.get("scale_mm_per_pixel")
        return cls(scale_mm_per_pixel=scale)
