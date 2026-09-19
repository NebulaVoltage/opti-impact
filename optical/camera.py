"""Camera capture management and device discovery module.

Provides:
  - CameraConfig: configuration for camera index, resolution, and FPS.
  - CameraInfo: metadata for discovered video devices.
  - list_available_cameras: device discovery probing indices and measuring actual FPS.
  - CameraManager: robust frame acquisition with actual FPS tracking and fallback.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Deque, List, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class CameraConfig:
    """Camera hardware and capture configuration."""

    camera_index: int = 1
    width: int = 1280
    height: int = 720
    requested_fps: float = 30.0
    exposure: Optional[float] = None
    autofocus: Optional[bool] = None
    buffer_size: int = 1


@dataclass
class CameraInfo:
    """Hardware discovery status and properties for a video capture device."""

    index: int
    is_available: bool
    width: int
    height: int
    actual_fps: float
    backend: str


def list_available_cameras(max_indices: int = 4, test_frames: int = 8) -> List[CameraInfo]:
    """Scan video device indices to discover connected cameras and measure actual performance.

    Args:
        max_indices: Maximum camera index to probe.
        test_frames: Number of test frames to read to measure actual FPS.

    Returns:
        List of CameraInfo descriptors for all probed devices.
    """
    results: List[CameraInfo] = []

    for idx in range(max_indices):
        # On Windows, try DirectShow first for fast index mapping
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        backend = "DSHOW"
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
            backend = "DEFAULT"

        if not cap.isOpened():
            results.append(
                CameraInfo(
                    index=idx,
                    is_available=False,
                    width=0,
                    height=0,
                    actual_fps=0.0,
                    backend=backend,
                )
            )
            continue

        # Device opened; measure actual frame grab rate
        frames_read = 0
        t0 = time.perf_counter()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for _ in range(test_frames):
            ret, frame = cap.read()
            if ret and frame is not None:
                frames_read += 1
            else:
                break

        elapsed = time.perf_counter() - t0
        actual_fps = (frames_read / elapsed) if (elapsed > 0 and frames_read > 0) else 0.0

        cap.release()

        results.append(
            CameraInfo(
                index=idx,
                is_available=frames_read > 0,
                width=w,
                height=h,
                actual_fps=actual_fps,
                backend=backend,
            )
        )

    return results


class CameraManager:
    """Manages OpenCV VideoCapture stream, frame timestamps, and measured FPS."""

    def __init__(self, config: Optional[CameraConfig] = None):
        """Initialize with optional configuration."""
        self.config = config or CameraConfig()
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened: bool = False
        self.actual_width: int = 0
        self.actual_height: int = 0

        # Rolling timestamp history for actual FPS measurement
        self._timestamps: Deque[float] = deque(maxlen=30)
        self.measured_fps: float = 0.0
        self.frame_count: int = 0

    def open(self) -> bool:
        """Open the camera stream with fallback handling."""
        idx = self.config.camera_index

        # Prefer DirectShow on Windows for external USB cameras
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(idx)

        if not self.cap.isOpened():
            self.is_opened = False
            return False

        # Attempt to configure requested resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.requested_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

        # Inspect actual resolution obtained
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.is_opened = True
        self.frame_count = 0
        self._timestamps.clear()

        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Read a frame from the camera with high-resolution monotonic timestamp.

        Returns:
            Tuple of (success_flag, frame_bgr_array_or_None, timestamp_seconds).
        """
        if not self.is_opened or self.cap is None:
            return False, None, time.perf_counter()

        now = time.perf_counter()
        ret, frame = self.cap.read()

        if ret and frame is not None:
            fh, fw = frame.shape[:2]
            # Lock resolution: verify that frame matches calibrated camera dimensions
            if self.actual_width > 0 and self.actual_height > 0:
                if fw != self.actual_width or fh != self.actual_height:
                    # Resolution changed unexpectedly
                    return False, None, now

            self.frame_count += 1
            self._timestamps.append(now)

            # Update running FPS
            if len(self._timestamps) > 1:
                dt_total = self._timestamps[-1] - self._timestamps[0]
                if dt_total > 0:
                    self.measured_fps = (len(self._timestamps) - 1) / dt_total

            return True, frame, now

        return False, None, now

    def verify_frame_dimensions(self, frame: np.ndarray) -> bool:
        """Verify that incoming frame matches locked resolution."""
        if frame is None:
            return False
        fh, fw = frame.shape[:2]
        return fw == self.actual_width and fh == self.actual_height

    def release(self) -> None:
        """Release video capture device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_opened = False

    def __enter__(self) -> "CameraManager":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
