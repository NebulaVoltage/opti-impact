"""Feature detection and Lucas–Kanade optical flow tracking module.

Detects high-contrast corner features on the structural specimen ROI using
cv2.goodFeaturesToTrack and computes sub-pixel frame-to-frame motion trajectories
using pyramidal Lucas–Kanade optical flow (cv2.calcOpticalFlowPyrLK).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class TrackerConfig:
    """Corner detection and Lucas–Kanade optical flow parameters."""

    max_corners: int = 100
    quality_level: float = 0.01
    min_distance: float = 10.0
    block_size: int = 7
    min_tracked_points: int = 15
    max_lk_error: float = 35.0
    lk_win_size: Tuple[int, int] = (21, 21)
    lk_max_level: int = 3


@dataclass
class TrackedPoints:
    """Results of frame-to-frame feature tracking.

    Attributes:
        prev_pts: N x 2 array of previous frame points (float32).
        curr_pts: N x 2 array of tracked current frame points (float32).
        displacements: N x 2 array of relative displacement vectors (curr_pts - ref_pts) in pixels.
        status: N-length boolean array indicating valid tracks.
        valid_count: Number of successfully tracked features.
        tracking_quality: Categorical status ('GOOD', 'DEGRADED', 'LOST').
        quality_score: Numerical tracking retention score in [0.0, 1.0].
        redetected: True if new features were detected on this frame.
        ref_pts: Optional N x 2 array of reference baseline positions.
        step_displacements: Optional N x 2 array of inter-frame displacements (curr_pts - prev_pts).
        measurement_valid: True if tracked points are physically valid.
    """

    prev_pts: np.ndarray
    curr_pts: np.ndarray
    displacements: np.ndarray
    status: np.ndarray
    valid_count: int
    tracking_quality: str
    quality_score: float
    redetected: bool = False
    ref_pts: Optional[np.ndarray] = None
    step_displacements: Optional[np.ndarray] = None
    measurement_valid: bool = True


class OpticalFeatureTracker:
    """Tracks natural visual features across frames using Lucas–Kanade optical flow."""

    def __init__(
        self,
        config: Optional[TrackerConfig] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ):
        """Initialize feature tracker.

        Args:
            config: TrackerConfig parameters.
            roi: Optional region of interest as (x, y, width, height).
        """
        self.config = config or TrackerConfig()
        self.roi = roi  # (x, y, w, h)

        self.prev_gray: Optional[np.ndarray] = None
        self.tracked_pts: Optional[np.ndarray] = None
        self.ref_pts: Optional[np.ndarray] = None
        self.reference_offset: np.ndarray = np.zeros(2, dtype=np.float32)
        self.initial_pts_count: int = 0

        # Sub-pixel LK parameters
        self.lk_params = dict(
            winSize=self.config.lk_win_size,
            maxLevel=self.config.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def set_roi(self, x: int, y: int, w: int, h: int) -> None:
        """Update region of interest rectangle."""
        self.roi = (int(x), int(y), int(w), int(h))

    def _get_roi_mask(self, shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Create binary mask for region of interest."""
        if self.roi is None:
            return None

        h_img, w_img = shape[:2]
        x, y, w, h = self.roi

        # Clamp ROI to image boundaries
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))

        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        mask[y : y + h, x : x + w] = 255
        return mask

    def detect_features(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Detect strong corners inside the specimen ROI using goodFeaturesToTrack.

        Args:
            gray: 2D grayscale image array.

        Returns:
            N x 1 x 2 float32 array of detected corner points or None.
        """
        mask = self._get_roi_mask(gray.shape)
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.config.max_corners,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance,
            blockSize=self.config.block_size,
            mask=mask,
        )

        if corners is not None and len(corners) > 0:
            return corners.astype(np.float32)
        return None

    def track(self, frame_bgr_or_gray: np.ndarray) -> TrackedPoints:
        """Process incoming frame and compute optical flow tracking relative to reference positions.

        Args:
            frame_bgr_or_gray: Incoming image frame.

        Returns:
            TrackedPoints object with trajectories, relative displacements, and quality scores.
        """
        if len(frame_bgr_or_gray.shape) == 3:
            curr_gray = cv2.cvtColor(frame_bgr_or_gray, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = frame_bgr_or_gray.copy()

        # First frame or re-initialization
        if self.prev_gray is None or self.tracked_pts is None:
            self.prev_gray = curr_gray
            corners = self.detect_features(curr_gray)

            if corners is not None and len(corners) > 0:
                self.tracked_pts = corners.astype(np.float32)
                # Anchor reference points taking into account current offset (0 on reset)
                raw_pts = corners.reshape(-1, 2).astype(np.float32)
                self.ref_pts = raw_pts - self.reference_offset
                self.initial_pts_count = len(corners)
                valid_count = len(corners)
                quality = "GOOD" if valid_count >= self.config.min_tracked_points else "DEGRADED"
                score = 1.0 if valid_count >= self.config.min_tracked_points else float(valid_count / max(1, self.config.min_tracked_points))
                disps = raw_pts - self.ref_pts
                step_disps = np.zeros_like(disps)

                return TrackedPoints(
                    prev_pts=raw_pts,
                    curr_pts=raw_pts,
                    displacements=disps,
                    status=np.ones(valid_count, dtype=bool),
                    valid_count=valid_count,
                    tracking_quality=quality,
                    quality_score=score,
                    redetected=True,
                    ref_pts=self.ref_pts,
                    step_displacements=step_disps,
                    measurement_valid=True,
                )
            else:
                empty_pts = np.empty((0, 2), dtype=np.float32)
                return TrackedPoints(
                    prev_pts=empty_pts,
                    curr_pts=empty_pts,
                    displacements=empty_pts,
                    status=np.empty(0, dtype=bool),
                    valid_count=0,
                    tracking_quality="LOST",
                    quality_score=0.0,
                    redetected=True,
                    ref_pts=empty_pts,
                    step_displacements=empty_pts,
                    measurement_valid=False,
                )

        # Execute Lucas-Kanade optical flow
        new_pts, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            curr_gray,
            self.tracked_pts,
            None,
            **self.lk_params,
        )

        # Filter valid points
        if new_pts is not None and st is not None:
            st_flat = st.flatten() == 1
            err_flat = err.flatten() if err is not None else np.zeros(len(st_flat))
            valid_mask = st_flat & (err_flat <= self.config.max_lk_error)

            # Strict ROI boundary check: Points must remain strictly inside specimen ROI
            if self.roi is not None:
                rx, ry, rw, rh = self.roi
                pts_flat = new_pts.reshape(-1, 2)
                in_roi = (
                    (pts_flat[:, 0] >= rx)
                    & (pts_flat[:, 0] < rx + rw)
                    & (pts_flat[:, 1] >= ry)
                    & (pts_flat[:, 1] < ry + rh)
                )
                valid_mask = valid_mask & in_roi

            good_prev = self.tracked_pts[valid_mask].reshape(-1, 2)
            good_curr = new_pts[valid_mask].reshape(-1, 2)
            if self.ref_pts is not None:
                aligned_ref = self.ref_pts[: len(valid_mask)]
                good_ref = aligned_ref[valid_mask].reshape(-1, 2)
            else:
                good_ref = good_prev
            rel_disp = good_curr - good_ref
            step_disp = good_curr - good_prev
            valid_count = len(good_curr)
        else:
            good_prev = np.empty((0, 2), dtype=np.float32)
            good_curr = np.empty((0, 2), dtype=np.float32)
            good_ref = np.empty((0, 2), dtype=np.float32)
            rel_disp = np.empty((0, 2), dtype=np.float32)
            step_disp = np.empty((0, 2), dtype=np.float32)
            valid_mask = np.empty(0, dtype=bool)
            valid_count = 0

        # Assess tracking quality
        score = float(valid_count / max(1, self.initial_pts_count)) if self.initial_pts_count > 0 else 0.0
        score = min(1.0, max(0.0, score))

        if valid_count >= self.config.min_tracked_points:
            quality = "GOOD"
        elif valid_count > 0:
            quality = "DEGRADED"
        else:
            quality = "LOST"

        # Check if re-detection is required
        redetected = False
        if valid_count < self.config.min_tracked_points:
            # Preserve structural displacement continuity across re-detection
            if valid_count > 0:
                current_offset = np.median(rel_disp, axis=0)
                self.reference_offset = current_offset.astype(np.float32)

            new_corners = self.detect_features(curr_gray)
            if new_corners is not None and len(new_corners) >= self.config.min_tracked_points:
                raw_new = new_corners.reshape(-1, 2).astype(np.float32)
                self.tracked_pts = new_corners.astype(np.float32)
                # Re-reference new corners to existing displacement offset to avoid step jump
                self.ref_pts = raw_new - self.reference_offset
                self.initial_pts_count = len(new_corners)
                good_curr = raw_new
                good_prev = raw_new
                good_ref = self.ref_pts
                rel_disp = good_curr - good_ref
                step_disp = np.zeros_like(good_curr)
                valid_count = len(new_corners)
                score = 1.0
                quality = "GOOD"
                redetected = True
            else:
                self.tracked_pts = good_curr.reshape(-1, 1, 2) if valid_count > 0 else None
                self.ref_pts = good_ref if valid_count > 0 else None
        else:
            # Advance tracked points to current positions
            self.tracked_pts = good_curr.reshape(-1, 1, 2)
            self.ref_pts = good_ref

        self.prev_gray = curr_gray

        return TrackedPoints(
            prev_pts=good_prev,
            curr_pts=good_curr,
            displacements=rel_disp,
            status=valid_mask,
            valid_count=valid_count,
            tracking_quality=quality,
            quality_score=score,
            redetected=redetected,
            ref_pts=good_ref,
            step_displacements=step_disp,
            measurement_valid=(quality != "LOST"),
        )

    def reset(self) -> None:
        """Reset internal history, clear reference origin, and force re-detection on next frame."""
        self.prev_gray = None
        self.tracked_pts = None
        self.ref_pts = None
        self.reference_offset = np.zeros(2, dtype=np.float32)
        self.initial_pts_count = 0
