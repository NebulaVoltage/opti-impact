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
    fb_max_error: float = 1.5  # Max bidirectional forward-backward error in pixels
    mad_k: float = 2.5         # Outlier multiplier for Median Absolute Deviation
    min_spread: float = 0.35   # Minimum spread floor (pixels) to avoid noise over-filtering


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
        fb_errors: Optional N-length array of bidirectional tracking errors in pixels.
        rejected_pts: Optional M x 2 array of rejected feature points.
        rejection_reasons: Optional list of rejection reasons for diagnostics.
        inlier_mask: Optional boolean array marking inliers after MAD filtering.
        confidence: Normalized measurement confidence score in [0.0, 1.0].
        mad_x: Median Absolute Deviation along X axis in pixels.
        mad_y: Median Absolute Deviation along Y axis in pixels.
        spatial_spread_ratio: Ratio of feature bounding box area to specimen ROI area.
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
    fb_errors: Optional[np.ndarray] = None
    rejected_pts: Optional[np.ndarray] = None
    rejection_reasons: Optional[List[str]] = None
    inlier_mask: Optional[np.ndarray] = None
    confidence: float = 1.0
    mad_x: float = 0.0
    mad_y: float = 0.0
    spatial_spread_ratio: float = 1.0


class OpticalFeatureTracker:
    """Tracks natural visual features across frames using bidirectional Lucas–Kanade optical flow."""

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
        if self.prev_gray is None or self.tracked_pts is None or len(self.tracked_pts) == 0:
            self.prev_gray = curr_gray
            corners = self.detect_features(curr_gray)

            if corners is not None and len(corners) > 0:
                self.tracked_pts = corners.astype(np.float32)
                raw_pts = corners.reshape(-1, 2).astype(np.float32)
                self.ref_pts = raw_pts - self.reference_offset
                self.initial_pts_count = len(corners)
                valid_count = len(corners)
                quality = "GOOD" if valid_count >= self.config.min_tracked_points else "DEGRADED"
                score = 1.0 if valid_count >= self.config.min_tracked_points else float(valid_count / max(1, self.config.min_tracked_points))
                disps = raw_pts - self.ref_pts
                step_disps = np.zeros_like(disps)
                inliers = np.ones(valid_count, dtype=bool)

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
                    fb_errors=np.zeros(valid_count, dtype=np.float32),
                    rejected_pts=np.empty((0, 2), dtype=np.float32),
                    rejection_reasons=[],
                    inlier_mask=inliers,
                    confidence=1.0 if quality == "GOOD" else 0.7,
                    mad_x=0.0,
                    mad_y=0.0,
                    spatial_spread_ratio=1.0,
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
                    fb_errors=np.empty(0, dtype=np.float32),
                    rejected_pts=empty_pts,
                    rejection_reasons=[],
                    inlier_mask=np.empty(0, dtype=bool),
                    confidence=0.0,
                    mad_x=0.0,
                    mad_y=0.0,
                    spatial_spread_ratio=0.0,
                )

        # 1. Forward Lucas-Kanade Optical Flow (prev -> curr)
        fwd_pts, st_fwd, err_fwd = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            curr_gray,
            self.tracked_pts,
            None,
            **self.lk_params,
        )

        # 2. Backward Lucas-Kanade Optical Flow (curr -> prev) for bidirectional consistency
        rejected_list: List[np.ndarray] = []
        rejection_reasons: List[str] = []

        if fwd_pts is not None and st_fwd is not None:
            bwd_pts, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
                curr_gray,
                self.prev_gray,
                fwd_pts,
                None,
                **self.lk_params,
            )

            st_fwd_flat = st_fwd.flatten() == 1
            st_bwd_flat = (st_bwd.flatten() == 1) if st_bwd is not None else np.zeros(len(st_fwd_flat), dtype=bool)
            err_fwd_flat = err_fwd.flatten() if err_fwd is not None else np.zeros(len(st_fwd_flat))

            prev_flat = self.tracked_pts.reshape(-1, 2)
            curr_flat = fwd_pts.reshape(-1, 2)
            bwd_flat = bwd_pts.reshape(-1, 2) if bwd_pts is not None else prev_flat

            # Compute bidirectional forward-backward Euclidean distance
            fb_errors = np.linalg.norm(prev_flat - bwd_flat, axis=1)

            # Check individual filtering conditions
            status_ok = st_fwd_flat & st_bwd_flat
            lk_err_ok = err_fwd_flat <= self.config.max_lk_error
            fb_ok = fb_errors <= self.config.fb_max_error

            # Specimen ROI boundary check: points must remain within ROI bounds
            if self.roi is not None:
                rx, ry, rw, rh = self.roi
                in_roi = (
                    (curr_flat[:, 0] >= rx)
                    & (curr_flat[:, 0] < rx + rw)
                    & (curr_flat[:, 1] >= ry)
                    & (curr_flat[:, 1] < ry + rh)
                )
            else:
                in_roi = np.ones(len(curr_flat), dtype=bool)

            valid_mask = status_ok & lk_err_ok & fb_ok & in_roi

            # Record rejection diagnostics
            for idx, is_valid in enumerate(valid_mask):
                if not is_valid:
                    pt = curr_flat[idx] if status_ok[idx] else prev_flat[idx]
                    rejected_list.append(pt)
                    if not status_ok[idx]:
                        rejection_reasons.append("LK_FLOW_FAILED")
                    elif not fb_ok[idx]:
                        rejection_reasons.append(f"FB_ERROR_HIGH_{fb_errors[idx]:.2f}")
                    elif not in_roi[idx]:
                        rejection_reasons.append("OUT_OF_ROI")
                    else:
                        rejection_reasons.append("PATCH_SSD_HIGH")

            good_prev = prev_flat[valid_mask]
            good_curr = curr_flat[valid_mask]
            good_fb = fb_errors[valid_mask]

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
            good_fb = np.empty(0, dtype=np.float32)
            valid_mask = np.empty(0, dtype=bool)
            valid_count = 0

        # 3. Robust Median Absolute Deviation (MAD) Spatial Outlier Filtering
        mad_x = 0.0
        mad_y = 0.0
        inlier_mask = np.ones(valid_count, dtype=bool)

        if valid_count >= 3 and len(rel_disp) > 0:
            dx_arr = rel_disp[:, 0]
            dy_arr = rel_disp[:, 1]
            med_dx = float(np.median(dx_arr))
            med_dy = float(np.median(dy_arr))
            mad_x = float(np.median(np.abs(dx_arr - med_dx)))
            mad_y = float(np.median(np.abs(dy_arr - med_dy)))

            spread_x = max(mad_x, self.config.min_spread)
            spread_y = max(mad_y, self.config.min_spread)

            inlier_mask = (np.abs(dx_arr - med_dx) <= self.config.mad_k * spread_x) & (
                np.abs(dy_arr - med_dy) <= self.config.mad_k * spread_y
            )

            # If at least 3 inliers agree, use inlier-filtered statistics
            if np.sum(inlier_mask) < 3:
                # Fallback: keep all if dispersion is uniform
                inlier_mask = np.ones(valid_count, dtype=bool)

        # 4. Feature Spatial Distribution Ratio
        if valid_count >= 2 and self.roi is not None:
            x_min, y_min = np.min(good_curr, axis=0)
            x_max, y_max = np.max(good_curr, axis=0)
            span_area = float(max(1.0, (x_max - x_min) * (y_max - y_min)))
            roi_area = float(max(1.0, self.roi[2] * self.roi[3]))
            spatial_spread_ratio = min(1.0, max(0.05, span_area / (0.40 * roi_area)))
        else:
            spatial_spread_ratio = 1.0 if valid_count > 0 else 0.0

        # 5. Continuous Measurement Confidence Score [0.0, 1.0]
        if valid_count > 0:
            s_count = min(1.0, float(valid_count / max(1, self.config.min_tracked_points)))
            mean_fb = float(np.mean(good_fb)) if len(good_fb) > 0 else 0.0
            s_fb = max(0.0, 1.0 - (mean_fb / max(0.1, self.config.fb_max_error)))
            s_spread = max(0.0, 1.0 - (mad_x + mad_y) / 10.0)
            s_dist = min(1.0, max(0.1, spatial_spread_ratio))

            confidence = float(0.35 * s_count + 0.30 * s_fb + 0.20 * s_spread + 0.15 * s_dist)
            confidence = min(1.0, max(0.0, confidence))
        else:
            confidence = 0.0

        # Assess tracking quality categorical
        score = float(valid_count / max(1, self.initial_pts_count)) if self.initial_pts_count > 0 else 0.0
        score = min(1.0, max(0.0, score))

        if valid_count >= self.config.min_tracked_points:
            quality = "GOOD"
        elif valid_count > 0:
            quality = "DEGRADED"
        else:
            quality = "LOST"

        # 6. Safe Feature Re-Detection Handling (Preventing Reference Drift)
        redetected = False
        if valid_count < self.config.min_tracked_points:
            # Preserve structural displacement continuity across re-detection only if inliers exist
            if valid_count >= 3 and len(rel_disp) > 0 and np.sum(inlier_mask) >= 3:
                current_offset = np.median(rel_disp[inlier_mask], axis=0)
                self.reference_offset = current_offset.astype(np.float32)
            elif valid_count == 0:
                # On total tracking loss, do not carry forward stale corrupted offsets
                self.reference_offset = np.zeros(2, dtype=np.float32)

            new_corners = self.detect_features(curr_gray)
            if new_corners is not None and len(new_corners) >= self.config.min_tracked_points:
                raw_new = new_corners.reshape(-1, 2).astype(np.float32)
                self.tracked_pts = new_corners.astype(np.float32)
                # Re-reference new corners to verified displacement offset
                self.ref_pts = raw_new - self.reference_offset
                self.initial_pts_count = len(new_corners)
                good_curr = raw_new
                good_prev = raw_new
                good_ref = self.ref_pts
                rel_disp = good_curr - good_ref
                step_disp = np.zeros_like(good_curr)
                good_fb = np.zeros(len(new_corners), dtype=np.float32)
                valid_count = len(new_corners)
                inlier_mask = np.ones(valid_count, dtype=bool)
                score = 1.0
                quality = "GOOD"
                confidence = 0.95
                redetected = True
            else:
                self.tracked_pts = good_curr.reshape(-1, 1, 2) if valid_count > 0 else None
                self.ref_pts = good_ref if valid_count > 0 else None
        else:
            # Advance tracked points to current positions
            self.tracked_pts = good_curr.reshape(-1, 1, 2)
            self.ref_pts = good_ref

        self.prev_gray = curr_gray
        rejected_arr = np.array(rejected_list, dtype=np.float32) if rejected_list else np.empty((0, 2), dtype=np.float32)

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
            fb_errors=good_fb,
            rejected_pts=rejected_arr,
            rejection_reasons=rejection_reasons,
            inlier_mask=inlier_mask,
            confidence=confidence,
            mad_x=mad_x,
            mad_y=mad_y,
            spatial_spread_ratio=spatial_spread_ratio,
        )

    def reset(self) -> None:
        """Reset internal history, clear reference origin, and force re-detection on next frame."""
        self.prev_gray = None
        self.tracked_pts = None
        self.ref_pts = None
        self.reference_offset = np.zeros(2, dtype=np.float32)
        self.initial_pts_count = 0

