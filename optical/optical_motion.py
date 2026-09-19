"""Structural motion estimation, camera motion rejection, kinematics, and frequency analysis.

Provides robust aggregation of multi-point optical flow vectors into physical
structural displacements, velocities, accelerations, and resonant frequencies.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np
from scipy.signal import savgol_filter

from optical.feature_tracker import TrackedPoints


@dataclass
class OpticalKinematics:
    """Instantaneous optical structural motion, kinematics, and resonant frequency.

    All units are expressed in pixels, pixels/s, pixels/s², and Hz.
    """

    timestamp: float
    frame_index: int
    dx_pixels: float
    dy_pixels: float
    displacement_pixels: float
    cum_x_pixels: float
    cum_y_pixels: float
    cum_displacement_pixels: float
    velocity_x_pixels_s: float
    velocity_y_pixels_s: float
    velocity_magnitude_pixels_s: float
    acceleration_x_pixels_s2: float
    acceleration_y_pixels_s2: float
    acceleration_magnitude_pixels_s2: float
    dominant_frequency_hz: float
    valid_feature_count: int
    tracking_quality: str
    measurement_valid: bool = True
    validity_reason: str = "VALID"
    confidence: float = 1.0
    mad_x_pixels: float = 0.0
    mad_y_pixels: float = 0.0
    inlier_feature_count: int = 0


class OpticalMotionEstimator:
    """Estimates robust structural displacement, kinematics, and frequency from optical flow."""

    def __init__(
        self,
        history_len: int = 300,
        min_freq: float = 0.5,
        smoothing_window: int = 7,
        frame_width: int = 1280,
        frame_height: int = 720,
        max_step_disp: float = 80.0,
    ):
        """Initialize motion estimator.

        Args:
            history_len: Length of rolling history buffer for differentiation and FFT.
            min_freq: Minimum frequency in Hz for dominant frequency peak search.
            smoothing_window: Odd integer window length for kinematic smoothing.
            frame_width: Max camera frame width in pixels for sanity validation.
            frame_height: Max camera frame height in pixels for sanity validation.
            max_step_disp: Maximum plausible single-frame displacement step in pixels.
        """
        self.history_len = history_len
        self.min_freq = min_freq
        self.smoothing_window = smoothing_window if smoothing_window % 2 != 0 else smoothing_window + 1
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_step_disp = max_step_disp

        # Structural displacement relative to reference baseline
        self.cum_x: float = 0.0
        self.cum_y: float = 0.0

        # Protected state: last confirmed valid displacement to prevent anomaly locking
        self._last_valid_x: Optional[float] = None
        self._last_valid_y: Optional[float] = None

        # Rolling time-series histories
        self._timestamps: Deque[float] = deque(maxlen=history_len)
        self._cum_x_hist: Deque[float] = deque(maxlen=history_len)
        self._cum_y_hist: Deque[float] = deque(maxlen=history_len)
        self._cum_disp_hist: Deque[float] = deque(maxlen=history_len)
        self._vel_x_hist: Deque[float] = deque(maxlen=history_len)
        self._vel_y_hist: Deque[float] = deque(maxlen=history_len)
        self._valid_flags: Deque[bool] = deque(maxlen=history_len)

        self.frame_counter: int = 0

    def set_frame_dimensions(self, width: int, height: int) -> None:
        """Dynamically synchronize frame boundaries with active camera resolution."""
        if width > 0:
            self.frame_width = int(width)
        if height > 0:
            self.frame_height = int(height)

    def update(
        self,
        tracked: TrackedPoints,
        timestamp: float,
        bg_tracked: Optional[TrackedPoints] = None,
    ) -> OpticalKinematics:
        """Process frame tracking result and compute robust structural kinematics.

        Args:
            tracked: TrackedPoints from the specimen ROI (displacements relative to reference).
            timestamp: Monotonic timestamp in seconds.
            bg_tracked: Optional TrackedPoints from stationary background for camera motion rejection.

        Returns:
            OpticalKinematics dataclass instance.
        """
        self.frame_counter += 1

        # 1. Robust MAD-filtered inlier displacement aggregation relative to reference
        inlier_mask = getattr(tracked, "inlier_mask", None)
        if inlier_mask is not None and np.sum(inlier_mask) >= 3 and len(tracked.displacements) > 0:
            inlier_disps = tracked.displacements[inlier_mask]
            median_dx = float(np.median(inlier_disps[:, 0]))
            median_dy = float(np.median(inlier_disps[:, 1]))
            inlier_count = int(np.sum(inlier_mask))
        elif tracked.valid_count > 0 and len(tracked.displacements) > 0:
            median_dx = float(np.median(tracked.displacements[:, 0]))
            median_dy = float(np.median(tracked.displacements[:, 1]))
            inlier_count = tracked.valid_count
        else:
            median_dx = 0.0
            median_dy = 0.0
            inlier_count = 0

        # 2. Camera Global Motion Rejection (Background Subtraction)
        if bg_tracked is not None and bg_tracked.valid_count > 0 and len(bg_tracked.displacements) > 0:
            bg_inliers = getattr(bg_tracked, "inlier_mask", None)
            if bg_inliers is not None and np.sum(bg_inliers) >= 3:
                bg_dx = float(np.median(bg_tracked.displacements[bg_inliers, 0]))
                bg_dy = float(np.median(bg_tracked.displacements[bg_inliers, 1]))
            else:
                bg_dx = float(np.median(bg_tracked.displacements[:, 0]))
                bg_dy = float(np.median(bg_tracked.displacements[:, 1]))
            median_dx -= bg_dx
            median_dy -= bg_dy

        structural_dx = median_dx
        structural_dy = median_dy
        structural_disp = float(np.sqrt(structural_dx**2 + structural_dy**2))

        # 3. Physical Sanity Checks & Measurement Validity Evaluation
        measurement_valid = True
        validity_reason = "VALID"

        if tracked.tracking_quality == "LOST" or tracked.valid_count == 0:
            measurement_valid = False
            validity_reason = "TRACKING_LOST"
        elif abs(structural_dx) > self.frame_width or abs(structural_dy) > self.frame_height:
            measurement_valid = False
            validity_reason = "EXCEEDS_FRAME_BOUNDS"
        elif getattr(tracked, "confidence", 1.0) < 0.15:
            measurement_valid = False
            validity_reason = "LOW_CONFIDENCE"
        elif self._last_valid_x is not None and not tracked.redetected:
            step_dx = abs(structural_dx - self._last_valid_x)
            step_dy = abs(structural_dy - self._last_valid_y)
            if step_dx > self.max_step_disp or step_dy > self.max_step_disp:
                measurement_valid = False
                validity_reason = "EXCESSIVE_STEP_DISPLACEMENT"

        # Update last verified valid baseline if current measurement passed all checks
        if measurement_valid:
            self._last_valid_x = structural_dx
            self._last_valid_y = structural_dy

        # Position-based structural displacement (relative to reference, NOT unbounded accumulator)
        self.cum_x = structural_dx
        self.cum_y = structural_dy
        cum_disp = structural_disp

        # Append to rolling histories
        self._timestamps.append(timestamp)
        self._cum_x_hist.append(self.cum_x)
        self._cum_y_hist.append(self.cum_y)
        self._cum_disp_hist.append(cum_disp)
        self._valid_flags.append(measurement_valid)

        # 4. Mathematically Consistent Velocity Calculation (v = Delta x / Delta t)
        vx = 0.0
        vy = 0.0
        if measurement_valid and len(self._timestamps) >= 2 and not tracked.redetected:
            # Check that previous measurement was also valid to avoid computing velocity across invalid jumps
            if len(self._valid_flags) >= 2 and list(self._valid_flags)[-2]:
                dt = self._timestamps[-1] - self._timestamps[-2]
                if dt > 1e-4:
                    vx = (self._cum_x_hist[-1] - self._cum_x_hist[-2]) / dt
                    vy = (self._cum_y_hist[-1] - self._cum_y_hist[-2]) / dt
        v_mag = float(np.sqrt(vx**2 + vy**2))
        self._vel_x_hist.append(vx)
        self._vel_y_hist.append(vy)

        # 5. Acceleration Calculation with Savitzky-Golay Smoothing
        ax = 0.0
        ay = 0.0
        if measurement_valid and len(self._timestamps) >= self.smoothing_window:
            t_arr = np.array(self._timestamps)
            dt_mean = float(np.mean(np.diff(t_arr[-self.smoothing_window :])))

            if dt_mean > 1e-4:
                vx_arr = np.array(list(self._vel_x_hist)[-self.smoothing_window :])
                vy_arr = np.array(list(self._vel_y_hist)[-self.smoothing_window :])

                try:
                    vx_smooth = savgol_filter(vx_arr, window_length=self.smoothing_window, polyorder=2)
                    vy_smooth = savgol_filter(vy_arr, window_length=self.smoothing_window, polyorder=2)
                    ax = float((vx_smooth[-1] - vx_smooth[-2]) / dt_mean)
                    ay = float((vy_smooth[-1] - vy_smooth[-2]) / dt_mean)
                except Exception:
                    ax = float((vx_arr[-1] - vx_arr[-2]) / dt_mean)
                    ay = float((vy_arr[-1] - vy_arr[-2]) / dt_mean)

        a_mag = float(np.sqrt(ax**2 + ay**2))

        # 6. Dominant Optical Frequency Estimation on Valid History Only
        dom_freq = self._estimate_dominant_frequency() if measurement_valid else float("nan")

        return OpticalKinematics(
            timestamp=timestamp,
            frame_index=self.frame_counter,
            dx_pixels=structural_dx,
            dy_pixels=structural_dy,
            displacement_pixels=structural_disp,
            cum_x_pixels=self.cum_x,
            cum_y_pixels=self.cum_y,
            cum_displacement_pixels=cum_disp,
            velocity_x_pixels_s=vx,
            velocity_y_pixels_s=vy,
            velocity_magnitude_pixels_s=v_mag,
            acceleration_x_pixels_s2=ax,
            acceleration_y_pixels_s2=ay,
            acceleration_magnitude_pixels_s2=a_mag,
            dominant_frequency_hz=dom_freq,
            valid_feature_count=tracked.valid_count,
            tracking_quality=tracked.tracking_quality,
            measurement_valid=measurement_valid,
            validity_reason=validity_reason,
            confidence=getattr(tracked, "confidence", 1.0),
            mad_x_pixels=getattr(tracked, "mad_x", 0.0),
            mad_y_pixels=getattr(tracked, "mad_y", 0.0),
            inlier_feature_count=inlier_count,
        )

    def _estimate_dominant_frequency(self) -> float:
        """Estimate the dominant frequency of optical displacement via FFT on the primary motion axis.

        Calculates frequency only from valid measurement frames, rejecting invalid or re-detection jumps.
        """
        # Require at least 32 points for meaningful spectral resolution
        if len(self._timestamps) < 32:
            return float("nan")

        # Verify that recent window contains valid measurements
        recent_flags = list(self._valid_flags)[-32:]
        if not all(recent_flags):
            return float("nan")

        x_arr = np.array(list(self._cum_x_hist)[-128:])
        y_arr = np.array(list(self._cum_y_hist)[-128:])
        t_arr = np.array(list(self._timestamps)[-128:])

        # Select primary axis of motion (oscillation coordinate) to avoid rectification doubling
        var_x = float(np.var(x_arr))
        var_y = float(np.var(y_arr))
        signal = x_arr if var_x >= var_y else y_arr

        # Remove DC offset / baseline drift
        signal_detrend = signal - np.mean(signal)

        # Check if signal has sufficient motion variance (avoid computing FFT on pure stationary noise)
        if np.std(signal_detrend) < 0.05:
            return float("nan")

        # Estimate sampling rate
        dt = np.mean(np.diff(t_arr))
        if dt <= 0:
            return float("nan")
        fs = 1.0 / dt

        # Apply Hanning window
        window = np.hanning(len(signal_detrend))
        windowed = signal_detrend * window

        # Compute one-sided FFT
        n = len(windowed)
        fft_vals = np.fft.rfft(windowed)
        fft_freqs = np.fft.rfftfreq(n, d=dt)
        amplitudes = np.abs(fft_vals) * 2.0 / n

        # Mask frequencies below minimum cutoff
        valid_idx = np.where(fft_freqs >= self.min_freq)[0]
        if len(valid_idx) == 0:
            return float("nan")

        peak_idx = valid_idx[np.argmax(amplitudes[valid_idx])]
        dom_freq = float(fft_freqs[peak_idx])

        return dom_freq

    def reset(self) -> None:
        """Reset displacement origin and rolling history."""
        self.cum_x = 0.0
        self.cum_y = 0.0
        self._last_valid_x = None
        self._last_valid_y = None
        self._timestamps.clear()
        self._cum_x_hist.clear()
        self._cum_y_hist.clear()
        self._cum_disp_hist.clear()
        self._vel_x_hist.clear()
        self._vel_y_hist.clear()
        self._valid_flags.clear()
        self.frame_counter = 0
