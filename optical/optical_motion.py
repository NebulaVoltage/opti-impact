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


class OpticalMotionEstimator:
    """Estimates robust structural displacement, kinematics, and frequency from optical flow."""

    def __init__(
        self,
        history_len: int = 300,
        min_freq: float = 0.5,
        smoothing_window: int = 7,
    ):
        """Initialize motion estimator.

        Args:
            history_len: Length of rolling history buffer for differentiation and FFT.
            min_freq: Minimum frequency in Hz for dominant frequency peak search.
            smoothing_window: Odd integer window length for kinematic smoothing.
        """
        self.history_len = history_len
        self.min_freq = min_freq
        self.smoothing_window = smoothing_window if smoothing_window % 2 != 0 else smoothing_window + 1

        # Cumulative displacement accumulators
        self.cum_x: float = 0.0
        self.cum_y: float = 0.0

        # Rolling time-series histories
        self._timestamps: Deque[float] = deque(maxlen=history_len)
        self._cum_x_hist: Deque[float] = deque(maxlen=history_len)
        self._cum_y_hist: Deque[float] = deque(maxlen=history_len)
        self._cum_disp_hist: Deque[float] = deque(maxlen=history_len)
        self._vel_x_hist: Deque[float] = deque(maxlen=history_len)
        self._vel_y_hist: Deque[float] = deque(maxlen=history_len)

        self.frame_counter: int = 0

    def update(
        self,
        tracked: TrackedPoints,
        timestamp: float,
        bg_tracked: Optional[TrackedPoints] = None,
    ) -> OpticalKinematics:
        """Process frame tracking result and compute robust structural kinematics.

        Args:
            tracked: TrackedPoints from the specimen ROI.
            timestamp: Monotonic timestamp in seconds.
            bg_tracked: Optional TrackedPoints from stationary background for camera motion rejection.

        Returns:
            OpticalKinematics dataclass instance.
        """
        self.frame_counter += 1

        # 1. Robust median displacement aggregation
        if tracked.valid_count > 0 and len(tracked.displacements) > 0:
            median_dx = float(np.median(tracked.displacements[:, 0]))
            median_dy = float(np.median(tracked.displacements[:, 1]))
        else:
            median_dx = 0.0
            median_dy = 0.0

        # 2. Camera Global Motion Rejection
        # If background points are tracked, subtract global camera motion
        if bg_tracked is not None and bg_tracked.valid_count > 0 and len(bg_tracked.displacements) > 0:
            bg_dx = float(np.median(bg_tracked.displacements[:, 0]))
            bg_dy = float(np.median(bg_tracked.displacements[:, 1]))
            # Subtract camera motion
            median_dx -= bg_dx
            median_dy -= bg_dy

        frame_disp = float(np.sqrt(median_dx**2 + median_dy**2))

        # Accumulate cumulative displacement relative to initial baseline
        self.cum_x += median_dx
        self.cum_y += median_dy
        cum_disp = float(np.sqrt(self.cum_x**2 + self.cum_y**2))

        # Append to histories
        self._timestamps.append(timestamp)
        self._cum_x_hist.append(self.cum_x)
        self._cum_y_hist.append(self.cum_y)
        self._cum_disp_hist.append(cum_disp)

        # 3. Velocity Calculation
        vx = 0.0
        vy = 0.0
        if len(self._timestamps) >= 2:
            dt = self._timestamps[-1] - self._timestamps[-2]
            if dt > 1e-4:
                vx = (self._cum_x_hist[-1] - self._cum_x_hist[-2]) / dt
                vy = (self._cum_y_hist[-1] - self._cum_y_hist[-2]) / dt
        v_mag = float(np.sqrt(vx**2 + vy**2))
        self._vel_x_hist.append(vx)
        self._vel_y_hist.append(vy)

        # 4. Acceleration Calculation with Smoothing
        ax = 0.0
        ay = 0.0
        if len(self._timestamps) >= self.smoothing_window:
            t_arr = np.array(self._timestamps)
            dt_mean = float(np.mean(np.diff(t_arr[-self.smoothing_window :])))

            if dt_mean > 1e-4:
                vx_arr = np.array(list(self._vel_x_hist)[-self.smoothing_window :])
                vy_arr = np.array(list(self._vel_y_hist)[-self.smoothing_window :])

                # Smooth velocity using Savitzky-Golay filter to suppress differentiation noise
                try:
                    vx_smooth = savgol_filter(vx_arr, window_length=self.smoothing_window, polyorder=2)
                    vy_smooth = savgol_filter(vy_arr, window_length=self.smoothing_window, polyorder=2)
                    ax = float((vx_smooth[-1] - vx_smooth[-2]) / dt_mean)
                    ay = float((vy_smooth[-1] - vy_smooth[-2]) / dt_mean)
                except Exception:
                    ax = float((vx_arr[-1] - vx_arr[-2]) / dt_mean)
                    ay = float((vy_arr[-1] - vy_arr[-2]) / dt_mean)

        a_mag = float(np.sqrt(ax**2 + ay**2))

        # 5. Dominant Optical Frequency Estimation
        dom_freq = self._estimate_dominant_frequency()

        return OpticalKinematics(
            timestamp=timestamp,
            frame_index=self.frame_counter,
            dx_pixels=median_dx,
            dy_pixels=median_dy,
            displacement_pixels=frame_disp,
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
        )

    def _estimate_dominant_frequency(self) -> float:
        """Estimate the dominant frequency of optical displacement via FFT on the primary motion axis."""
        # Require at least 32 points for meaningful spectral resolution
        if len(self._timestamps) < 32:
            return float("nan")

        x_arr = np.array(self._cum_x_hist)
        y_arr = np.array(self._cum_y_hist)
        t_arr = np.array(self._timestamps)

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
        self._timestamps.clear()
        self._cum_x_hist.clear()
        self._cum_y_hist.clear()
        self._cum_disp_hist.clear()
        self._vel_x_hist.clear()
        self._vel_y_hist.clear()
        self.frame_counter = 0
