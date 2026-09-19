"""Live computer vision visualization, tracking overlays, HUD telemetry, and rolling plot."""

from collections import deque
from typing import Deque, List, Optional, Tuple
import cv2
import numpy as np

from optical.calibration import PlanarCalibration
from optical.feature_tracker import TrackedPoints
from optical.optical_motion import OpticalKinematics


class OpticalVisualizer:
    """Renders feature tracking points, flow vectors, HUD telemetry, and live displacement graph."""

    def __init__(
        self,
        plot_history_len: int = 150,
        vector_scale: float = 3.0,
    ):
        """Initialize visualizer.

        Args:
            plot_history_len: Rolling buffer size for the live time-series plot.
            vector_scale: Amplification factor for visualizing subtle optical flow arrows.
        """
        self.plot_history_len = plot_history_len
        self.vector_scale = vector_scale

        # Rolling history for embedded live plot
        self._disp_history: Deque[float] = deque(maxlen=plot_history_len)
        self._x_history: Deque[float] = deque(maxlen=plot_history_len)
        self._y_history: Deque[float] = deque(maxlen=plot_history_len)

    def draw_frame(
        self,
        frame_bgr: np.ndarray,
        tracked: TrackedPoints,
        kinematics: OpticalKinematics,
        camera_fps: float,
        roi: Optional[Tuple[int, int, int, int]] = None,
        calibration: Optional[PlanarCalibration] = None,
        show_plot: bool = True,
    ) -> np.ndarray:
        """Render optical flow vectors, ROI, HUD telemetry, and live graph onto frame.

        Args:
            frame_bgr: Raw input BGR image.
            tracked: TrackedPoints containing point coordinates and displacements.
            kinematics: OpticalKinematics computed for the current frame.
            camera_fps: Measured camera frame rate.
            roi: Optional (x, y, w, h) bounding rectangle.
            calibration: Optional PlanarCalibration instance.
            show_plot: Whether to append a live displacement plot strip at the bottom.

        Returns:
            Annotated BGR image ready for cv2.imshow or video streaming.
        """
        vis = frame_bgr.copy()
        h_img, w_img = vis.shape[:2]

        # 1. Draw ROI Box
        if roi is not None:
            rx, ry, rw, rh = roi
            cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
            cv2.putText(
                vis,
                "SPECIMEN ROI",
                (rx + 5, ry + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 2. Draw Tracked Points & Optical Flow Vectors
        if tracked.valid_count > 0:
            for p_prev, p_curr in zip(tracked.prev_pts, tracked.curr_pts):
                x0, y0 = int(round(p_prev[0])), int(round(p_prev[1]))
                x1, y1 = int(round(p_curr[0])), int(round(p_curr[1]))

                # Feature point circle
                cv2.circle(vis, (x1, y1), 3, (0, 255, 0), -1, cv2.LINE_AA)

                # Motion vector arrow (scaled for visibility)
                dx = (p_curr[0] - p_prev[0]) * self.vector_scale
                dy = (p_curr[1] - p_prev[1]) * self.vector_scale
                if abs(dx) > 0.5 or abs(dy) > 0.5:
                    pt_end = (int(round(x1 + dx)), int(round(y1 + dy)))
                    cv2.arrowedLine(vis, (x1, y1), pt_end, (0, 0, 255), 1, tipLength=0.3)

        # 3. Draw Semi-Transparent Telemetry HUD Overlay
        overlay = vis.copy()
        hud_w = 340
        hud_h = 240
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)
        cv2.rectangle(vis, (10, 10), (10 + hud_w, 10 + hud_h), (0, 255, 255), 1)

        # Telemetry Texts
        q_color = (
            (0, 255, 0)
            if tracked.tracking_quality == "GOOD"
            else ((0, 165, 255) if tracked.tracking_quality == "DEGRADED" else (0, 0, 255))
        )

        lines = [
            ("OPTICAL SENSOR TELEMETRY", (0, 255, 255), 0.55, 2),
            (f"Camera FPS:        {camera_fps:5.1f}", (255, 255, 255), 0.45, 1),
            (f"Valid Features:    {tracked.valid_count:5d}", (255, 255, 255), 0.45, 1),
            (f"Tracking Quality:  {tracked.tracking_quality}", q_color, 0.45, 2),
            (f"Disp X:           {kinematics.cum_x_pixels:+7.2f} px", (255, 255, 255), 0.45, 1),
            (f"Disp Y:           {kinematics.cum_y_pixels:+7.2f} px", (255, 255, 255), 0.45, 1),
            (f"Total Disp:        {kinematics.cum_displacement_pixels:7.2f} px", (0, 255, 255), 0.45, 2),
            (f"Velocity Mag:      {kinematics.velocity_magnitude_pixels_s:7.2f} px/s", (255, 255, 255), 0.45, 1),
            (
                f"Opt Frequency:     {kinematics.dominant_frequency_hz:5.2f} Hz"
                if not np.isnan(kinematics.dominant_frequency_hz)
                else "Opt Frequency:     N/A (insufficient)",
                (255, 200, 100),
                0.45,
                1,
            ),
        ]

        # Metric calibration line
        if calibration is not None and calibration.is_calibrated and calibration.scale_mm_per_pixel is not None:
            disp_mm = kinematics.cum_displacement_pixels * calibration.scale_mm_per_pixel
            lines.append((f"Physical Disp:     {disp_mm:6.2f} mm", (50, 255, 50), 0.45, 2))
        else:
            lines.append(("METRIC CALIBRATION: NOT ACTIVE", (150, 150, 150), 0.40, 1))

        y_offset = 32
        for text, col, scale, thickness in lines:
            cv2.putText(
                vis,
                text,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                col,
                thickness,
                cv2.LINE_AA,
            )
            y_offset += 20

        # 4. Live Rolling Displacement Graph Strip
        if show_plot:
            self._disp_history.append(kinematics.cum_displacement_pixels)
            vis = self._draw_embedded_plot(vis)

        return vis

    def _draw_embedded_plot(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Render a clean rolling time-series graph strip at the bottom of the frame."""
        h_img, w_img = frame_bgr.shape[:2]
        plot_h = 120
        plot_w = w_img

        # Create plot canvas
        canvas = np.full((plot_h, plot_w, 3), 25, dtype=np.uint8)

        # Border and title
        cv2.line(canvas, (0, 0), (plot_w, 0), (80, 80, 80), 1)
        cv2.putText(
            canvas,
            "LIVE DISPLACEMENT HISTORY (Rolling Pixels)",
            (15, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        n_pts = len(self._disp_history)
        if n_pts >= 2:
            data = np.array(self._disp_history)
            max_val = max(10.0, float(np.max(data)) * 1.2)
            min_val = min(0.0, float(np.min(data)))
            val_range = max(1.0, max_val - min_val)

            # Map points onto canvas coordinates
            xs = np.linspace(20, plot_w - 20, n_pts).astype(int)
            ys = (plot_h - 15 - ((data - min_val) / val_range) * (plot_h - 40)).astype(int)

            pts = np.column_stack((xs, ys)).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=False, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

            # Display current and max scale values
            cur_val = self._disp_history[-1]
            cv2.putText(
                canvas,
                f"Current: {cur_val:5.2f} px | Max: {max_val:5.2f} px",
                (plot_w - 260, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        # Concatenate vertically
        combined = np.vstack([frame_bgr, canvas])
        return combined
