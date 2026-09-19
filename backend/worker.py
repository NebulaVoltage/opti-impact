"""Optical structural motion acquisition worker and telemetry adapter.

Encapsulates the single-owner physical webcam capture loop, runs the frozen
optical vision pipeline (Lucas-Kanade optical flow, kinematics estimation, and
visualizer overlay), and provides thread-safe access to MJPEG frames and telemetry.
"""

from collections import deque
import logging
import threading
import time
from typing import Callable, Deque, Dict, List, Optional, Tuple
import cv2
import numpy as np

from optical.calibration import PlanarCalibration
from optical.camera import CameraConfig, CameraManager
from optical.feature_tracker import OpticalFeatureTracker
from optical.optical_motion import OpticalKinematics, OpticalMotionEstimator
from optical.visualization import OpticalVisualizer

logger = logging.getLogger("optical_backend.worker")


class OpticalPipelineWorker:
    """Acquisition thread that runs the physical optical monitoring pipeline."""

    def __init__(
        self,
        camera_index: int = 1,
        width: int = 1280,
        height: int = 720,
        roi: Optional[Tuple[int, int, int, int]] = None,
        debug: bool = False,
        allow_fallback: bool = True,
    ):
        self.camera_index = camera_index
        self.requested_width = width
        self.requested_height = height
        self.custom_roi = roi
        self.debug = debug
        self.allow_fallback = allow_fallback

        self.cam_mgr: Optional[CameraManager] = None
        self.tracker: Optional[OpticalFeatureTracker] = None
        self.motion_estimator: Optional[OpticalMotionEstimator] = None
        self.visualizer: Optional[OpticalVisualizer] = None
        self.calibration: Optional[PlanarCalibration] = None

        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._lock = threading.Lock()

        # Telemetry & Video frame state
        self._latest_jpeg: Optional[bytes] = None
        self._latest_telemetry: Optional[Dict] = None
        self._latest_spectrum: Optional[Dict] = None
        self._frame_count: int = 0
        self._is_camera_connected: bool = False
        self._connection_error: Optional[str] = None
        self._roi: Optional[Tuple[int, int, int, int]] = None

        # Rolling history for spectrogram (last 30 spectral slices)
        self._spectrogram_history: Deque[Dict] = deque(maxlen=40)

        # Baseline capture record
        self._baseline_record: Optional[Dict] = None

        # Telemetry broadcast listeners
        self._subscribers: List[Callable[[Dict], None]] = []

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_camera_connected(self) -> bool:
        return self._is_camera_connected

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def connection_error(self) -> Optional[str]:
        return self._connection_error

    def add_subscriber(self, callback: Callable[[Dict], None]) -> None:
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def remove_subscriber(self, callback: Callable[[Dict], None]) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def start(self) -> bool:
        """Start the background physical optical sensing thread."""
        if self._running:
            return True

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="OpticalWorkerThread", daemon=True)
        self._thread.start()
        logger.info(f"OpticalPipelineWorker started on camera index {self.camera_index}.")
        return True

    def stop(self) -> None:
        """Stop worker and release camera hardware."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
            self._thread = None

        with self._lock:
            if self.cam_mgr is not None:
                try:
                    self.cam_mgr.release()
                except Exception as e:
                    logger.warning(f"Error releasing camera: {e}")
                self.cam_mgr = None
            self._is_camera_connected = False

        logger.info("OpticalPipelineWorker stopped.")

    def reset_baseline(self) -> None:
        """Re-center tracker baseline and motion kinematics."""
        with self._lock:
            if self.tracker is not None:
                self.tracker.reset()
            if self.motion_estimator is not None:
                self.motion_estimator.reset()
        logger.info("Optical tracking baseline and origin reset.")

    def capture_baseline(self) -> Optional[Dict]:
        """Capture and record current optical response metrics as the session baseline."""
        with self._lock:
            if self._latest_telemetry is None:
                return None
            t = self._latest_telemetry
            self._baseline_record = {
                "timestamp": t.get("timestamp", 0.0),
                "baseline_frequency_hz": t.get("dominant_frequency_hz"),
                "baseline_displacement_mag_px": t.get("displacement_magnitude_px", 0.0),
                "baseline_velocity_mag_px_s": t.get("velocity_magnitude_px_s", 0.0),
                "baseline_confidence": t.get("confidence", 1.0),
                "captured_at": time.strftime("%H:%M:%S"),
            }
            return self._baseline_record

    def get_baseline(self) -> Optional[Dict]:
        with self._lock:
            return self._baseline_record

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def get_latest_telemetry(self) -> Optional[Dict]:
        with self._lock:
            return self._latest_telemetry

    def get_latest_spectrum(self) -> Optional[Dict]:
        with self._lock:
            return self._latest_spectrum

    def get_spectrogram(self) -> List[Dict]:
        with self._lock:
            return list(self._spectrogram_history)

    def _initialize_pipeline(self) -> bool:
        """Initialize camera hardware and optical algorithms."""
        cam_cfg = CameraConfig(
            camera_index=self.camera_index,
            width=self.requested_width,
            height=self.requested_height,
        )
        self.cam_mgr = CameraManager(config=cam_cfg)

        opened = self.cam_mgr.open()
        if not opened and self.allow_fallback and self.camera_index != 0:
            logger.warning(f"Camera index {self.camera_index} failed to open. Retrying index 0...")
            cam_cfg.camera_index = 0
            self.cam_mgr = CameraManager(config=cam_cfg)
            opened = self.cam_mgr.open()

        if not opened:
            self._is_camera_connected = False
            self._connection_error = f"Failed to open video capture device (index {self.camera_index})."
            logger.error(self._connection_error)
            return False

        self._is_camera_connected = True
        self._connection_error = None
        actual_w = self.cam_mgr.actual_width
        actual_h = self.cam_mgr.actual_height
        logger.info(f"Camera opened successfully: {actual_w}x{actual_h} resolution.")

        # Settle auto-exposure / gain
        for _ in range(8):
            self.cam_mgr.read_frame()
            time.sleep(0.02)

        # Configure specimen ROI
        if self.custom_roi is not None:
            self._roi = self.custom_roi
        else:
            rx = int(actual_w * 0.15)
            ry = int(actual_h * 0.15)
            rw = int(actual_w * 0.70)
            rh = int(actual_h * 0.70)
            self._roi = (rx, ry, rw, rh)

        self.tracker = OpticalFeatureTracker(roi=self._roi)
        self.motion_estimator = OpticalMotionEstimator(frame_width=actual_w, frame_height=actual_h)
        self.visualizer = OpticalVisualizer()
        self.calibration = PlanarCalibration()

        return True

    def _run_loop(self) -> None:
        """Continuous physical acquisition and processing loop."""
        if not self._initialize_pipeline():
            # If camera hardware failed, stay alive to report offline status
            while self._running:
                time.sleep(0.5)
            return

        assert self.cam_mgr is not None
        assert self.tracker is not None
        assert self.motion_estimator is not None
        assert self.visualizer is not None

        while self._running:
            t_frame_start = time.perf_counter()

            ret, frame, t_stamp = self.cam_mgr.read_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            self._frame_count += 1

            # 1. Track optical features via Lucas-Kanade
            tracked = self.tracker.track(frame)

            # 2. Compute kinematics, displacement, and frequency
            kinematics = self.motion_estimator.update(tracked, timestamp=t_stamp)

            # 3. Compute FFT spectrum from rolling buffer
            spectrum = self._compute_fft_spectrum(self.motion_estimator)

            # 4. Render overlay on frame
            annotated = self.visualizer.draw_frame(
                frame_bgr=frame,
                tracked=tracked,
                kinematics=kinematics,
                camera_fps=self.cam_mgr.measured_fps,
                roi=self._roi,
                calibration=self.calibration,
                show_plot=True,
                debug_mode=self.debug,
            )

            # 5. Compress annotated frame to JPEG for MJPEG stream
            success, enc_jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            jpeg_bytes = enc_jpg.tobytes() if success else None

            # 6. Assemble telemetry packet
            packet = self._build_telemetry_packet(kinematics, self.cam_mgr.measured_fps, spectrum)

            # 7. Thread-safe state update
            with self._lock:
                self._latest_jpeg = jpeg_bytes
                self._latest_telemetry = packet
                self._latest_spectrum = spectrum
                if spectrum and "frequencies" in spectrum:
                    self._spectrogram_history.append({
                        "timestamp": packet["timestamp"],
                        "frequencies": spectrum["frequencies"],
                        "amplitudes": spectrum["amplitudes"],
                        "peak_frequency_hz": spectrum["peak_frequency_hz"],
                    })
                subs = list(self._subscribers)

            # 8. Notify WebSocket subscribers
            for cb in subs:
                try:
                    cb(packet)
                except Exception as e:
                    logger.debug(f"Subscriber notify error: {e}")

            # Regulate frame rate to ~30 FPS
            elapsed = time.perf_counter() - t_frame_start
            target_period = 1.0 / 30.0
            if elapsed < target_period:
                time.sleep(target_period - elapsed)

        # Loop exited, release resources
        if self.cam_mgr is not None:
            self.cam_mgr.release()
            self.cam_mgr = None

    def _compute_fft_spectrum(self, motion_estimator: OpticalMotionEstimator) -> Dict:
        """Compute one-sided FFT spectrum directly from motion_estimator rolling buffer."""
        hist_len = len(motion_estimator._timestamps)
        if hist_len < 32:
            return {"frequencies": [], "amplitudes": [], "peak_frequency_hz": None, "peak_amplitude": None}

        x_arr = np.array(list(motion_estimator._cum_x_hist)[-128:])
        y_arr = np.array(list(motion_estimator._cum_y_hist)[-128:])
        t_arr = np.array(list(motion_estimator._timestamps)[-128:])

        # Primary axis of motion
        var_x = float(np.var(x_arr))
        var_y = float(np.var(y_arr))
        signal = x_arr if var_x >= var_y else y_arr
        signal_detrend = signal - np.mean(signal)

        dt = float(np.mean(np.diff(t_arr)))
        if dt <= 0:
            return {"frequencies": [], "amplitudes": [], "peak_frequency_hz": None, "peak_amplitude": None}

        n = len(signal_detrend)
        window = np.hanning(n)
        windowed = signal_detrend * window
        fft_vals = np.fft.rfft(windowed)
        fft_freqs = np.fft.rfftfreq(n, d=dt)
        amplitudes = np.abs(fft_vals) * 2.0 / n

        # Filter to reasonable structural frequency band [0.5 - 15.0 Hz]
        band_mask = (fft_freqs >= 0.5) & (fft_freqs <= 15.0)
        band_freqs = fft_freqs[band_mask]
        band_amps = amplitudes[band_mask]

        if len(band_freqs) == 0:
            return {"frequencies": [], "amplitudes": [], "peak_frequency_hz": None, "peak_amplitude": None}

        peak_idx = int(np.argmax(band_amps))
        peak_f = float(band_freqs[peak_idx])
        peak_amp = float(band_amps[peak_idx])

        # Return formatted frequency and amplitude bins
        return {
            "frequencies": [round(float(f), 2) for f in band_freqs],
            "amplitudes": [round(float(a), 4) for a in band_amps],
            "peak_frequency_hz": round(peak_f, 2),
            "peak_amplitude": round(peak_amp, 4),
        }

    def _build_telemetry_packet(self, kinematics: OpticalKinematics, fps: float, spectrum: Dict) -> Dict:
        """Construct JSON telemetry packet strictly adhering to section 7 contract."""
        dom_freq = kinematics.dominant_frequency_hz
        if dom_freq is not None and (np.isnan(dom_freq) or np.isinf(dom_freq)):
            dom_freq = None
        else:
            dom_freq = round(float(dom_freq), 2) if dom_freq is not None else None

        inliers = kinematics.inlier_feature_count
        detected = max(50, kinematics.valid_feature_count)
        rejected = max(0, detected - inliers)
        retention = round((inliers / detected) * 100.0, 1) if detected > 0 else 0.0

        # Frequency deviation from baseline if baseline established
        baseline_dev = None
        with self._lock:
            if self._baseline_record and dom_freq is not None:
                base_f = self._baseline_record.get("baseline_frequency_hz")
                if base_f is not None:
                    baseline_dev = round(dom_freq - base_f, 2)

        return {
            "timestamp": round(kinematics.timestamp, 3),
            "fps": round(fps, 1),
            "frame_width": self.cam_mgr.actual_width if self.cam_mgr else 1280,
            "frame_height": self.cam_mgr.actual_height if self.cam_mgr else 720,
            "displacement_x_px": round(kinematics.cum_x_pixels, 3),
            "displacement_y_px": round(kinematics.cum_y_pixels, 3),
            "displacement_magnitude_px": round(kinematics.cum_displacement_pixels, 3),
            "velocity_x_px_s": round(kinematics.velocity_x_pixels_s, 2),
            "velocity_y_px_s": round(kinematics.velocity_y_pixels_s, 2),
            "velocity_magnitude_px_s": round(kinematics.velocity_magnitude_pixels_s, 2),
            "acceleration_x_px_s2": round(kinematics.acceleration_x_pixels_s2, 2),
            "acceleration_y_px_s2": round(kinematics.acceleration_y_pixels_s2, 2),
            "acceleration_magnitude_px_s2": round(kinematics.acceleration_magnitude_pixels_s2, 2),
            "dominant_frequency_hz": dom_freq,
            "feature_count": kinematics.valid_feature_count,
            "inlier_feature_count": inliers,
            "rejected_feature_count": rejected,
            "retention_rate": retention,
            "tracking_quality": kinematics.tracking_quality,
            "measurement_valid": kinematics.measurement_valid,
            "validity_reason": kinematics.validity_reason,
            "confidence": round(kinematics.confidence, 3),
            "mad_x_px": round(kinematics.mad_x_pixels, 3),
            "mad_y_px": round(kinematics.mad_y_pixels, 3),
            "baseline_frequency_hz": self._baseline_record.get("baseline_frequency_hz") if self._baseline_record else None,
            "frequency_deviation_hz": baseline_dev,
            "fft_frequencies": spectrum.get("frequencies", []),
            "fft_amplitudes": spectrum.get("amplitudes", []),
            "peak_frequency_hz": spectrum.get("peak_frequency_hz"),
            "peak_amplitude": spectrum.get("peak_amplitude"),
            "is_live": True,
        }
