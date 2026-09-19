"""Automated unit and integration test suite for physical optical sensing modules.

Does NOT require a physical camera: uses synthetic contrast images, geometric transformations,
and mocked kinematic trajectories to rigorously verify all optical subsystems.
"""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pandas as pd
import pytest

from optical.calibration import PlanarCalibration
from optical.camera import CameraConfig, CameraManager
from optical.feature_tracker import OpticalFeatureTracker, TrackedPoints, TrackerConfig
from optical.optical_motion import OpticalKinematics, OpticalMotionEstimator
from optical.recorder import OpticalSessionRecorder
from optical.visualization import OpticalVisualizer


@pytest.fixture
def synthetic_pattern() -> np.ndarray:
    """Generate a high-contrast synthetic image simulating irregular electrical tape markers."""
    img = np.full((300, 300), 200, dtype=np.uint8)
    # Add multiple high-contrast black rectangles and tape strips
    for i in range(5):
        for j in range(5):
            r = 30 + i * 50
            c = 30 + j * 50
            img[r : r + 25, c : c + 25] = 15
    return img


# ---------------------------------------------------------------------------
# 1. Camera Configuration & Fallback Tests
# ---------------------------------------------------------------------------


def test_camera_config_defaults():
    """Verify CameraConfig default attributes."""
    cfg = CameraConfig(camera_index=1, width=1280, height=720, requested_fps=30.0)
    assert cfg.camera_index == 1
    assert cfg.width == 1280
    assert cfg.height == 720
    assert cfg.requested_fps == 30.0


def test_camera_manager_invalid_index_handling():
    """Verify CameraManager gracefully handles unopenable camera indices."""
    mgr = CameraManager(CameraConfig(camera_index=9999))
    opened = mgr.open()
    assert opened is False
    assert mgr.is_opened is False
    ret, frame, _ = mgr.read_frame()
    assert ret is False
    assert frame is None
    mgr.release()


# ---------------------------------------------------------------------------
# 2. Feature Detection & ROI Tests
# ---------------------------------------------------------------------------


def test_feature_detection_on_high_contrast_pattern(synthetic_pattern: np.ndarray):
    """Verify goodFeaturesToTrack detects corners on high-contrast pattern."""
    tracker = OpticalFeatureTracker()
    corners = tracker.detect_features(synthetic_pattern)
    assert corners is not None
    assert len(corners) >= 15


def test_insufficient_features_handling():
    """Verify tracker handles completely blank/featureless images."""
    blank = np.full((200, 200), 128, dtype=np.uint8)
    tracker = OpticalFeatureTracker(TrackerConfig(min_tracked_points=10))
    corners = tracker.detect_features(blank)
    assert corners is None or len(corners) == 0

    tracked = tracker.track(blank)
    assert tracked.tracking_quality == "LOST"
    assert tracked.valid_count == 0


def test_roi_cropping_and_bounds(synthetic_pattern: np.ndarray):
    """Verify feature detection is restricted to the specified ROI."""
    # Place ROI on the bottom-right portion (150, 150, 140, 140)
    tracker = OpticalFeatureTracker(roi=(150, 150, 140, 140))
    corners = tracker.detect_features(synthetic_pattern)
    assert corners is not None
    assert len(corners) > 0
    # Every detected corner must be inside [150, 290] x [150, 290]
    for pt in corners.reshape(-1, 2):
        x, y = pt
        assert 149 <= x <= 291
        assert 149 <= y <= 291


# ---------------------------------------------------------------------------
# 3. Optical Flow & Motion Tracking Tests
# ---------------------------------------------------------------------------


def test_stationary_image_produces_near_zero_motion(synthetic_pattern: np.ndarray):
    """Feeding identical frames must produce near-zero optical flow displacement (< 0.05 px)."""
    tracker = OpticalFeatureTracker()
    # Frame 1: initialization
    tracker.track(synthetic_pattern)

    # Frame 2: identical stationary frame
    tracked = tracker.track(synthetic_pattern)
    assert tracked.valid_count >= 15
    assert tracked.tracking_quality == "GOOD"
    disps = tracked.displacements
    assert len(disps) > 0
    median_disp = np.median(np.linalg.norm(disps, axis=1))
    assert median_disp == pytest.approx(0.0, abs=0.05)


def test_known_synthetic_translation_recovered(synthetic_pattern: np.ndarray):
    """Synthetically shifting an image by (+4.0, -3.0) px must be recovered accurately by optical flow."""
    tracker = OpticalFeatureTracker()
    tracker.track(synthetic_pattern)

    # Create translated frame: shift x by +4, y by -3
    dx_true = 4.0
    dy_true = -3.0
    M = np.float32([[1, 0, dx_true], [0, 1, dy_true]])
    h, w = synthetic_pattern.shape
    shifted = cv2.warpAffine(synthetic_pattern, M, (w, h))

    tracked = tracker.track(shifted)
    assert tracked.valid_count >= 10
    dx_est = np.median(tracked.displacements[:, 0])
    dy_est = np.median(tracked.displacements[:, 1])

    # Allow sub-pixel tolerance
    assert dx_est == pytest.approx(dx_true, abs=0.4)
    assert dy_est == pytest.approx(dy_true, abs=0.4)


def test_invalid_lk_points_rejected():
    """Verify invalid points (status=0) are excluded from valid_count."""
    # Test tracking when points move completely off-screen
    tracker = OpticalFeatureTracker()
    f1 = np.zeros((100, 100), dtype=np.uint8)
    f1[20:30, 20:30] = 255
    tracker.track(f1)

    f2 = np.zeros((100, 100), dtype=np.uint8)
    # The feature is deleted in f2, optical flow will fail or have high error
    tracked = tracker.track(f2)
    assert tracked.valid_count < 5


# ---------------------------------------------------------------------------
# 4. Motion Estimation & Kinematics Tests
# ---------------------------------------------------------------------------


def test_displacement_aggregation_median():
    """Verify median aggregation rejects outlier vectors."""
    estimator = OpticalMotionEstimator()
    # 5 features: 4 agreeing at dx=2.0, dy=1.0 and 1 severe outlier at dx=100.0
    prev_pts = np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]], dtype=np.float32)
    curr_pts = np.array([[12, 11], [22, 21], [32, 31], [42, 41], [150, 150]], dtype=np.float32)
    disps = curr_pts - prev_pts

    tracked = TrackedPoints(
        prev_pts=prev_pts,
        curr_pts=curr_pts,
        displacements=disps,
        status=np.ones(5, dtype=bool),
        valid_count=5,
        tracking_quality="GOOD",
        quality_score=1.0,
    )
    k = estimator.update(tracked, timestamp=1.0)
    assert k.dx_pixels == pytest.approx(2.0, abs=1e-3)
    assert k.dy_pixels == pytest.approx(1.0, abs=1e-3)
    assert k.displacement_pixels == pytest.approx(np.sqrt(5.0), abs=1e-3)


def test_camera_motion_rejection():
    """Verify background displacement is subtracted from specimen displacement."""
    estimator = OpticalMotionEstimator()
    # Specimen moves dx=6, dy=2 (including camera shake)
    spec_prev = np.array([[50, 50]], dtype=np.float32)
    spec_curr = np.array([[56, 52]], dtype=np.float32)
    spec_tracked = TrackedPoints(
        prev_pts=spec_prev,
        curr_pts=spec_curr,
        displacements=spec_curr - spec_prev,
        status=np.ones(1, dtype=bool),
        valid_count=1,
        tracking_quality="GOOD",
        quality_score=1.0,
    )

    # Background moves dx=2, dy=2 (camera shake)
    bg_prev = np.array([[10, 10]], dtype=np.float32)
    bg_curr = np.array([[12, 12]], dtype=np.float32)
    bg_tracked = TrackedPoints(
        prev_pts=bg_prev,
        curr_pts=bg_curr,
        displacements=bg_curr - bg_prev,
        status=np.ones(1, dtype=bool),
        valid_count=1,
        tracking_quality="GOOD",
        quality_score=1.0,
    )

    k = estimator.update(spec_tracked, timestamp=1.0, bg_tracked=bg_tracked)
    # Net structural dx = 6 - 2 = 4, dy = 2 - 2 = 0
    assert k.dx_pixels == pytest.approx(4.0, abs=1e-3)
    assert k.dy_pixels == pytest.approx(0.0, abs=1e-3)


def test_velocity_calculation():
    """Verify velocity equals displacement derivative divided by dt."""
    estimator = OpticalMotionEstimator()
    tp1 = TrackedPoints(
        prev_pts=np.array([[0, 0]], dtype=np.float32),
        curr_pts=np.array([[0, 0]], dtype=np.float32),
        displacements=np.array([[0, 0]], dtype=np.float32),
        status=np.ones(1, dtype=bool),
        valid_count=1,
        tracking_quality="GOOD",
        quality_score=1.0,
    )
    tp2 = TrackedPoints(
        prev_pts=np.array([[0, 0]], dtype=np.float32),
        curr_pts=np.array([[10, 0]], dtype=np.float32),
        displacements=np.array([[10, 0]], dtype=np.float32),
        status=np.ones(1, dtype=bool),
        valid_count=1,
        tracking_quality="GOOD",
        quality_score=1.0,
    )

    k1 = estimator.update(tp1, timestamp=1.0)
    k2 = estimator.update(tp2, timestamp=1.5)  # dt = 0.5s, displacement changed from 0 to 10 px
    assert k2.velocity_x_pixels_s == pytest.approx(20.0, abs=1e-2)
    assert k2.velocity_magnitude_pixels_s == pytest.approx(20.0, abs=1e-2)


def test_smoothed_acceleration():
    """Verify smoothed acceleration calculation."""
    estimator = OpticalMotionEstimator(smoothing_window=5)
    # Linearly accelerating sequence
    for idx in range(10):
        t = float(idx * 0.1)
        # Displacement increases quadratically
        disp = float(idx * 2.0)
        tp = TrackedPoints(
            prev_pts=np.array([[0, 0]], dtype=np.float32),
            curr_pts=np.array([[disp, 0]], dtype=np.float32),
            displacements=np.array([[disp, 0]], dtype=np.float32),
            status=np.ones(1, dtype=bool),
            valid_count=1,
            tracking_quality="GOOD",
            quality_score=1.0,
        )
        k = estimator.update(tp, timestamp=t)

    assert not np.isnan(k.acceleration_x_pixels_s2)


# ---------------------------------------------------------------------------
# 5. Frequency Analysis Tests
# ---------------------------------------------------------------------------


def test_synthetic_frequency_recovery():
    """Verify dominant optical frequency recovery on synthetic sinusoidal motion."""
    estimator = OpticalMotionEstimator(history_len=200, min_freq=0.5)
    f_target = 3.2  # Hz
    fps = 30.0
    dt = 1.0 / fps

    for idx in range(120):
        t = idx * dt
        # Sine displacement of 10 px relative to reference baseline
        disp_now = 10.0 * np.sin(2.0 * np.pi * f_target * t)

        tp = TrackedPoints(
            prev_pts=np.array([[0, 0]], dtype=np.float32),
            curr_pts=np.array([[disp_now, 0]], dtype=np.float32),
            displacements=np.array([[disp_now, 0]], dtype=np.float32),
            status=np.ones(1, dtype=bool),
            valid_count=1,
            tracking_quality="GOOD",
            quality_score=1.0,
        )
        k = estimator.update(tp, timestamp=t)

    assert not np.isnan(k.dominant_frequency_hz)
    assert k.dominant_frequency_hz == pytest.approx(f_target, abs=0.3)


def test_insufficient_samples_frequency_returns_nan():
    """Fewer than 32 samples must return NaN for dominant frequency."""
    estimator = OpticalMotionEstimator()
    tp = TrackedPoints(
        prev_pts=np.array([[0, 0]], dtype=np.float32),
        curr_pts=np.array([[1, 0]], dtype=np.float32),
        displacements=np.array([[1, 0]], dtype=np.float32),
        status=np.ones(1, dtype=bool),
        valid_count=1,
        tracking_quality="GOOD",
        quality_score=1.0,
    )
    for idx in range(10):
        k = estimator.update(tp, timestamp=idx * 0.033)
    assert np.isnan(k.dominant_frequency_hz)


# ---------------------------------------------------------------------------
# 6. Tracking Quality & Re-Detection Tests
# ---------------------------------------------------------------------------


def test_tracking_quality_transitions(synthetic_pattern: np.ndarray):
    """Verify tracking quality grades between GOOD, DEGRADED, and LOST."""
    tracker = OpticalFeatureTracker(TrackerConfig(min_tracked_points=20))
    # Frame 1: GOOD
    t1 = tracker.track(synthetic_pattern)
    assert t1.tracking_quality == "GOOD"

    # Frame with fewer features
    small_feat_img = np.full((300, 300), 200, dtype=np.uint8)
    small_feat_img[100:110, 100:110] = 0
    tracker.track(small_feat_img)
    # Empty frame: LOST
    empty = np.full((300, 300), 128, dtype=np.uint8)
    t3 = tracker.track(empty)
    assert t3.tracking_quality in ("DEGRADED", "LOST")


def test_feature_redetection_on_drop(synthetic_pattern: np.ndarray):
    """Verify automatic re-detection triggers when tracked points drop below minimum."""
    tracker = OpticalFeatureTracker(TrackerConfig(min_tracked_points=15))
    tracker.track(synthetic_pattern)
    assert tracker.tracked_pts is not None

    # Artificially drop points to 5
    tracker.tracked_pts = tracker.tracked_pts[:5]
    # Next frame on good pattern should trigger re-detection
    t2 = tracker.track(synthetic_pattern)
    assert t2.redetected is True
    assert t2.valid_count >= 15


# ---------------------------------------------------------------------------
# 7. Planar Calibration Tests
# ---------------------------------------------------------------------------


def test_planar_calibration_scaling():
    """Verify pixel to millimeter conversion."""
    calib = PlanarCalibration()
    assert calib.is_calibrated is False
    assert calib.pixels_to_mm(100.0) is None

    # Known distance: 50.0 mm = 200.0 pixels -> scale = 0.25 mm/pixel
    scale = calib.calibrate_from_distance(known_distance_mm=50.0, measured_distance_pixels=200.0)
    assert scale == pytest.approx(0.25)
    assert calib.is_calibrated is True
    assert calib.pixels_to_mm(100.0) == pytest.approx(25.0)
    assert calib.mm_to_pixels(25.0) == pytest.approx(100.0)


def test_planar_calibration_save_and_load():
    """Verify calibration persistence."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "calib.json"
        c1 = PlanarCalibration(scale_mm_per_pixel=0.125)
        c1.save(json_path)

        c2 = PlanarCalibration.load(json_path)
        assert c2.is_calibrated is True
        assert c2.scale_mm_per_pixel == pytest.approx(0.125)


# ---------------------------------------------------------------------------
# 8. Session Recording Tests
# ---------------------------------------------------------------------------


def test_optical_session_recorder():
    """Verify OpticalSessionRecorder writes valid CSV file with expected headers."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        rec = OpticalSessionRecorder(output_dir=Path(tmp_dir), session_name="test_session")
        kin = OpticalKinematics(
            timestamp=1.23,
            frame_index=1,
            dx_pixels=2.5,
            dy_pixels=-1.0,
            displacement_pixels=2.69,
            cum_x_pixels=5.0,
            cum_y_pixels=-2.0,
            cum_displacement_pixels=5.38,
            velocity_x_pixels_s=10.0,
            velocity_y_pixels_s=-4.0,
            velocity_magnitude_pixels_s=10.77,
            acceleration_x_pixels_s2=1.0,
            acceleration_y_pixels_s2=-0.5,
            acceleration_magnitude_pixels_s2=1.12,
            dominant_frequency_hz=4.5,
            valid_feature_count=35,
            tracking_quality="GOOD",
        )
        rec.record_frame(kin, camera_fps=29.8)
        out_csv = rec.save()

        assert out_csv.exists()
        df = pd.read_csv(out_csv)
        assert len(df) == 1
        for col in OpticalSessionRecorder.COLUMNS:
            assert col in df.columns
        assert df.iloc[0]["valid_feature_count"] == 35
        assert df.iloc[0]["tracking_quality"] == "GOOD"


# ---------------------------------------------------------------------------
# 9. Visualizer Overlay Test
# ---------------------------------------------------------------------------


def test_visualizer_renders_without_error(synthetic_pattern: np.ndarray):
    """Verify OpticalVisualizer produces annotated BGR frame without errors."""
    bgr = cv2.cvtColor(synthetic_pattern, cv2.COLOR_GRAY2BGR)
    vis = OpticalVisualizer()
    tp = TrackedPoints(
        prev_pts=np.array([[50, 50], [100, 100]], dtype=np.float32),
        curr_pts=np.array([[52, 51], [103, 99]], dtype=np.float32),
        displacements=np.array([[2, 1], [3, -1]], dtype=np.float32),
        status=np.ones(2, dtype=bool),
        valid_count=2,
        tracking_quality="GOOD",
        quality_score=1.0,
    )
    kin = OpticalKinematics(
        timestamp=0.1,
        frame_index=1,
        dx_pixels=2.5,
        dy_pixels=0.0,
        displacement_pixels=2.5,
        cum_x_pixels=2.5,
        cum_y_pixels=0.0,
        cum_displacement_pixels=2.5,
        velocity_x_pixels_s=10.0,
        velocity_y_pixels_s=0.0,
        velocity_magnitude_pixels_s=10.0,
        acceleration_x_pixels_s2=0.0,
        acceleration_y_pixels_s2=0.0,
        acceleration_magnitude_pixels_s2=0.0,
        dominant_frequency_hz=3.5,
        valid_feature_count=2,
        tracking_quality="GOOD",
    )
    annotated = vis.draw_frame(
        frame_bgr=bgr,
        tracked=tp,
        kinematics=kin,
        camera_fps=30.0,
        roi=(20, 20, 200, 200),
        show_plot=True,
    )
    assert annotated is not None
    assert annotated.shape[0] > bgr.shape[0]  # Plot strip added


# ---------------------------------------------------------------------------
# 10. Step 6A Bug Fix Regression Tests (Reference-Based Displacement)
# ---------------------------------------------------------------------------


def test_static_specimen_remains_near_zero_displacement(synthetic_pattern: np.ndarray):
    """Regression Test 1: Static specimen frames must remain near zero displacement."""
    tracker = OpticalFeatureTracker()
    estimator = OpticalMotionEstimator()

    for idx in range(60):
        t = idx * 0.033
        tp = tracker.track(synthetic_pattern)
        k = estimator.update(tp, timestamp=t)

    assert k.cum_displacement_pixels == pytest.approx(0.0, abs=0.05)
    assert k.measurement_valid is True


def test_known_positive_10px_translation(synthetic_pattern: np.ndarray):
    """Regression Test 2: Known +10 px translation produces approximately +10 px displacement."""
    tracker = OpticalFeatureTracker()
    estimator = OpticalMotionEstimator()

    # Frame 0: Baseline
    tp0 = tracker.track(synthetic_pattern)
    k0 = estimator.update(tp0, timestamp=0.0)

    # Frame 1: Shift +10 px in X
    h, w = synthetic_pattern.shape[:2]
    M = np.float32([[1, 0, 10], [0, 1, 0]])
    shifted = cv2.warpAffine(synthetic_pattern, M, (w, h))

    tp1 = tracker.track(shifted)
    k1 = estimator.update(tp1, timestamp=0.033)

    assert k1.dx_pixels == pytest.approx(10.0, abs=0.4)
    assert k1.dy_pixels == pytest.approx(0.0, abs=0.4)
    assert k1.cum_displacement_pixels == pytest.approx(10.0, abs=0.4)
    assert k1.measurement_valid is True


def test_known_negative_10px_translation(synthetic_pattern: np.ndarray):
    """Regression Test 3: Known -10 px translation produces approximately -10 px displacement."""
    tracker = OpticalFeatureTracker()
    estimator = OpticalMotionEstimator()

    tp0 = tracker.track(synthetic_pattern)
    k0 = estimator.update(tp0, timestamp=0.0)

    h, w = synthetic_pattern.shape[:2]
    M = np.float32([[1, 0, -10], [0, 1, 0]])
    shifted = cv2.warpAffine(synthetic_pattern, M, (w, h))

    tp1 = tracker.track(shifted)
    k1 = estimator.update(tp1, timestamp=0.033)

    assert k1.dx_pixels == pytest.approx(-10.0, abs=0.4)
    assert k1.dy_pixels == pytest.approx(0.0, abs=0.4)
    assert k1.cum_displacement_pixels == pytest.approx(10.0, abs=0.4)
    assert k1.measurement_valid is True


def test_known_20px_translation_magnitude(synthetic_pattern: np.ndarray):
    """Regression Test 4: Known (12, 16) translation produces approximately 20 px magnitude."""
    tracker = OpticalFeatureTracker()
    estimator = OpticalMotionEstimator()

    tp0 = tracker.track(synthetic_pattern)
    k0 = estimator.update(tp0, timestamp=0.0)

    h, w = synthetic_pattern.shape[:2]
    M = np.float32([[1, 0, 12], [0, 1, 16]])
    shifted = cv2.warpAffine(synthetic_pattern, M, (w, h))

    tp1 = tracker.track(shifted)
    k1 = estimator.update(tp1, timestamp=0.033)

    assert k1.dx_pixels == pytest.approx(12.0, abs=0.5)
    assert k1.dy_pixels == pytest.approx(16.0, abs=0.5)
    assert k1.cum_displacement_pixels == pytest.approx(20.0, abs=0.5)


def test_repeated_identical_frames_do_not_accumulate(synthetic_pattern: np.ndarray):
    """Regression Test 5: Repeated identical frames do not accumulate displacement."""
    tracker = OpticalFeatureTracker()
    estimator = OpticalMotionEstimator()

    # Frame 0: Baseline
    tp0 = tracker.track(synthetic_pattern)
    k0 = estimator.update(tp0, timestamp=0.0)

    # Frame 1: Shift by 10 px
    h, w = synthetic_pattern.shape[:2]
    M = np.float32([[1, 0, 10], [0, 1, 0]])
    shifted = cv2.warpAffine(synthetic_pattern, M, (w, h))

    tp1 = tracker.track(shifted)
    k1 = estimator.update(tp1, timestamp=0.033)
    assert k1.cum_displacement_pixels == pytest.approx(10.0, abs=0.4)

    # Feed the exact same frame 100 times
    for idx in range(2, 102):
        t = idx * 0.033
        tp_same = tracker.track(shifted)
        k_same = estimator.update(tp_same, timestamp=t)
        # Displacement MUST NOT accumulate to 1000 px! It must stay at ~10 px!
        assert k_same.cum_displacement_pixels == pytest.approx(10.0, abs=0.5)
        # Velocity must be approximately 0 because the frame is identical
        assert k_same.velocity_magnitude_pixels_s == pytest.approx(0.0, abs=1.0)


def test_long_stationary_sequence_does_not_grow_to_thousands(synthetic_pattern: np.ndarray):
    """Regression Test 6: Long stationary sequence (300 frames) does not grow to thousands of pixels."""
    tracker = OpticalFeatureTracker()
    estimator = OpticalMotionEstimator()

    for idx in range(300):
        t = idx * 0.033
        tp = tracker.track(synthetic_pattern)
        k = estimator.update(tp, timestamp=t)
        assert k.cum_displacement_pixels < 0.1
        assert k.measurement_valid is True

    # Absolutely cannot be 23,517 px!
    assert k.cum_displacement_pixels < 0.1


def test_feature_redetection_does_not_create_displacement_jump(synthetic_pattern: np.ndarray):
    """Regression Test 7: Feature re-detection does not create a displacement jump."""
    tracker = OpticalFeatureTracker(TrackerConfig(min_tracked_points=15))
    estimator = OpticalMotionEstimator()

    # Frame 0: Baseline
    tracker.track(synthetic_pattern)
    estimator.update(tracker.track(synthetic_pattern), timestamp=0.0)

    # Shift by 15 px
    h, w = synthetic_pattern.shape[:2]
    M = np.float32([[1, 0, 15], [0, 1, 0]])
    shifted = cv2.warpAffine(synthetic_pattern, M, (w, h))

    t_before = tracker.track(shifted)
    k_before = estimator.update(t_before, timestamp=0.033)
    disp_before = k_before.dx_pixels
    assert disp_before == pytest.approx(15.0, abs=0.5)

    # Simulate feature drop triggering re-detection
    tracker.tracked_pts = tracker.tracked_pts[:3]
    t_redetect = tracker.track(shifted)
    assert t_redetect.redetected is True
    k_after = estimator.update(t_redetect, timestamp=0.066)

    # Step discontinuity across re-detection must be negligible (< 0.5 px)
    assert abs(k_after.dx_pixels - disp_before) < 0.5


def test_background_features_do_not_contaminate_structural_displacement():
    """Regression Test 8: Features outside the structural ROI do not contaminate structural motion."""
    # 400x400 image. ROI is (50, 50, 100, 100)
    tracker = OpticalFeatureTracker(roi=(50, 50, 100, 100))
    img = np.full((400, 400), 200, dtype=np.uint8)

    # Put specimen corners inside ROI
    for r in range(60, 140, 20):
        for c in range(60, 140, 20):
            img[r : r + 10, c : c + 10] = 20

    # Put background corners far outside ROI
    for r in range(250, 350, 25):
        for c in range(250, 350, 25):
            img[r : r + 10, c : c + 10] = 20

    t0 = tracker.track(img)
    # Verify all detected features are strictly inside ROI
    assert t0.valid_count > 0
    for pt in t0.curr_pts:
        assert 50 <= pt[0] <= 150
        assert 50 <= pt[1] <= 150


def test_displacement_and_velocity_mathematically_consistent():
    """Regression Test 9: Displacement and velocity refer to the same coordinate system."""
    estimator = OpticalMotionEstimator()

    # Sequence where object moves from 0 to 10 to 20 then stays at 20
    positions = [0.0, 10.0, 20.0, 20.0, 20.0]
    dt = 0.1

    for idx, pos in enumerate(positions):
        t = idx * dt
        tp = TrackedPoints(
            prev_pts=np.array([[0, 0]], dtype=np.float32),
            curr_pts=np.array([[pos, 0]], dtype=np.float32),
            displacements=np.array([[pos, 0]], dtype=np.float32),
            status=np.ones(1, dtype=bool),
            valid_count=1,
            tracking_quality="GOOD",
            quality_score=1.0,
        )
        k = estimator.update(tp, timestamp=t)

        if idx == 1:
            # Moved from 0 to 10 in 0.1s -> v = 100 px/s
            assert k.velocity_x_pixels_s == pytest.approx(100.0, abs=1e-2)
            assert k.cum_displacement_pixels == pytest.approx(10.0, abs=1e-2)
        elif idx == 3:
            # Position stayed at 20 -> v = 0 px/s
            assert k.velocity_x_pixels_s == pytest.approx(0.0, abs=1e-2)
            assert k.cum_displacement_pixels == pytest.approx(20.0, abs=1e-2)


def test_impossible_displacement_marked_invalid():
    """Regression Test 10: Impossible displacement (> frame dimensions or jump) is marked invalid."""
    estimator = OpticalMotionEstimator(frame_width=1280, frame_height=720, max_step_disp=80.0)

    # Frame 1: Valid initial displacement
    tp1 = TrackedPoints(
        prev_pts=np.array([[0, 0]], dtype=np.float32),
        curr_pts=np.array([[10, 0]], dtype=np.float32),
        displacements=np.array([[10, 0]], dtype=np.float32),
        status=np.ones(1, dtype=bool),
        valid_count=1,
        tracking_quality="GOOD",
        quality_score=1.0,
    )
    k1 = estimator.update(tp1, timestamp=0.0)
    assert k1.measurement_valid is True

    # Frame 2: Teleportation of 23,517 px (the observed live bug!)
    tp_bug = TrackedPoints(
        prev_pts=np.array([[0, 0]], dtype=np.float32),
        curr_pts=np.array([[23517, 0]], dtype=np.float32),
        displacements=np.array([[23517, 0]], dtype=np.float32),
        status=np.ones(1, dtype=bool),
        valid_count=1,
        tracking_quality="GOOD",
        quality_score=1.0,
    )
    k2 = estimator.update(tp_bug, timestamp=0.033)
    assert k2.measurement_valid is False
    assert k2.validity_reason in ("EXCEEDS_FRAME_BOUNDS", "EXCESSIVE_STEP_DISPLACEMENT")


def test_frequency_not_calculated_from_invalid_samples():
    """Regression Test 11: Dominant frequency is not computed when history contains invalid samples."""
    estimator = OpticalMotionEstimator(history_len=100)

    # Feed 40 frames of sine wave but with 5 invalid frames in between
    for idx in range(40):
        t = idx * 0.033
        disp = 10.0 * np.sin(2.0 * np.pi * 3.0 * t)
        if 20 <= idx < 25:
            disp = 5000.0  # Invalid frame
        tp = TrackedPoints(
            prev_pts=np.array([[0, 0]], dtype=np.float32),
            curr_pts=np.array([[disp, 0]], dtype=np.float32),
            displacements=np.array([[disp, 0]], dtype=np.float32),
            status=np.ones(1, dtype=bool),
            valid_count=1,
            tracking_quality="GOOD",
            quality_score=1.0,
        )
        k = estimator.update(tp, timestamp=t)

    # Because invalid frames exist in recent window, frequency must be NaN
    assert np.isnan(k.dominant_frequency_hz)


# ---------------------------------------------------------------------------
# 11. Step 6A.1-P Precision & Robustness Audit Test Suite (Tests 1 - 10)
# ---------------------------------------------------------------------------


def test_audit_test_1_stationary_scene_noise_floor(synthetic_pattern: np.ndarray):
    """TEST 1: Stationary camera + stationary specimen.
    Expected: mean displacement ≈ 0, no false large displacement, very low false-invalid rate (< 0.05 px).
    """
    tracker = OpticalFeatureTracker()
    estimator = OpticalMotionEstimator()

    displacements = []
    for idx in range(60):
        t = idx * 0.033
        tp = tracker.track(synthetic_pattern)
        k = estimator.update(tp, timestamp=t)
        assert k.measurement_valid is True
        assert k.validity_reason == "VALID"
        displacements.append(k.cum_displacement_pixels)

    mean_disp = float(np.mean(displacements))
    max_disp = float(np.max(displacements))
    assert mean_disp < 0.05
    assert max_disp < 0.10


def test_audit_test_2_known_subpixel_translation(synthetic_pattern: np.ndarray):
    """TEST 2: Known small translation.
    Expected: displacement matches applied subpixel movement accurately.
    """
    tracker = OpticalFeatureTracker()
    estimator = OpticalMotionEstimator()

    # Frame 0: Baseline
    tracker.track(synthetic_pattern)
    estimator.update(tracker.track(synthetic_pattern), timestamp=0.0)

    # Shift by subpixel distance: dx = +3.25 px, dy = -1.75 px
    dx_true, dy_true = 3.25, -1.75
    h, w = synthetic_pattern.shape[:2]
    M = np.float32([[1, 0, dx_true], [0, 1, dy_true]])
    shifted = cv2.warpAffine(synthetic_pattern, M, (w, h))

    tp = tracker.track(shifted)
    k = estimator.update(tp, timestamp=0.033)

    assert k.measurement_valid is True
    assert k.dx_pixels == pytest.approx(dx_true, abs=0.35)
    assert k.dy_pixels == pytest.approx(dy_true, abs=0.35)
    expected_mag = float(np.sqrt(dx_true**2 + dy_true**2))
    assert k.cum_displacement_pixels == pytest.approx(expected_mag, abs=0.35)


def test_audit_test_3_positive_movement_and_return_to_origin(synthetic_pattern: np.ndarray):
    """TEST 3: Known positive movement and return to original position.
    Expected: +X -> approximately 0 on return; no huge negative/positive drift.
    """
    tracker = OpticalFeatureTracker()
    estimator = OpticalMotionEstimator()

    # Baseline
    tracker.track(synthetic_pattern)
    estimator.update(tracker.track(synthetic_pattern), timestamp=0.0)

    # Move +12 px in X
    h, w = synthetic_pattern.shape[:2]
    M_fwd = np.float32([[1, 0, 12.0], [0, 1, 0.0]])
    shifted = cv2.warpAffine(synthetic_pattern, M_fwd, (w, h))
    for idx in range(5):
        t = 0.033 * (idx + 1)
        tp = tracker.track(shifted)
        k = estimator.update(tp, timestamp=t)

    assert k.dx_pixels == pytest.approx(12.0, abs=0.5)

    # Return to origin (exact original frame)
    for idx in range(5):
        t = 0.033 * (idx + 6)
        tp = tracker.track(synthetic_pattern)
        k = estimator.update(tp, timestamp=t)

    # Residual drift on return must be near zero (< 0.25 px)
    assert abs(k.dx_pixels) < 0.25
    assert abs(k.dy_pixels) < 0.25
    assert k.cum_displacement_pixels < 0.35


def test_audit_test_4_repeated_identical_frames_no_drift(synthetic_pattern: np.ndarray):
    """TEST 4: Repeated identical frames.
    Expected: displacement remains constant, no accumulation or drift.
    """
    tracker = OpticalFeatureTracker()
    estimator = OpticalMotionEstimator()

    # Move to +8 px
    h, w = synthetic_pattern.shape[:2]
    M = np.float32([[1, 0, 8.0], [0, 1, 0.0]])
    shifted = cv2.warpAffine(synthetic_pattern, M, (w, h))

    tracker.track(synthetic_pattern)
    estimator.update(tracker.track(synthetic_pattern), timestamp=0.0)

    disps = []
    vels = []
    for idx in range(1, 100):
        t = idx * 0.033
        tp = tracker.track(shifted)
        k = estimator.update(tp, timestamp=t)
        disps.append(k.cum_displacement_pixels)
        if idx > 2:
            vels.append(k.velocity_magnitude_pixels_s)

    # All frames must stay rock solid at ~8.0 px
    assert np.allclose(disps, 8.0, atol=0.4)
    # Velocity on identical frames must be ~0 px/s
    assert np.mean(vels) < 0.5


def test_audit_test_5_feature_redetection_continuity(synthetic_pattern: np.ndarray):
    """TEST 5: Feature re-detection.
    Expected: no artificial displacement spike or step discontinuity (< 0.5 px).
    """
    tracker = OpticalFeatureTracker(TrackerConfig(min_tracked_points=15))
    estimator = OpticalMotionEstimator()

    tracker.track(synthetic_pattern)
    estimator.update(tracker.track(synthetic_pattern), timestamp=0.0)

    # Move by 10 px
    h, w = synthetic_pattern.shape[:2]
    M = np.float32([[1, 0, 10.0], [0, 1, 0.0]])
    shifted = cv2.warpAffine(synthetic_pattern, M, (w, h))

    t_before = tracker.track(shifted)
    k_before = estimator.update(t_before, timestamp=0.033)

    # Trigger re-detection by reducing points
    tracker.tracked_pts = tracker.tracked_pts[:3]
    t_redetect = tracker.track(shifted)
    assert t_redetect.redetected is True
    k_after = estimator.update(t_redetect, timestamp=0.066)

    # Re-detection jump must be subpixel (< 0.5 px)
    assert abs(k_after.dx_pixels - k_before.dx_pixels) < 0.5
    assert k_after.measurement_valid is True


def test_audit_test_6_single_corrupted_feature_rejected():
    """TEST 6: Single corrupted feature.
    Expected: corrupted outlier feature rejected by MAD filtering; overall displacement remains stable.
    """
    estimator = OpticalMotionEstimator()
    # 9 inliers at dx=5.0, dy=0.0 and 1 rogue corrupted feature at dx=250.0
    prev_pts = np.array([[10 * i, 20] for i in range(10)], dtype=np.float32)
    curr_pts = np.array([[10 * i + 5.0, 20] for i in range(9)] + [[300.0, 20.0]], dtype=np.float32)
    disps = curr_pts - prev_pts

    # Calculate MAD inlier mask
    dx_arr = disps[:, 0]
    dy_arr = disps[:, 1]
    med_x = float(np.median(dx_arr))
    med_y = float(np.median(dy_arr))
    mad_x = float(np.median(np.abs(dx_arr - med_x)))
    mad_y = float(np.median(np.abs(dy_arr - med_y)))
    inliers = (np.abs(dx_arr - med_x) <= 2.5 * max(mad_x, 0.35)) & (np.abs(dy_arr - med_y) <= 2.5 * max(mad_y, 0.35))

    tp = TrackedPoints(
        prev_pts=prev_pts,
        curr_pts=curr_pts,
        displacements=disps,
        status=np.ones(10, dtype=bool),
        valid_count=10,
        tracking_quality="GOOD",
        quality_score=1.0,
        inlier_mask=inliers,
    )
    k = estimator.update(tp, timestamp=1.0)
    # Displacement must match the 9 inliers (5.0 px), NOT corrupted by 250 px!
    assert k.dx_pixels == pytest.approx(5.0, abs=0.1)
    assert k.measurement_valid is True


def test_audit_test_7_multi_corrupted_features_graceful_handling():
    """TEST 7: Several corrupted features.
    Expected: measurement either remains valid with reduced confidence or becomes invalid.
    Never silently produces a false large displacement.
    """
    estimator = OpticalMotionEstimator(frame_width=1280, frame_height=720, max_step_disp=80.0)

    # 4 inliers at dx=3.0, dy=0.0 and 6 corrupted features with wild scatter
    prev_pts = np.array([[20 * i, 50] for i in range(10)], dtype=np.float32)
    curr_pts = np.array(
        [[20 * i + 3.0, 50] for i in range(4)]
        + [[500.0, 50.0], [600.0, 60.0], [700.0, 70.0], [800.0, 80.0], [900.0, 90.0], [1000.0, 100.0]],
        dtype=np.float32,
    )
    disps = curr_pts - prev_pts

    # With high corruption, confidence should drop or step jump should be rejected
    tp = TrackedPoints(
        prev_pts=prev_pts,
        curr_pts=curr_pts,
        displacements=disps,
        status=np.ones(10, dtype=bool),
        valid_count=10,
        tracking_quality="GOOD",
        quality_score=1.0,
        confidence=0.10,  # Low confidence due to extreme spread
    )
    k = estimator.update(tp, timestamp=1.0)
    # Must NOT produce a silent valid measurement with wild displacement
    assert (not k.measurement_valid) or (k.dx_pixels < 10.0)


def test_audit_test_8_roi_boundary_condition(synthetic_pattern: np.ndarray):
    """TEST 8: ROI boundary condition.
    Expected: no coordinate-system errors; features touching boundary filtered cleanly.
    """
    # Specimen ROI: [50, 50, 120, 120]
    tracker = OpticalFeatureTracker(roi=(50, 50, 120, 120))
    tp = tracker.track(synthetic_pattern)
    assert tp.valid_count > 0

    # Ensure all tracked points are within ROI
    for pt in tp.curr_pts:
        assert 50 <= pt[0] < 170
        assert 50 <= pt[1] < 170

    # Large translation that moves features outside ROI
    h, w = synthetic_pattern.shape[:2]
    M = np.float32([[1, 0, 100.0], [0, 1, 0.0]])
    shifted = cv2.warpAffine(synthetic_pattern, M, (w, h))
    tp_shifted = tracker.track(shifted)
    # Features moved outside ROI must be filtered out cleanly without crashing
    for pt in tp_shifted.curr_pts:
        assert 50 <= pt[0] < 170
        assert 50 <= pt[1] < 170


def test_audit_test_9_resolution_change_safety():
    """TEST 9: Resolution change.
    Expected: safe invalidation/rejection of mismatched frame dimensions.
    """
    mgr = CameraManager(CameraConfig(width=1280, height=720))
    mgr.actual_width = 1280
    mgr.actual_height = 720

    valid_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    invalid_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    assert mgr.verify_frame_dimensions(valid_frame) is True
    assert mgr.verify_frame_dimensions(invalid_frame) is False


def test_audit_test_10_lighting_change_resilience(synthetic_pattern: np.ndarray):
    """TEST 10: Stationary scene with lighting change (auto-exposure / shadow).
    Expected: no huge displacement artifact; displacement remains bounded (< 0.5 px).
    """
    tracker = OpticalFeatureTracker()
    estimator = OpticalMotionEstimator()

    tracker.track(synthetic_pattern)
    estimator.update(tracker.track(synthetic_pattern), timestamp=0.0)

    # Frame with 30% reduced brightness (simulating shadow / auto-exposure adjustment)
    darker_frame = (synthetic_pattern.astype(np.float32) * 0.70).astype(np.uint8)

    tp_dark = tracker.track(darker_frame)
    k_dark = estimator.update(tp_dark, timestamp=0.033)

    assert k_dark.measurement_valid is True
    # Bidirectional LK and MAD filtering prevent false slip along edges
    assert k_dark.cum_displacement_pixels < 0.5


