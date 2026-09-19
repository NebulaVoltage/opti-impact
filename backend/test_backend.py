"""Unit tests for the optical monitoring FastAPI backend and worker adapter."""

import json
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.worker import OpticalPipelineWorker
from optical.optical_motion import OpticalKinematics, OpticalMotionEstimator


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_api_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "camera_connected" in data
    assert "camera_index" in data
    assert "frame_count" in data


def test_api_telemetry_endpoint(client):
    response = client.get("/api/telemetry")
    # May return 200 with data or 204 if waiting for first frame
    assert response.status_code in [200, 204]
    if response.status_code == 200:
        data = response.json()
        assert "timestamp" in data
        assert "displacement_magnitude_px" in data


def test_api_spectrum_endpoint(client):
    response = client.get("/api/spectrum")
    assert response.status_code == 200
    data = response.json()
    assert "spectrum" in data
    assert "spectrogram" in data


def test_api_reset_endpoint(client):
    response = client.post("/api/reset")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_api_baseline_endpoints(client):
    # Retrieve baseline
    response = client.get("/api/baseline")
    assert response.status_code == 200


def test_worker_fft_spectrum_calculation():
    worker = OpticalPipelineWorker(camera_index=99)
    estimator = OpticalMotionEstimator()

    # Feed synthetic 5.0 Hz oscillation into estimator
    t = 0.0
    dt = 1.0 / 30.0
    for i in range(64):
        t += dt
        disp = 10.0 * np.sin(2.0 * np.pi * 5.0 * t)
        estimator._timestamps.append(t)
        estimator._cum_x_hist.append(disp)
        estimator._cum_y_hist.append(0.2 * disp)
        estimator._valid_flags.append(True)

    spectrum = worker._compute_fft_spectrum(estimator)
    assert len(spectrum["frequencies"]) > 0
    assert len(spectrum["amplitudes"]) > 0
    assert spectrum["peak_frequency_hz"] is not None
    # Estimated peak should be approximately 5.0 Hz (within spectral resolution)
    assert abs(spectrum["peak_frequency_hz"] - 5.0) <= 0.6


def test_worker_telemetry_schema_integrity():
    worker = OpticalPipelineWorker(camera_index=99)
    kinematics = OpticalKinematics(
        timestamp=12.345,
        frame_index=100,
        dx_pixels=0.5,
        dy_pixels=-0.2,
        displacement_pixels=0.54,
        cum_x_pixels=2.5,
        cum_y_pixels=-1.1,
        cum_displacement_pixels=2.73,
        velocity_x_pixels_s=15.2,
        velocity_y_pixels_s=-4.1,
        velocity_magnitude_pixels_s=15.74,
        acceleration_x_pixels_s2=45.2,
        acceleration_y_pixels_s2=-12.0,
        acceleration_magnitude_pixels_s2=46.76,
        dominant_frequency_hz=4.85,
        valid_feature_count=32,
        tracking_quality="GOOD",
        measurement_valid=True,
        validity_reason="VALID",
        confidence=0.92,
        mad_x_pixels=0.04,
        mad_y_pixels=0.02,
        inlier_feature_count=30,
    )

    dummy_spectrum = {
        "frequencies": [1.0, 2.0, 3.0, 4.0, 5.0],
        "amplitudes": [0.1, 0.2, 0.3, 0.8, 0.2],
        "peak_frequency_hz": 4.0,
        "peak_amplitude": 0.8,
    }

    packet = worker._build_telemetry_packet(kinematics, fps=30.1, spectrum=dummy_spectrum)

    required_keys = [
        "timestamp",
        "fps",
        "frame_width",
        "frame_height",
        "displacement_x_px",
        "displacement_y_px",
        "displacement_magnitude_px",
        "velocity_x_px_s",
        "velocity_y_px_s",
        "velocity_magnitude_px_s",
        "acceleration_x_px_s2",
        "acceleration_y_px_s2",
        "acceleration_magnitude_px_s2",
        "dominant_frequency_hz",
        "feature_count",
        "inlier_feature_count",
        "rejected_feature_count",
        "retention_rate",
        "tracking_quality",
        "measurement_valid",
        "validity_reason",
        "confidence",
        "mad_x_px",
        "mad_y_px",
        "fft_frequencies",
        "fft_amplitudes",
        "peak_frequency_hz",
        "peak_amplitude",
        "is_live",
    ]

    for k in required_keys:
        assert k in packet, f"Missing key {k} in packet"

    assert packet["is_live"] is True
    assert packet["tracking_quality"] == "GOOD"
    assert packet["dominant_frequency_hz"] == 4.85
    assert packet["inlier_feature_count"] == 30
    assert packet["rejected_feature_count"] == 20
    assert packet["retention_rate"] == 60.0
