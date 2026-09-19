import React from "react";

export const OpticalSensorIndicator: React.FC = () => {
    return (
        <div className="optical-sensor-card">
            <div className="sensor-card-header">
                <div className="sensor-icon">📷</div>
                <div className="sensor-title-group">
                    <div className="sensor-title">OPTICAL SENSING INTERFACE</div>
                    <div className="sensor-subtitle">Marker-Free Structural Computer Vision</div>
                </div>
                <div className="sensor-status-tag">
                    SIMULATION MODE
                </div>
            </div>

            <div className="sensor-specs-grid">
                <div className="spec-item">
                    <span className="spec-name">Tracking Principle</span>
                    <span className="spec-val">Shi–Tomasi Corners + Lucas–Kanade</span>
                </div>
                <div className="spec-item">
                    <span className="spec-name">Virtual Resolution</span>
                    <span className="spec-val">1280 × 720 HD</span>
                </div>
                <div className="spec-item">
                    <span className="spec-name">Sampling Cadence</span>
                    <span className="spec-val">30.0 FPS Quasi-Periodic</span>
                </div>
                <div className="spec-item">
                    <span className="spec-name">Hardware Link</span>
                    <span className="spec-val text-amber">Virtual Simulator Active</span>
                </div>
            </div>

            <div className="sensor-note">
                ℹ Physical USB webcam is disconnected during simulation mode. Telemetry mirrors the Step 6A optical kinematics contract for UI verification.
            </div>
        </div>
    );
};
