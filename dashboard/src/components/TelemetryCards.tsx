import React from "react";
import type { VisionTelemetry } from "../types/telemetry";

interface TelemetryCardsProps {
    telemetry: VisionTelemetry;
}

export const TelemetryCards: React.FC<TelemetryCardsProps> = ({ telemetry }) => {
    const trackingColor =
        telemetry.trackingQuality === "GOOD"
            ? "text-green"
            : telemetry.trackingQuality === "DEGRADED"
            ? "text-orange"
            : "text-red";

    const measurementColor = telemetry.measurementValid ? "text-green" : "text-red";

    const scenarioBadgeClass =
        telemetry.scenario === "CRITICAL"
            ? "badge-critical"
            : telemetry.scenario === "WARNING"
            ? "badge-warning"
            : "badge-normal";

    return (
        <div className="telemetry-cards-container">
            {/* 1. Structural Displacement Card */}
            <div className="telemetry-card">
                <div className="card-label">OPTICAL DISPLACEMENT</div>
                <div className="card-value highlight-cyan">
                    {telemetry.displacementMagnitude.toFixed(2)}
                    <span className="card-unit"> mm</span>
                </div>
                <div className="card-subtext">
                    X: {telemetry.displacementX > 0 ? "+" : ""}{telemetry.displacementX.toFixed(2)} mm &nbsp;|&nbsp;
                    Y: {telemetry.displacementY > 0 ? "+" : ""}{telemetry.displacementY.toFixed(2)} mm
                </div>
            </div>

            {/* 2. Optical Velocity Card */}
            <div className="telemetry-card">
                <div className="card-label">OPTICAL VELOCITY</div>
                <div className="card-value">
                    {telemetry.velocityMagnitude.toFixed(1)}
                    <span className="card-unit"> mm/s</span>
                </div>
                <div className="card-subtext">
                    Numerical finite difference Δx / Δt
                </div>
            </div>

            {/* 3. Optical Acceleration Card */}
            <div className="telemetry-card">
                <div className="card-label">OPTICAL ACCELERATION</div>
                <div className="card-value">
                    {telemetry.accelerationMagnitude.toFixed(2)}
                    <span className="card-unit"> m/s²</span>
                </div>
                <div className="card-subtext">
                    Derived smoothed d²x / dt²
                </div>
            </div>

            {/* 4. Resonant Frequency Card */}
            <div className="telemetry-card">
                <div className="card-label">DOMINANT FREQUENCY</div>
                <div className="card-value highlight-gold">
                    {telemetry.dominantFrequency.toFixed(2)}
                    <span className="card-unit"> Hz</span>
                </div>
                <div className="card-subtext">
                    FFT modal resonance peak (f &gt; 0.5 Hz)
                </div>
            </div>

            {/* 5. Active Features Card */}
            <div className="telemetry-card">
                <div className="card-label">TRACKED FEATURES</div>
                <div className="card-value">
                    {telemetry.featureCount}
                    <span className="card-unit"> / 50</span>
                </div>
                <div className="card-subtext">
                    Shi–Tomasi corner persistence
                </div>
            </div>

            {/* 6. Tracking Quality Card */}
            <div className="telemetry-card">
                <div className="card-label">TRACKING QUALITY</div>
                <div className={`card-value ${trackingColor}`}>
                    {telemetry.trackingQuality}
                </div>
                <div className="card-subtext">
                    Feature retention status
                </div>
            </div>

            {/* 7. Measurement Status Card */}
            <div className="telemetry-card">
                <div className="card-label">MEASUREMENT VALIDITY</div>
                <div className={`card-value ${measurementColor}`}>
                    {telemetry.measurementStatus}
                </div>
                <div className="card-subtext">
                    Physical bounds: 1280x720 frame limits
                </div>
            </div>

            {/* 8. Scenario Response State Card */}
            <div className="telemetry-card scenario-card">
                <div className="card-label">RESPONSE SCENARIO</div>
                <div className="scenario-badge-wrapper">
                    <span className={`scenario-badge ${scenarioBadgeClass}`}>
                        {telemetry.scenario}
                    </span>
                </div>
                <div className="card-subtext">
                    Simulated structural response state
                </div>
            </div>
        </div>
    );
};
