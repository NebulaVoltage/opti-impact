import React from "react";
import type { VisionTelemetry } from "../types/telemetry";

interface TelemetryCardsProps {
    telemetry: VisionTelemetry;
}

export const TelemetryCards: React.FC<TelemetryCardsProps> = ({ telemetry }) => {
    const isLive = telemetry.isLive || false;
    const dispUnit = telemetry.unit || (isLive ? "px" : "mm");
    const velUnit = telemetry.velocityUnit || (isLive ? "px/s" : "mm/s");
    const accelUnit = telemetry.accelerationUnit || (isLive ? "px/s²" : "m/s²");

    const trackingColor =
        telemetry.trackingQuality === "GOOD"
            ? "text-green"
            : telemetry.trackingQuality === "DEGRADED"
            ? "text-orange"
            : "text-red";

    const measurementColor = telemetry.measurementValid ? "text-green" : "text-red";

    const scenarioBadgeClass = isLive
        ? "badge-live"
        : telemetry.scenario === "CRITICAL"
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
                    <span className="card-unit"> {dispUnit}</span>
                </div>
                <div className="card-subtext">
                    X: {telemetry.displacementX > 0 ? "+" : ""}{telemetry.displacementX.toFixed(2)} {dispUnit} &nbsp;|&nbsp;
                    Y: {telemetry.displacementY > 0 ? "+" : ""}{telemetry.displacementY.toFixed(2)} {dispUnit}
                </div>
            </div>

            {/* 2. Optical Velocity Card */}
            <div className="telemetry-card">
                <div className="card-label">OPTICAL VELOCITY</div>
                <div className="card-value">
                    {telemetry.velocityMagnitude.toFixed(1)}
                    <span className="card-unit"> {velUnit}</span>
                </div>
                <div className="card-subtext">
                    Numerical derivative dx/dt
                </div>
            </div>

            {/* 3. Optical Acceleration Card */}
            <div className="telemetry-card">
                <div className="card-label">OPTICAL ACCELERATION</div>
                <div className="card-value">
                    {telemetry.accelerationMagnitude.toFixed(2)}
                    <span className="card-unit"> {accelUnit}</span>
                </div>
                <div className="card-subtext">
                    Numerical derivative d²x/dt²
                </div>
            </div>

            {/* 4. Resonant Frequency Card */}
            <div className="telemetry-card">
                <div className="card-label">DOMINANT FREQUENCY</div>
                <div className="card-value highlight-gold">
                    {telemetry.dominantFrequency !== null ? telemetry.dominantFrequency.toFixed(2) : "N/A"}
                    <span className="card-unit"> Hz</span>
                </div>
                <div className="card-subtext">
                    FFT modal resonance peak (f &gt; 0.5 Hz)
                </div>
            </div>

            {/* 5. Active Inlier Features Card */}
            <div className="telemetry-card">
                <div className="card-label">OPTICAL INLIERS</div>
                <div className="card-value highlight-cyan">
                    {telemetry.inlierCount}
                    <span className="card-unit"> / {telemetry.featureCount || 50}</span>
                </div>
                <div className="card-subtext">
                    {telemetry.retentionRate.toFixed(1)}% retention ({telemetry.rejectedCount} rejected)
                </div>
            </div>

            {/* 6. Tracking Confidence Card */}
            <div className="telemetry-card">
                <div className="card-label">TRACKING CONFIDENCE</div>
                <div className={`card-value ${trackingColor}`}>
                    {(telemetry.confidence * 100).toFixed(1)}%
                </div>
                <div className="card-subtext">
                    Quality: {telemetry.trackingQuality} &bull; MAD: &plusmn;{telemetry.madX.toFixed(2)} {dispUnit}
                </div>
            </div>

            {/* 7. Measurement Status Card */}
            <div className="telemetry-card">
                <div className="card-label">MEASUREMENT VALIDITY</div>
                <div className={`card-value ${measurementColor}`}>
                    {telemetry.measurementStatus}
                </div>
                <div className="card-subtext">
                    Reason: {telemetry.validityReason || telemetry.measurementStatus}
                </div>
            </div>

            {/* 8. Operating / Scenario State Card */}
            <div className="telemetry-card scenario-card">
                <div className="card-label">{isLive ? "OPERATING MODE" : "RESPONSE SCENARIO"}</div>
                <div className="scenario-badge-wrapper">
                    <span className={`scenario-badge ${scenarioBadgeClass}`}>
                        {isLive ? "● LIVE WEBCAM" : telemetry.scenario}
                    </span>
                </div>
                <div className="card-subtext">
                    {isLive ? "Physical external video telemetry" : "Simulated structural response state"}
                </div>
            </div>
        </div>
    );
};
