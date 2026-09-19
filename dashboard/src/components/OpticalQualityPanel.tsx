import React from "react";
import type { VisionTelemetry } from "../types/telemetry";

interface OpticalQualityPanelProps {
    telemetry: VisionTelemetry;
}

export const OpticalQualityPanel: React.FC<OpticalQualityPanelProps> = ({ telemetry }) => {
    const confPercent = (telemetry.confidence * 100).toFixed(1);

    const confClass =
        telemetry.confidence >= 0.85
            ? "conf-high"
            : telemetry.confidence >= 0.65
            ? "conf-medium"
            : "conf-low";

    const qualityClass =
        telemetry.trackingQuality === "GOOD"
            ? "text-green"
            : telemetry.trackingQuality === "DEGRADED"
            ? "text-orange"
            : "text-red";

    return (
        <div className="optical-quality-panel">
            <div className="quality-header">
                <div className="quality-title-group">
                    <span className="live-dot dot-cyan"></span>
                    <span className="quality-title">SIMULATED CONFIDENCE & OPTICAL QUALITY</span>
                </div>
                <div className="quality-badge">
                    PRECISION AUDIT METRICS (STEP 6A.1-P)
                </div>
            </div>

            {/* Continuous Confidence Meter */}
            <div className="confidence-meter-container">
                <div className="meter-label-row">
                    <span className="meter-label">TRACKING CONFIDENCE SCORE</span>
                    <span className={`meter-value ${confClass}`}>{confPercent}%</span>
                </div>
                <div className="meter-bar-track">
                    <div
                        className={`meter-bar-fill ${confClass}`}
                        style={{ width: `${Math.min(100, Math.max(5, telemetry.confidence * 100))}%` }}
                    />
                </div>
                <div className="meter-ticks">
                    <span>0%</span>
                    <span>25%</span>
                    <span>50%</span>
                    <span>75%</span>
                    <span>100%</span>
                </div>
            </div>

            {/* Pipeline Stage Feature Breakdown */}
            <div className="feature-audit-grid">
                <div className="audit-card">
                    <div className="audit-val text-white">{50}</div>
                    <div className="audit-label">Detected Corners</div>
                    <div className="audit-hint">Shi–Tomasi Eigenvalues</div>
                </div>

                <div className="audit-card">
                    <div className="audit-val text-cyan">{telemetry.forwardValidCount}</div>
                    <div className="audit-label">Forward-Valid</div>
                    <div className="audit-hint">Forward LK Flow (st=1)</div>
                </div>

                <div className="audit-card">
                    <div className="audit-val text-green">{telemetry.backwardValidCount}</div>
                    <div className="audit-label">Backward-Valid</div>
                    <div className="audit-hint">Reverse LK Flow (st=2)</div>
                </div>

                <div className="audit-card">
                    <div className="audit-val text-red">{telemetry.rejectedCount}</div>
                    <div className="audit-label">Rejected Outliers</div>
                    <div className="audit-hint">FB Error + MAD Filter</div>
                </div>

                <div className="audit-card highlight-inliers">
                    <div className="audit-val text-vivid-green">{telemetry.inlierCount}</div>
                    <div className="audit-label">Spatial Inliers</div>
                    <div className="audit-hint">Pass Bidirectional & MAD</div>
                </div>

                <div className="audit-card">
                    <div className="audit-val text-yellow">{telemetry.retentionRate.toFixed(1)}%</div>
                    <div className="audit-label">Retention Rate</div>
                    <div className="audit-hint">Inliers / Detected</div>
                </div>
            </div>

            {/* Spatial Dispersion & Quality Meta */}
            <div className="quality-submeta-row">
                <div className="submeta-item">
                    <span className="submeta-label">Tracking Quality:</span>
                    <span className={`submeta-value ${qualityClass}`}>{telemetry.trackingQuality}</span>
                </div>
                <div className="submeta-item">
                    <span className="submeta-label">Dispersion MAD X:</span>
                    <span className="submeta-value text-white">{telemetry.madX.toFixed(3)} mm</span>
                </div>
                <div className="submeta-item">
                    <span className="submeta-label">Dispersion MAD Y:</span>
                    <span className="submeta-value text-white">{telemetry.madY.toFixed(3)} mm</span>
                </div>
                <div className="submeta-item">
                    <span className="submeta-label">Measurement:</span>
                    <span className={`submeta-value ${telemetry.measurementValid ? "text-green" : "text-red"}`}>
                        {telemetry.measurementStatus}
                    </span>
                </div>
            </div>
        </div>
    );
};
