import React from "react";
import type { BaselineRecord, VisionTelemetry } from "../types/telemetry";

interface BaselineDeviationPanelProps {
    telemetry: VisionTelemetry;
    baseline: BaselineRecord | null;
    onCaptureBaseline?: () => void;
}

export const BaselineDeviationPanel: React.FC<BaselineDeviationPanelProps> = ({
    telemetry,
    baseline,
    onCaptureBaseline,
}) => {
    const curFreq = telemetry.dominantFrequency;
    const baseFreq = baseline?.baseline_frequency_hz ?? telemetry.baselineFrequencyHz ?? null;

    let deviation: number | null = null;
    if (curFreq !== null && baseFreq !== null) {
        deviation = parseFloat((curFreq - baseFreq).toFixed(2));
    }

    const devColor =
        deviation === null
            ? "text-muted"
            : Math.abs(deviation) < 0.25
            ? "text-green"
            : Math.abs(deviation) < 0.75
            ? "text-orange"
            : "text-red";

    return (
        <div className="baseline-deviation-panel">
            <div className="panel-header">
                <div className="panel-title">
                    <span className="live-dot dot-yellow"></span>
                    OPTICAL RESPONSE DEVIATION ANALYSIS
                </div>
                {onCaptureBaseline && (
                    <button className="btn-capture-baseline" onClick={onCaptureBaseline}>
                        🎯 CAPTURE CURRENT BASELINE
                    </button>
                )}
            </div>

            <div className="deviation-cards-grid">
                <div className="dev-card">
                    <div className="dev-label">ESTABLISHED BASELINE f₀</div>
                    <div className="dev-value text-gold">
                        {baseFreq !== null ? `${baseFreq.toFixed(2)} Hz` : "NOT ESTABLISHED"}
                    </div>
                    <div className="dev-subtext">
                        {baseline ? `Captured at ${baseline.captured_at}` : "Click capture to lock current baseline"}
                    </div>
                </div>

                <div className="dev-card">
                    <div className="dev-label">CURRENT DOMINANT f</div>
                    <div className="dev-value text-cyan">
                        {curFreq !== null ? `${curFreq.toFixed(2)} Hz` : "SEARCHING..."}
                    </div>
                    <div className="dev-subtext">
                        Real-time resonant peak (FFT)
                    </div>
                </div>

                <div className="dev-card">
                    <div className="dev-label">FREQUENCY SHIFT (Δf)</div>
                    <div className={`dev-value ${devColor}`}>
                        {deviation !== null
                            ? `${deviation > 0 ? "+" : ""}${deviation.toFixed(2)} Hz`
                            : "N/A"}
                    </div>
                    <div className="dev-subtext">
                        Δf = f_current - f_baseline
                    </div>
                </div>
            </div>

            <div className="deviation-footer-notice">
                ℹ <strong>Optical Response Deviation:</strong> Quantifies the resonant modal frequency delta from initial baseline. A negative shift indicates stiffness softening or support boundary degradation.
            </div>
        </div>
    );
};
