import React, { useRef, useEffect } from "react";
import type { VisionTelemetry } from "../types/telemetry";

interface FrequencySpectrumChartProps {
    telemetry: VisionTelemetry;
}

export const FrequencySpectrumChart: React.FC<FrequencySpectrumChartProps> = ({ telemetry }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    const freqs = telemetry.fftFrequencies || [];
    const amps = telemetry.fftAmplitudes || [];
    const peakFreq = telemetry.peakFrequencyHz ?? telemetry.dominantFrequency;
    const peakAmp = telemetry.peakAmplitude ?? (amps.length > 0 ? Math.max(...amps) : 0.0);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const w = canvas.width;
        const h = canvas.height;

        // Clear canvas
        ctx.fillStyle = "#080c14";
        ctx.fillRect(0, 0, w, h);

        const padLeft = 45;
        const padRight = 20;
        const padTop = 25;
        const padBottom = 25;
        const plotW = w - padLeft - padRight;
        const plotH = h - padTop - padBottom;

        // If no spectral bins yet, render waiting notice
        if (freqs.length === 0 || amps.length === 0) {
            ctx.fillStyle = "#607590";
            ctx.font = "11px 'JetBrains Mono', monospace";
            ctx.textAlign = "center";
            ctx.fillText("WAITING FOR RESIDUAL STRUCTURAL SAMPLES (N >= 32)...", w / 2, h / 2);
            return;
        }

        const minF = freqs[0];
        const maxF = freqs[freqs.length - 1];

        // Max amplitude scale
        let maxA = 0.5;
        for (const a of amps) {
            maxA = Math.max(maxA, a * 1.25);
        }
        maxA = Math.ceil(maxA * 10.0) / 10.0;

        // Grid lines Y
        ctx.strokeStyle = "rgba(40, 55, 75, 0.45)";
        ctx.lineWidth = 1;
        ctx.fillStyle = "#7085a0";
        ctx.font = "9px 'JetBrains Mono', monospace";
        ctx.textAlign = "right";

        for (let i = 0; i <= 4; i++) {
            const val = (i / 4) * maxA;
            const y = padTop + plotH - (val / maxA) * plotH;
            ctx.beginPath();
            ctx.moveTo(padLeft, y);
            ctx.lineTo(w - padRight, y);
            ctx.stroke();
            ctx.fillText(val.toFixed(2), padLeft - 6, y + 3);
        }

        // Grid lines X (Frequency in Hz)
        ctx.textAlign = "center";
        const fSteps = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0];
        for (const f of fSteps) {
            if (f < minF || f > maxF) continue;
            const x = padLeft + ((f - minF) / (maxF - minF)) * plotW;
            ctx.beginPath();
            ctx.moveTo(x, padTop);
            ctx.lineTo(x, padTop + plotH);
            ctx.stroke();
            ctx.fillText(`${f.toFixed(0)} Hz`, x, h - 8);
        }

        // Draw Spectrum Area Fill & Line
        ctx.beginPath();
        freqs.forEach((f, idx) => {
            const x = padLeft + ((f - minF) / (maxF - minF)) * plotW;
            const y = padTop + plotH - (amps[idx] / maxA) * plotH;
            if (idx === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        // Stroke spectrum line
        ctx.strokeStyle = "#00d4ff";
        ctx.lineWidth = 1.8;
        ctx.stroke();

        // Shaded area under FFT curve
        ctx.lineTo(padLeft + plotW, padTop + plotH);
        ctx.lineTo(padLeft, padTop + plotH);
        ctx.closePath();
        const grad = ctx.createLinearGradient(0, padTop, 0, padTop + plotH);
        grad.addColorStop(0, "rgba(0, 212, 255, 0.25)");
        grad.addColorStop(1, "rgba(0, 212, 255, 0.0)");
        ctx.fillStyle = grad;
        ctx.fill();

        // Highlight dominant resonant peak
        if (peakFreq !== null && peakFreq >= minF && peakFreq <= maxF && peakAmp > 0) {
            const px = padLeft + ((peakFreq - minF) / (maxF - minF)) * plotW;
            const py = padTop + plotH - (peakAmp / maxA) * plotH;

            // Vertical marker line
            ctx.setLineDash([3, 3]);
            ctx.strokeStyle = "rgba(255, 204, 0, 0.7)";
            ctx.beginPath();
            ctx.moveTo(px, padTop);
            ctx.lineTo(px, padTop + plotH);
            ctx.stroke();
            ctx.setLineDash([]);

            // Peak circle
            ctx.beginPath();
            ctx.arc(px, py, 4.5, 0, 2 * Math.PI);
            ctx.fillStyle = "#ffcc00";
            ctx.fill();
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Peak callout badge
            ctx.fillStyle = "rgba(10, 15, 24, 0.9)";
            ctx.fillRect(px - 45, py - 24, 90, 18);
            ctx.strokeStyle = "#ffcc00";
            ctx.strokeRect(px - 45, py - 24, 90, 18);

            ctx.fillStyle = "#ffdd44";
            ctx.font = "bold 9px 'JetBrains Mono', monospace";
            ctx.textAlign = "center";
            ctx.fillText(`${peakFreq.toFixed(2)} Hz (f₀)`, px, py - 12);
        }

    }, [telemetry, freqs, amps, peakFreq, peakAmp]);

    return (
        <div className="spectrum-chart-panel">
            <div className="panel-header">
                <div className="panel-title">
                    <span className="live-dot dot-yellow"></span>
                    FFT / FREQUENCY SPECTRUM (OPTICAL RESONANCE)
                </div>
                <div className="spectrum-meta-group">
                    <span className="spec-meta-item">
                        f₀: <strong>{peakFreq ? `${peakFreq.toFixed(2)} Hz` : "N/A"}</strong>
                    </span>
                    <span className="spec-meta-item">
                        Amp: <strong>{peakAmp ? peakAmp.toFixed(3) : "N/A"}</strong>
                    </span>
                </div>
            </div>
            <div className="canvas-wrapper">
                <canvas
                    ref={canvasRef}
                    width={720}
                    height={200}
                    className="chart-canvas"
                />
            </div>
            <div className="spectrum-footer">
                <span>Domain: 0.5 – 15.0 Hz &bull; Hanning-windowed one-sided FFT of detrended optical displacement.</span>
            </div>
        </div>
    );
};
