import React, { useRef, useEffect } from "react";
import type { VisionTelemetry } from "../types/telemetry";

interface SpectrogramChartProps {
    telemetry: VisionTelemetry;
}

interface SpectrogramSlice {
    time: number;
    amplitudes: number[];
}

export const SpectrogramChart: React.FC<SpectrogramChartProps> = ({ telemetry }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const historyRef = useRef<SpectrogramSlice[]>([]);

    const amps = telemetry.fftAmplitudes || [];
    const freqs = telemetry.fftFrequencies || [];

    useEffect(() => {
        if (amps.length > 0) {
            historyRef.current.push({
                time: telemetry.timestamp,
                amplitudes: amps,
            });
            if (historyRef.current.length > 80) {
                historyRef.current.shift();
            }
        }

        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const w = canvas.width;
        const h = canvas.height;

        ctx.fillStyle = "#080c14";
        ctx.fillRect(0, 0, w, h);

        const padLeft = 45;
        const padRight = 20;
        const padTop = 20;
        const padBottom = 22;
        const plotW = w - padLeft - padRight;
        const plotH = h - padTop - padBottom;

        const slices = historyRef.current;
        if (slices.length === 0 || freqs.length === 0) {
            ctx.fillStyle = "#607590";
            ctx.font = "11px 'JetBrains Mono', monospace";
            ctx.textAlign = "center";
            ctx.fillText("ACCUMULATING TIME-FREQUENCY SLICES...", w / 2, h / 2);
            return;
        }

        const numSlices = slices.length;
        const sliceWidth = plotW / Math.max(1, numSlices);
        const numBins = freqs.length;
        const binHeight = plotH / Math.max(1, numBins);

        // Color mapping function from spectral intensity [0.0 - 1.0]
        const getColor = (intensity: number): string => {
            const val = Math.min(1.0, Math.max(0.0, intensity));
            if (val < 0.15) return "#0c1524";
            if (val < 0.35) return "#084081";
            if (val < 0.55) return "#088090";
            if (val < 0.75) return "#00d4ff";
            if (val < 0.90) return "#ffcc00";
            return "#ff3333";
        };

        // Render waterfall heatmap grid
        slices.forEach((sl, sIdx) => {
            const x = padLeft + sIdx * sliceWidth;
            sl.amplitudes.forEach((amp, bIdx) => {
                // Low freq at bottom, high freq at top
                const y = padTop + plotH - (bIdx + 1) * binHeight;
                ctx.fillStyle = getColor(amp * 2.5);
                ctx.fillRect(x, y, sliceWidth + 0.5, binHeight + 0.5);
            });
        });

        // Axes and labels
        ctx.strokeStyle = "rgba(45, 60, 80, 0.5)";
        ctx.strokeRect(padLeft, padTop, plotW, plotH);

        // Frequency tick labels Y
        ctx.fillStyle = "#7085a0";
        ctx.font = "9px 'JetBrains Mono', monospace";
        ctx.textAlign = "right";

        const fSteps = [2, 5, 10, 15];
        const minF = freqs[0];
        const maxF = freqs[freqs.length - 1];

        for (const f of fSteps) {
            if (f < minF || f > maxF) continue;
            const y = padTop + plotH - ((f - minF) / (maxF - minF)) * plotH;
            ctx.fillText(`${f} Hz`, padLeft - 6, y + 3);
        }

        // Time axis label X
        ctx.textAlign = "center";
        ctx.fillText("Time Evolution (Latest →)", padLeft + plotW / 2, h - 6);

    }, [telemetry, amps, freqs]);

    return (
        <div className="spectrogram-panel">
            <div className="panel-header">
                <div className="panel-title">
                    <span className="live-dot dot-cyan"></span>
                    OPTICAL RESPONSE — TIME-FREQUENCY SPECTROGRAM
                </div>
                <div className="spectrogram-legend">
                    <span>Intensity:</span>
                    <span className="legend-bar"></span>
                    <span>High (Resonance)</span>
                </div>
            </div>
            <div className="canvas-wrapper">
                <canvas
                    ref={canvasRef}
                    width={720}
                    height={160}
                    className="chart-canvas"
                />
            </div>
        </div>
    );
};
