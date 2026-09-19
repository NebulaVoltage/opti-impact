import React, { useRef, useEffect } from "react";
import type { VisionTelemetry } from "../types/telemetry";

interface DisplacementChartProps {
    telemetry: VisionTelemetry;
}

interface DataPoint {
    time: number;
    dispX: number;
    dispMag: number;
}

export const DisplacementChart: React.FC<DisplacementChartProps> = ({ telemetry }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const historyRef = useRef<DataPoint[]>([]);

    useEffect(() => {
        const history = historyRef.current;
        history.push({
            time: telemetry.timestamp,
            dispX: telemetry.displacementX,
            dispMag: telemetry.displacementMagnitude,
        });

        // Retain 10 seconds of data (at ~30 FPS -> 300 points)
        const windowDuration = 10.0;
        const cutoff = telemetry.timestamp - windowDuration;
        while (history.length > 0 && history[0].time < cutoff) {
            history.shift();
        }

        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const w = canvas.width;
        const h = canvas.height;

        // Clear canvas
        ctx.fillStyle = "#0c1017";
        ctx.fillRect(0, 0, w, h);

        const padLeft = 55;
        const padRight = 20;
        const padTop = 25;
        const padBottom = 25;
        const plotW = w - padLeft - padRight;
        const plotH = h - padTop - padBottom;

        // Auto-scale Y based on maximum absolute displacement with minimum headroom
        let maxVal = 2.0;
        for (const pt of history) {
            maxVal = Math.max(maxVal, Math.abs(pt.dispX) * 1.25);
        }
        maxVal = Math.ceil(maxVal * 2.0) / 2.0; // Round to nearest 0.5

        // Draw horizontal gridlines
        ctx.strokeStyle = "rgba(40, 55, 75, 0.5)";
        ctx.lineWidth = 1;
        ctx.fillStyle = "#607590";
        ctx.font = "10px 'JetBrains Mono', monospace";
        ctx.textAlign = "right";

        const ySteps = [-maxVal, -maxVal / 2, 0, maxVal / 2, maxVal];
        for (const val of ySteps) {
            const y = padTop + plotH / 2 - (val / maxVal) * (plotH / 2);
            ctx.beginPath();
            ctx.moveTo(padLeft, y);
            ctx.lineTo(w - padRight, y);
            ctx.stroke();

            ctx.fillText(`${val > 0 ? "+" : ""}${val.toFixed(1)}`, padLeft - 8, y + 3);
        }

        // Zero reference axis (bright accent)
        const zeroY = padTop + plotH / 2;
        ctx.strokeStyle = "rgba(100, 140, 190, 0.7)";
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(padLeft, zeroY);
        ctx.lineTo(w - padRight, zeroY);
        ctx.stroke();

        // Draw time axis labels
        ctx.textAlign = "center";
        const tStart = telemetry.timestamp - windowDuration;
        for (let i = 0; i <= 5; i++) {
            const fraction = i / 5;
            const x = padLeft + fraction * plotW;
            const tVal = tStart + fraction * windowDuration;
            ctx.beginPath();
            ctx.moveTo(x, zeroY - 3);
            ctx.lineTo(x, zeroY + 3);
            ctx.stroke();
            ctx.fillText(`${Math.max(0, tVal).toFixed(1)}s`, x, h - 8);
        }

        // Draw Displacement Waveform
        if (history.length >= 2) {
            ctx.strokeStyle = "#00d4ff";
            ctx.lineWidth = 2;
            ctx.beginPath();

            history.forEach((pt, idx) => {
                const fracX = (pt.time - tStart) / windowDuration;
                const x = padLeft + Math.max(0, Math.min(1, fracX)) * plotW;
                const y = zeroY - (pt.dispX / maxVal) * (plotH / 2);

                if (idx === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            ctx.stroke();

            // Gradient fill under curve
            const lastPt = history[history.length - 1];
            const lastFracX = (lastPt.time - tStart) / windowDuration;
            const lastX = padLeft + Math.max(0, Math.min(1, lastFracX)) * plotW;
            const lastY = zeroY - (lastPt.dispX / maxVal) * (plotH / 2);

            // Pulsing live head circle
            ctx.beginPath();
            ctx.arc(lastX, lastY, 4, 0, 2 * Math.PI);
            ctx.fillStyle = "#00ffff";
            ctx.fill();
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        // Live Readout Label in Plot
        ctx.textAlign = "left";
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 11px 'JetBrains Mono', monospace";
        ctx.fillText(`Live: ${telemetry.displacementX > 0 ? "+" : ""}${telemetry.displacementX.toFixed(2)} mm (Peak Mag: ${telemetry.displacementMagnitude.toFixed(2)} mm)`, padLeft + 12, padTop + 14);

    }, [telemetry]);

    return (
        <div className="displacement-chart-panel">
            <div className="panel-header">
                <div className="panel-title">
                    <span className="live-dot dot-cyan"></span>
                    REAL-TIME OPTICAL DISPLACEMENT WAVEFORM [10s SCROLLING]
                </div>
                <div className="panel-badge">
                    CHANNEL: PRIMARY AXIS X (mm)
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
        </div>
    );
};
