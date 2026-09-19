import React, { useRef, useEffect } from "react";
import type { VisionTelemetry } from "../types/telemetry";

interface FrequencyChartProps {
    telemetry: VisionTelemetry;
}

interface FreqPoint {
    time: number;
    freq: number;
}

export const FrequencyChart: React.FC<FrequencyChartProps> = ({ telemetry }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const historyRef = useRef<FreqPoint[]>([]);

    useEffect(() => {
        const history = historyRef.current;
        history.push({
            time: telemetry.timestamp,
            freq: telemetry.dominantFrequency,
        });

        // Retain 15 seconds of frequency trend
        const windowDuration = 15.0;
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

        const padLeft = 45;
        const padRight = 15;
        const padTop = 20;
        const padBottom = 25;
        const plotW = w - padLeft - padRight;
        const plotH = h - padTop - padBottom;

        // Y-axis fixed range from 2.5 Hz to 6.0 Hz
        const minY = 2.5;
        const maxY = 6.0;

        // Draw horizontal gridlines
        ctx.strokeStyle = "rgba(40, 55, 75, 0.4)";
        ctx.lineWidth = 1;
        ctx.fillStyle = "#7085a0";
        ctx.font = "10px 'JetBrains Mono', monospace";
        ctx.textAlign = "right";

        const fSteps = [3.0, 4.0, 5.0, 6.0];
        for (const f of fSteps) {
            const y = padTop + plotH - ((f - minY) / (maxY - minY)) * plotH;
            ctx.beginPath();
            ctx.moveTo(padLeft, y);
            ctx.lineTo(w - padRight, y);
            ctx.stroke();

            ctx.fillText(`${f.toFixed(1)}`, padLeft - 6, y + 3);
        }

        // Draw frequency trend curve
        const tStart = telemetry.timestamp - windowDuration;
        if (history.length >= 2) {
            ctx.strokeStyle = "#ffcc00";
            ctx.lineWidth = 2;
            ctx.beginPath();

            history.forEach((pt, idx) => {
                const fracX = (pt.time - tStart) / windowDuration;
                const x = padLeft + Math.max(0, Math.min(1, fracX)) * plotW;
                const fracY = (pt.freq - minY) / (maxY - minY);
                const y = padTop + plotH - Math.max(0, Math.min(1, fracY)) * plotH;

                if (idx === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            ctx.stroke();

            const lastPt = history[history.length - 1];
            const lastFracX = (lastPt.time - tStart) / windowDuration;
            const lastX = padLeft + Math.max(0, Math.min(1, lastFracX)) * plotW;
            const lastFracY = (lastPt.freq - minY) / (maxY - minY);
            const lastY = padTop + plotH - Math.max(0, Math.min(1, lastFracY)) * plotH;

            ctx.beginPath();
            ctx.arc(lastX, lastY, 3.5, 0, 2 * Math.PI);
            ctx.fillStyle = "#ffdd44";
            ctx.fill();
        }
    }, [telemetry]);

    return (
        <div className="frequency-chart-panel">
            <div className="panel-header">
                <div className="panel-title">
                    <span className="live-dot dot-yellow"></span>
                    MODAL FREQUENCY EVOLUTION [15s TREND]
                </div>
                <div className="freq-live-badge">
                    {telemetry.dominantFrequency.toFixed(2)} Hz
                </div>
            </div>
            <div className="canvas-wrapper">
                <canvas
                    ref={canvasRef}
                    width={350}
                    height={160}
                    className="chart-canvas"
                />
            </div>
        </div>
    );
};
