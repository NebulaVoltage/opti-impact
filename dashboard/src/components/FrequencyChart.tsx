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

        const getYForFreq = (f: number): number => {
            const clamped = Math.max(minY, Math.min(maxY, f));
            return padTop + plotH - ((clamped - minY) / (maxY - minY)) * plotH;
        };

        // Draw Scenario Reference Modal Bands
        // Normal band (4.6 - 5.4 Hz)
        ctx.fillStyle = "rgba(52, 199, 89, 0.08)";
        ctx.fillRect(padLeft, getYForFreq(5.4), plotW, getYForFreq(4.6) - getYForFreq(5.4));

        // Warning band (3.9 - 4.6 Hz)
        ctx.fillStyle = "rgba(255, 149, 0, 0.08)";
        ctx.fillRect(padLeft, getYForFreq(4.6), plotW, getYForFreq(3.9) - getYForFreq(4.6));

        // Critical band (2.8 - 3.9 Hz)
        ctx.fillStyle = "rgba(255, 59, 48, 0.08)";
        ctx.fillRect(padLeft, getYForFreq(3.9), plotW, getYForFreq(2.8) - getYForFreq(3.9));

        // Draw horizontal gridlines & labels
        ctx.strokeStyle = "rgba(40, 55, 75, 0.45)";
        ctx.lineWidth = 1;
        ctx.fillStyle = "#7085a0";
        ctx.font = "10px 'JetBrains Mono', monospace";
        ctx.textAlign = "right";

        const fSteps = [3.0, 4.0, 5.0, 6.0];
        for (const f of fSteps) {
            const y = getYForFreq(f);
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
            ctx.lineWidth = 2.2;
            ctx.beginPath();

            history.forEach((pt, idx) => {
                const fracX = (pt.time - tStart) / windowDuration;
                const x = padLeft + Math.max(0, Math.min(1, fracX)) * plotW;
                const y = getYForFreq(pt.freq);

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
            const lastY = getYForFreq(lastPt.freq);

            ctx.beginPath();
            ctx.arc(lastX, lastY, 4, 0, 2 * Math.PI);
            ctx.fillStyle = "#ffdd44";
            ctx.fill();
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 1.4;
            ctx.stroke();
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
