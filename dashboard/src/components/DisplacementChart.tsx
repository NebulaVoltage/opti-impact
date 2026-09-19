import React, { useRef, useEffect, useState } from "react";
import type { VisionTelemetry } from "../types/telemetry";

interface DisplacementChartProps {
    telemetry: VisionTelemetry;
}

export type KinematicMode = "DISPLACEMENT" | "VELOCITY" | "ACCELERATION";

interface KinematicPoint {
    time: number;
    dispX: number;
    dispMag: number;
    velX: number;
    velMag: number;
    accelX: number;
    accelMag: number;
}

export const DisplacementChart: React.FC<DisplacementChartProps> = ({ telemetry }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const historyRef = useRef<KinematicPoint[]>([]);
    const [mode, setMode] = useState<KinematicMode>("DISPLACEMENT");

    useEffect(() => {
        const history = historyRef.current;
        history.push({
            time: telemetry.timestamp,
            dispX: telemetry.displacementX,
            dispMag: telemetry.displacementMagnitude,
            velX: telemetry.velocityX,
            velMag: telemetry.velocityMagnitude,
            accelX: telemetry.accelerationX,
            accelMag: telemetry.accelerationMagnitude,
        });

        // Retain 10 seconds of data (at ~30 FPS -> ~300 points)
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
        ctx.fillStyle = "#080c14";
        ctx.fillRect(0, 0, w, h);

        const padLeft = 60;
        const padRight = 20;
        const padTop = 28;
        const padBottom = 26;
        const plotW = w - padLeft - padRight;
        const plotH = h - padTop - padBottom;

        // Extract value according to active mode
        const getValue = (pt: KinematicPoint): number => {
            if (mode === "VELOCITY") return pt.velX;
            if (mode === "ACCELERATION") return pt.accelX;
            return pt.dispX;
        };

        const getUnit = (): string => {
            if (mode === "VELOCITY") return "mm/s";
            if (mode === "ACCELERATION") return "m/s²";
            return "mm";
        };

        const getThemeColor = (): string => {
            if (mode === "VELOCITY") return "#00ff88";
            if (mode === "ACCELERATION") return "#ffaa00";
            return "#00d4ff";
        };

        // Auto-scale Y based on maximum absolute amplitude with minimum floor
        let minCeiling = mode === "ACCELERATION" ? 0.5 : mode === "VELOCITY" ? 5.0 : 1.5;
        let maxVal = minCeiling;
        for (const pt of history) {
            const val = Math.abs(getValue(pt));
            maxVal = Math.max(maxVal, val * 1.25);
        }
        // Round to clean step
        if (mode === "ACCELERATION") {
            maxVal = Math.ceil(maxVal * 4.0) / 4.0;
        } else if (mode === "VELOCITY") {
            maxVal = Math.ceil(maxVal / 5.0) * 5.0;
        } else {
            maxVal = Math.ceil(maxVal * 2.0) / 2.0;
        }

        // Horizontal gridlines & Y-labels
        ctx.strokeStyle = "rgba(35, 50, 75, 0.45)";
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

            const formatted =
                mode === "ACCELERATION"
                    ? val.toFixed(2)
                    : mode === "VELOCITY"
                    ? val.toFixed(0)
                    : val.toFixed(1);
            ctx.fillText(`${val > 0 ? "+" : ""}${formatted}`, padLeft - 8, y + 3);
        }

        // Zero reference axis
        const zeroY = padTop + plotH / 2;
        ctx.strokeStyle = "rgba(100, 160, 220, 0.65)";
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(padLeft, zeroY);
        ctx.lineTo(w - padRight, zeroY);
        ctx.stroke();

        // Time axis labels
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

        // Draw kinematic waveform
        const themeColor = getThemeColor();
        if (history.length >= 2) {
            ctx.strokeStyle = themeColor;
            ctx.lineWidth = 2;
            ctx.beginPath();

            history.forEach((pt, idx) => {
                const fracX = (pt.time - tStart) / windowDuration;
                const x = padLeft + Math.max(0, Math.min(1, fracX)) * plotW;
                const y = zeroY - (getValue(pt) / maxVal) * (plotH / 2);

                if (idx === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            ctx.stroke();

            // Pulsing live head dot
            const lastPt = history[history.length - 1];
            const lastFracX = (lastPt.time - tStart) / windowDuration;
            const lastX = padLeft + Math.max(0, Math.min(1, lastFracX)) * plotW;
            const lastY = zeroY - (getValue(lastPt) / maxVal) * (plotH / 2);

            ctx.beginPath();
            ctx.arc(lastX, lastY, 4.5, 0, 2 * Math.PI);
            ctx.fillStyle = themeColor;
            ctx.fill();
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 1.6;
            ctx.stroke();
        }

        // Live Readout Label in Plot
        ctx.textAlign = "left";
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 11px 'JetBrains Mono', monospace";

        let liveValStr = "";
        if (mode === "DISPLACEMENT") {
            liveValStr = `Live: ${telemetry.displacementX > 0 ? "+" : ""}${telemetry.displacementX.toFixed(2)} mm (Peak Mag: ${telemetry.displacementMagnitude.toFixed(2)} mm)`;
        } else if (mode === "VELOCITY") {
            liveValStr = `Live: ${telemetry.velocityX > 0 ? "+" : ""}${telemetry.velocityX.toFixed(1)} mm/s (Peak Mag: ${telemetry.velocityMagnitude.toFixed(1)} mm/s)`;
        } else {
            liveValStr = `Live: ${telemetry.accelerationX > 0 ? "+" : ""}${telemetry.accelerationX.toFixed(3)} m/s² (Peak Mag: ${telemetry.accelerationMagnitude.toFixed(3)} m/s²)`;
        }

        ctx.fillText(`${liveValStr} [${getUnit()}]`, padLeft + 12, padTop + 14);

    }, [telemetry, mode]);

    return (
        <div className="displacement-chart-panel">
            <div className="panel-header">
                <div className="panel-title">
                    <span className="live-dot dot-cyan"></span>
                    {mode === "DISPLACEMENT"
                        ? "REAL-TIME OPTICAL DISPLACEMENT [10s SCROLLING]"
                        : mode === "VELOCITY"
                        ? "REAL-TIME OPTICAL VELOCITY (dx/dt) [10s SCROLLING]"
                        : "REAL-TIME OPTICAL ACCELERATION (d²x/dt²) [10s SCROLLING]"}
                </div>
                <div className="mode-toggle-group">
                    <button
                        className={`btn-mode ${mode === "DISPLACEMENT" ? "active" : ""}`}
                        onClick={() => setMode("DISPLACEMENT")}
                    >
                        DISPLACEMENT
                    </button>
                    <button
                        className={`btn-mode ${mode === "VELOCITY" ? "active" : ""}`}
                        onClick={() => setMode("VELOCITY")}
                    >
                        VELOCITY
                    </button>
                    <button
                        className={`btn-mode ${mode === "ACCELERATION" ? "active" : ""}`}
                        onClick={() => setMode("ACCELERATION")}
                    >
                        ACCELERATION
                    </button>
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
