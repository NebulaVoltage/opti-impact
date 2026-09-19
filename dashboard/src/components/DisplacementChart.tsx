import React, { useRef, useEffect, useState } from "react";
import type { VisionTelemetry } from "../types/telemetry";

export type KinematicMode = "DISPLACEMENT" | "VELOCITY" | "ACCELERATION";
export type TimeWindowSec = 10 | 30 | 60;

interface DisplacementChartProps {
    telemetry: VisionTelemetry;
    isLive?: boolean;
}

interface KinematicPoint {
    time: number;
    dispX: number;
    dispMag: number;
    velX: number;
    velMag: number;
    accelX: number;
    accelMag: number;
}

export const DisplacementChart: React.FC<DisplacementChartProps> = ({ telemetry, isLive = false }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const historyRef = useRef<KinematicPoint[]>([]);
    const [mode, setMode] = useState<KinematicMode>("DISPLACEMENT");
    const [timeWindow, setTimeWindow] = useState<TimeWindowSec>(10);
    const [isFrozen, setIsFrozen] = useState<boolean>(false);
    const [stats, setStats] = useState({ rms: 0, peak: 0, pkToPk: 0 });

    const isPx = isLive || telemetry.displayUnit === "px";

    const getUnit = (m: KinematicMode): string => {
        if (m === "VELOCITY") return isPx ? "px/s" : "mm/s";
        if (m === "ACCELERATION") return isPx ? "px/s²" : "m/s²";
        return isPx ? "px" : "mm";
    };

    useEffect(() => {
        if (!isFrozen) {
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

            // Retain up to maximum window (60s at 30 FPS = ~1800 points)
            const cutoff = telemetry.timestamp - timeWindow;
            while (history.length > 0 && history[0].time < cutoff - 5) {
                history.shift();
            }
        }

        const history = historyRef.current;
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

        const getThemeColor = (): string => {
            if (mode === "VELOCITY") return "#00ff88";
            if (mode === "ACCELERATION") return "#ffaa00";
            return "#00d4ff";
        };

        // Filter points within the current active timeWindow
        const latestTime = history.length > 0 ? history[history.length - 1].time : telemetry.timestamp;
        const tStart = latestTime - timeWindow;
        const visiblePoints = history.filter((p) => p.time >= tStart);

        // Compute RMS, Peak, Pk-Pk on visible points
        let sumSq = 0;
        let peakVal = 0;
        let minVal = 0;
        let maxValFound = 0;

        if (visiblePoints.length > 0) {
            minVal = getValue(visiblePoints[0]);
            maxValFound = minVal;
            for (const pt of visiblePoints) {
                const val = getValue(pt);
                sumSq += val * val;
                const absVal = Math.abs(val);
                if (absVal > peakVal) peakVal = absVal;
                if (val < minVal) minVal = val;
                if (val > maxValFound) maxValFound = val;
            }
            const rmsVal = Math.sqrt(sumSq / visiblePoints.length);
            setStats({
                rms: rmsVal,
                peak: peakVal,
                pkToPk: maxValFound - minVal,
            });
        }

        // Auto-scale Y based on maximum absolute amplitude with minimum floor
        let minCeiling = isPx
            ? (mode === "ACCELERATION" ? 50.0 : mode === "VELOCITY" ? 20.0 : 10.0)
            : (mode === "ACCELERATION" ? 0.5 : mode === "VELOCITY" ? 5.0 : 1.5);
        let maxVal = minCeiling;
        for (const pt of visiblePoints) {
            const val = Math.abs(getValue(pt));
            maxVal = Math.max(maxVal, val * 1.25);
        }

        // Round to clean step
        if (!isPx && mode === "ACCELERATION") {
            maxVal = Math.ceil(maxVal * 4.0) / 4.0;
        } else if (!isPx && mode === "VELOCITY") {
            maxVal = Math.ceil(maxVal / 5.0) * 5.0;
        } else if (!isPx) {
            maxVal = Math.ceil(maxVal * 2.0) / 2.0;
        } else {
            maxVal = Math.ceil(maxVal / 5.0) * 5.0;
        }

        // Horizontal gridlines & Y-labels
        ctx.strokeStyle = "rgba(35, 50, 75, 0.45)";
        ctx.lineWidth = 1;
        ctx.fillStyle = "#607590";
        ctx.font = "10px monospace";
        ctx.textAlign = "right";

        const ySteps = [-maxVal, -maxVal / 2, 0, maxVal / 2, maxVal];
        for (const val of ySteps) {
            const y = padTop + plotH / 2 - (val / maxVal) * (plotH / 2);
            ctx.beginPath();
            ctx.moveTo(padLeft, y);
            ctx.lineTo(w - padRight, y);
            ctx.stroke();

            const formatted =
                !isPx && mode === "ACCELERATION"
                    ? val.toFixed(2)
                    : isPx
                    ? val.toFixed(1)
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
        for (let i = 0; i <= 5; i++) {
            const fraction = i / 5;
            const x = padLeft + fraction * plotW;
            const tVal = tStart + fraction * timeWindow;
            ctx.beginPath();
            ctx.moveTo(x, zeroY - 3);
            ctx.lineTo(x, zeroY + 3);
            ctx.stroke();
            ctx.fillText(`${Math.max(0, tVal).toFixed(1)}s`, x, h - 8);
        }

        // Draw kinematic waveform
        const themeColor = getThemeColor();
        if (visiblePoints.length >= 2) {
            ctx.strokeStyle = themeColor;
            ctx.lineWidth = 2;
            ctx.beginPath();

            visiblePoints.forEach((pt, idx) => {
                const fracX = (pt.time - tStart) / timeWindow;
                const x = padLeft + Math.max(0, Math.min(1, fracX)) * plotW;
                const y = zeroY - (getValue(pt) / maxVal) * (plotH / 2);

                if (idx === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            ctx.stroke();

            // Pulsing live head dot if not frozen
            if (!isFrozen) {
                const lastPt = visiblePoints[visiblePoints.length - 1];
                const lastFracX = (lastPt.time - tStart) / timeWindow;
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
        }

        // Live Readout Label in Plot
        ctx.textAlign = "left";
        ctx.fillStyle = isFrozen ? "#ffaa00" : "#ffffff";
        ctx.font = "bold 11px monospace";

        let liveValStr = "";
        const unit = getUnit(mode);
        if (mode === "DISPLACEMENT") {
            liveValStr = `Live X: ${telemetry.displacementX > 0 ? "+" : ""}${telemetry.displacementX.toFixed(2)} ${unit} (Mag: ${telemetry.displacementMagnitude.toFixed(2)} ${unit})`;
        } else if (mode === "VELOCITY") {
            liveValStr = `Live Vx: ${telemetry.velocityX > 0 ? "+" : ""}${telemetry.velocityX.toFixed(1)} ${unit} (Mag: ${telemetry.velocityMagnitude.toFixed(1)} ${unit})`;
        } else {
            liveValStr = `Live Ax: ${telemetry.accelerationX > 0 ? "+" : ""}${telemetry.accelerationX.toFixed(2)} ${unit} (Mag: ${telemetry.accelerationMagnitude.toFixed(2)} ${unit})`;
        }

        if (isFrozen) {
            liveValStr = `[FROZEN / PAUSED] ` + liveValStr;
        }

        ctx.fillText(liveValStr, padLeft + 12, padTop + 14);

    }, [telemetry, mode, timeWindow, isFrozen, isPx]);

    return (
        <div className="displacement-chart-panel">
            <div className="panel-header">
                <div className="panel-title">
                    <span className={`live-dot ${isFrozen ? "dot-amber" : "dot-cyan"}`}></span>
                    {mode === "DISPLACEMENT"
                        ? `OPTICAL DISPLACEMENT [${timeWindow}s WINDOW]`
                        : mode === "VELOCITY"
                        ? `OPTICAL VELOCITY (dx/dt) [${timeWindow}s WINDOW]`
                        : `OPTICAL ACCELERATION (d²x/dt²) [${timeWindow}s WINDOW]`}
                </div>
                <div className="chart-controls-group">
                    {/* Time Window Tabs */}
                    <div className="time-window-toggle">
                        <button
                            className={`btn-toggle-sub ${timeWindow === 10 ? "active" : ""}`}
                            onClick={() => setTimeWindow(10)}
                        >
                            10s
                        </button>
                        <button
                            className={`btn-toggle-sub ${timeWindow === 30 ? "active" : ""}`}
                            onClick={() => setTimeWindow(30)}
                        >
                            30s
                        </button>
                        <button
                            className={`btn-toggle-sub ${timeWindow === 60 ? "active" : ""}`}
                            onClick={() => setTimeWindow(60)}
                        >
                            60s
                        </button>
                    </div>

                    {/* Kinematic Mode Toggles */}
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

                    {/* Freeze Button */}
                    <button
                        className={`btn-freeze ${isFrozen ? "frozen" : ""}`}
                        onClick={() => setIsFrozen((prev) => !prev)}
                        title="Pause chart updating to inspect historical waveform"
                    >
                        {isFrozen ? "▶ RESUME" : "⏸ FREEZE"}
                    </button>
                </div>
            </div>

            {/* Quick waveform statistics bar */}
            <div className="chart-stats-strip">
                <span className="stat-item">
                    <span className="stat-k">RMS:</span>{" "}
                    <span className="stat-v">{stats.rms.toFixed(2)} {getUnit(mode)}</span>
                </span>
                <span className="stat-item">
                    <span className="stat-k">PEAK:</span>{" "}
                    <span className="stat-v">{stats.peak.toFixed(2)} {getUnit(mode)}</span>
                </span>
                <span className="stat-item">
                    <span className="stat-k">PK-TO-PK:</span>{" "}
                    <span className="stat-v">{stats.pkToPk.toFixed(2)} {getUnit(mode)}</span>
                </span>
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
