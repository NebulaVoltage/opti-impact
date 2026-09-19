import React from "react";
import type { SessionStats } from "../types/telemetry";

interface SessionStatsPanelProps {
    stats: SessionStats;
    unit?: string;
    onResetStats?: () => void;
}

export const SessionStatsPanel: React.FC<SessionStatsPanelProps> = ({ stats, unit = "px", onResetStats }) => {
    const formatDuration = (sec: number): string => {
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60);
        return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
    };

    return (
        <div className="session-stats-panel">
            <div className="panel-header">
                <div className="panel-title">
                    <span className="live-dot dot-cyan"></span>
                    SESSION KINEMATIC & OPTICAL STATISTICS
                </div>
                <div className="session-header-controls">
                    <span className="session-duration-tag">
                        DURATION: {formatDuration(stats.durationSeconds)} ({stats.totalFrames} FRAMES)
                    </span>
                    {onResetStats && (
                        <button className="btn-icon-sub" onClick={onResetStats} title="Reset session counters">
                            ↺ RESET
                        </button>
                    )}
                </div>
            </div>

            <div className="session-stats-grid">
                {/* 1. Reliability & Validity */}
                <div className="stats-group-card">
                    <div className="group-title">ACQUISITION INTEGRITY</div>
                    <div className="group-metric-row">
                        <span className="metric-name">Valid Measurement:</span>
                        <span className="metric-val text-green">{stats.validPercent.toFixed(1)}%</span>
                    </div>
                    <div className="group-metric-row">
                        <span className="metric-name">Valid / Invalid Frames:</span>
                        <span className="metric-val">{stats.validFrames} / {stats.invalidFrames}</span>
                    </div>
                    <div className="group-metric-row">
                        <span className="metric-name">Measured Average FPS:</span>
                        <span className="metric-val text-cyan">{stats.avgFps.toFixed(1)} FPS</span>
                    </div>
                    <div className="group-metric-row">
                        <span className="metric-name">Confidence (Mean / Min):</span>
                        <span className="metric-val">{(stats.meanConfidence * 100).toFixed(1)}% / {(stats.minConfidence * 100).toFixed(1)}%</span>
                    </div>
                </div>

                {/* 2. Kinematic Extremes */}
                <div className="stats-group-card">
                    <div className="group-title">KINEMATIC RESPONSE ({unit.toUpperCase()})</div>
                    <div className="group-metric-row">
                        <span className="metric-name">Peak Displacement:</span>
                        <span className="metric-val text-cyan">{stats.peakDisplacement.toFixed(2)} {unit}</span>
                    </div>
                    <div className="group-metric-row">
                        <span className="metric-name">RMS Displacement:</span>
                        <span className="metric-val text-white">{stats.rmsDisplacement.toFixed(2)} {unit}</span>
                    </div>
                    <div className="group-metric-row">
                        <span className="metric-name">Peak Velocity:</span>
                        <span className="metric-val text-green">{stats.peakVelocity.toFixed(1)} {unit}/s</span>
                    </div>
                    <div className="group-metric-row">
                        <span className="metric-name">Peak Acceleration:</span>
                        <span className="metric-val text-yellow">{stats.peakAcceleration.toFixed(1)} {unit}/s²</span>
                    </div>
                </div>

                {/* 3. Modal Frequency Statistics */}
                <div className="stats-group-card">
                    <div className="group-title">RESONANT MODAL FREQUENCY</div>
                    <div className="group-metric-row">
                        <span className="metric-name">Mean Frequency:</span>
                        <span className="metric-val text-gold">{stats.freqMean ? `${stats.freqMean.toFixed(2)} Hz` : "N/A"}</span>
                    </div>
                    <div className="group-metric-row">
                        <span className="metric-name">Median Frequency:</span>
                        <span className="metric-val">{stats.freqMedian ? `${stats.freqMedian.toFixed(2)} Hz` : "N/A"}</span>
                    </div>
                    <div className="group-metric-row">
                        <span className="metric-name">Frequency Range (Min–Max):</span>
                        <span className="metric-val">
                            {stats.freqMin && stats.freqMax ? `${stats.freqMin.toFixed(2)} – ${stats.freqMax.toFixed(2)} Hz` : "N/A"}
                        </span>
                    </div>
                    <div className="group-metric-row">
                        <span className="metric-name">Std Deviation (σ):</span>
                        <span className="metric-val">{stats.freqStd ? `±${stats.freqStd.toFixed(3)} Hz` : "N/A"}</span>
                    </div>
                </div>
            </div>

            <div className="session-footer-note">
                ℹ Statistics exclude frames marked invalid (e.g. EXCESSIVE_STEP_DISPLACEMENT or OUT_OF_FRAME) to preserve analytical integrity.
            </div>
        </div>
    );
};
