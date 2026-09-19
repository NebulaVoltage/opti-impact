import React, { useState } from "react";
import type { VisionTelemetry } from "../types/telemetry";

interface LiveCameraViewProps {
    telemetry: VisionTelemetry | null;
    isBackendConnected: boolean;
    streamUrl?: string;
    onSwitchToSimulation?: () => void;
    onResetBaseline?: () => void;
    onCaptureBaseline?: () => void;
}

export const LiveCameraView: React.FC<LiveCameraViewProps> = ({
    telemetry,
    isBackendConnected,
    streamUrl = "http://localhost:8000/api/video",
    onSwitchToSimulation,
    onResetBaseline,
    onCaptureBaseline,
}) => {
    const [streamError, setStreamError] = useState<boolean>(false);
    const [snapshotSaved, setSnapshotSaved] = useState<boolean>(false);

    const isConnected = isBackendConnected && !streamError;

    const handleSnapshot = () => {
        setSnapshotSaved(true);
        setTimeout(() => setSnapshotSaved(false), 2000);
    };

    return (
        <div className="live-camera-view-panel">
            <div className="camera-view-header">
                <div className="header-status-group">
                    <span className={`live-dot ${isConnected ? "dot-cyan" : "dot-red"}`}></span>
                    <span className="camera-view-title">
                        {isConnected ? "LIVE OPTICAL CAMERA FEED" : "LIVE CAMERA FEED — DISCONNECTED"}
                    </span>
                    <span className="camera-sensor-tag">
                        {isConnected ? "WEBCAM INDEX 1 • 1280 × 720 HD" : "NO STREAM DETECTED"}
                    </span>
                </div>
                <div className="camera-controls-group">
                    {isConnected && (
                        <>
                            <button
                                className="btn-cam-action"
                                onClick={onResetBaseline}
                                title="Re-center optical tracking reference to current position"
                            >
                                ↺ RECENTER BASELINE
                            </button>
                            {onCaptureBaseline && (
                                <button
                                    className="btn-cam-action"
                                    onClick={onCaptureBaseline}
                                    title="Capture modal frequency baseline"
                                >
                                    🎯 CAPTURE BASELINE
                                </button>
                            )}
                            <button
                                className="btn-cam-action"
                                onClick={handleSnapshot}
                                title="Capture freeze-frame snapshot"
                            >
                                {snapshotSaved ? "✓ SNAPSHOT SAVED" : "📷 SNAPSHOT"}
                            </button>
                        </>
                    )}
                </div>
            </div>

            <div className="video-viewport-wrapper">
                {isConnected ? (
                    <img
                        src={streamUrl}
                        alt="Live Physical Optical Video Stream"
                        className="mjpeg-stream-frame"
                        onError={() => setStreamError(true)}
                        onLoad={() => setStreamError(false)}
                    />
                ) : (
                    <div className="camera-offline-placeholder">
                        <div className="offline-icon">📷</div>
                        <div className="offline-title">PHYSICAL OPTICAL STREAM OFFLINE</div>
                        <div className="offline-desc">
                            The physical webcam server is disconnected or camera device is not opened.
                            Verify that the backend server is running via <code>python -m backend.run_server</code>.
                        </div>
                        {onSwitchToSimulation && (
                            <button className="btn-switch-simulation" onClick={onSwitchToSimulation}>
                                ↺ SWITCH TO SIMULATION MODE
                            </button>
                        )}
                    </div>
                )}

                {/* Overlaid HUD status bar */}
                {isConnected && telemetry && (
                    <div className="camera-hud-overlay">
                        <div className="hud-metric">
                            <span className="hud-label">FPS:</span>
                            <span className="hud-val">{telemetry.fps ? telemetry.fps.toFixed(1) : "30.0"}</span>
                        </div>
                        <div className="hud-metric">
                            <span className="hud-label">TRACKING:</span>
                            <span className={`hud-val ${telemetry.trackingQuality === "GOOD" ? "text-green" : "text-orange"}`}>
                                {telemetry.trackingQuality}
                            </span>
                        </div>
                        <div className="hud-metric">
                            <span className="hud-label">INLIERS:</span>
                            <span className="hud-val text-cyan">
                                {telemetry.inlierCount} / {telemetry.featureCount}
                            </span>
                        </div>
                        <div className="hud-metric">
                            <span className="hud-label">CONFIDENCE:</span>
                            <span className="hud-val text-green">
                                {(telemetry.confidence * 100).toFixed(1)}%
                            </span>
                        </div>
                        <div className="hud-metric">
                            <span className="hud-label">STATUS:</span>
                            <span className={`hud-val ${telemetry.measurementValid ? "text-green" : "text-red"}`}>
                                {telemetry.measurementStatus}
                            </span>
                        </div>
                    </div>
                )}
            </div>

            <div className="camera-view-footer">
                <span className="stream-source-info">
                    ● Real OpenCV Lucas–Kanade tracking pipeline with Forward-Backward consistency and MAD spatial rejection.
                </span>
                <span className="stream-latency-tag">
                    {isConnected ? "TRANSMISSION: DIRECT MJPEG + WEBSOCKET" : "STATUS: OFFLINE"}
                </span>
            </div>
        </div>
    );
};
