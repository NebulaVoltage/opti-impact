import React, { useState, useEffect, useRef } from "react";
import { VisionSimulator } from "./simulation/visionSimulator";
import { OpticalLiveService } from "./services/opticalLiveService";
import type {
    VisionTelemetry,
    ScenarioType,
    SimulationEvent,
    OperatingMode,
    AnalysisTab,
    SessionStats,
    BaselineRecord,
} from "./types/telemetry";

import { StructuralViewer } from "./components/StructuralViewer";
import { LiveCameraView } from "./components/LiveCameraView";
import { TelemetryCards } from "./components/TelemetryCards";
import { OpticalQualityPanel } from "./components/OpticalQualityPanel";
import { DisplacementChart } from "./components/DisplacementChart";
import { FrequencyChart } from "./components/FrequencyChart";
import { FrequencySpectrumChart } from "./components/FrequencySpectrumChart";
import { SpectrogramChart } from "./components/SpectrogramChart";
import { SessionStatsPanel } from "./components/SessionStatsPanel";
import { BaselineDeviationPanel } from "./components/BaselineDeviationPanel";
import { ScenarioControls } from "./components/ScenarioControls";
import { EventTimeline } from "./components/EventTimeline";
import { OpticalSensorIndicator } from "./components/OpticalSensorIndicator";
import { OpticsEducationalPanel } from "./components/OpticsEducationalPanel";

const createInitialStats = (): SessionStats => ({
    startTime: Date.now(),
    durationSeconds: 0,
    totalFrames: 0,
    validFrames: 0,
    invalidFrames: 0,
    validPercent: 100,
    avgFps: 30,
    peakDisplacement: 0,
    rmsDisplacement: 0,
    peakVelocity: 0,
    peakAcceleration: 0,
    meanConfidence: 1.0,
    minConfidence: 1.0,
    freqMean: null,
    freqMedian: null,
    freqMin: null,
    freqMax: null,
    freqStd: null,
});

export const App: React.FC = () => {
    // Mode state
    const [mode, setMode] = useState<OperatingMode>("LIVE");
    const [analysisTab, setAnalysisTab] = useState<AnalysisTab>("TIME_DOMAIN");

    // Telemetry state
    const [telemetry, setTelemetry] = useState<VisionTelemetry | null>(null);
    const [events, setEvents] = useState<SimulationEvent[]>([]);

    // Simulation controller state
    const simulatorRef = useRef<VisionSimulator | null>(null);
    const [isSimRunning, setIsSimRunning] = useState<boolean>(true);
    const [currentScenario, setCurrentScenario] = useState<ScenarioType>("NORMAL");
    const [autoScenarioActive, setAutoScenarioActive] = useState<boolean>(false);

    // Live controller state
    const liveServiceRef = useRef<OpticalLiveService | null>(null);
    const [isLiveConnected, setIsLiveConnected] = useState<boolean>(false);
    const [baselineRecord, setBaselineRecord] = useState<BaselineRecord | null>(null);

    // Session statistics state
    const [sessionStats, setSessionStats] = useState<SessionStats>(createInitialStats);

    const sessionAccumulator = useRef({
        startTime: Date.now(),
        total: 0,
        valid: 0,
        sumSqDisp: 0,
        maxDisp: 0,
        maxVel: 0,
        maxAccel: 0,
        confidenceSum: 0,
        minConf: 1.0,
        fpsSum: 0,
        freqHistory: [] as number[],
    });

    const handleResetSessionStats = () => {
        sessionAccumulator.current = {
            startTime: Date.now(),
            total: 0,
            valid: 0,
            sumSqDisp: 0,
            maxDisp: 0,
            maxVel: 0,
            maxAccel: 0,
            confidenceSum: 0,
            minConf: 1.0,
            fpsSum: 0,
            freqHistory: [],
        };
        setSessionStats(createInitialStats());
    };

    const updateStatsWithTelemetry = (t: VisionTelemetry) => {
        const acc = sessionAccumulator.current;
        acc.total += 1;
        const durSec = (Date.now() - acc.startTime) / 1000;
        acc.confidenceSum += t.confidence;
        if (t.confidence < acc.minConf) acc.minConf = t.confidence;
        acc.fpsSum += t.fps ?? 30;

        if (t.measurementValid) {
            acc.valid += 1;
            const disp = t.displacementMagnitude;
            acc.sumSqDisp += disp * disp;
            if (disp > acc.maxDisp) acc.maxDisp = disp;

            const vel = t.velocityMagnitude;
            if (vel > acc.maxVel) acc.maxVel = vel;

            const accel = t.accelerationMagnitude;
            if (accel > acc.maxAccel) acc.maxAccel = accel;

            if (t.dominantFrequency !== null && t.dominantFrequency > 0.05) {
                acc.freqHistory.push(t.dominantFrequency);
                if (acc.freqHistory.length > 500) acc.freqHistory.shift();
            }
        }

        const validRate = acc.total > 0 ? (acc.valid / acc.total) * 100 : 100;
        const rms = acc.valid > 0 ? Math.sqrt(acc.sumSqDisp / acc.valid) : 0;
        const avgFps = acc.total > 0 ? acc.fpsSum / acc.total : 30;
        const meanConf = acc.total > 0 ? acc.confidenceSum / acc.total : 1.0;

        let meanF: number | null = null;
        let medianF: number | null = null;
        let stdF: number | null = null;
        let minF: number | null = null;
        let maxF: number | null = null;

        if (acc.freqHistory.length > 0) {
            const sorted = [...acc.freqHistory].sort((a, b) => a - b);
            const sumF = acc.freqHistory.reduce((a, b) => a + b, 0);
            meanF = parseFloat((sumF / acc.freqHistory.length).toFixed(2));
            const mid = Math.floor(sorted.length / 2);
            medianF = sorted.length % 2 === 0 ? parseFloat(((sorted[mid - 1] + sorted[mid]) / 2).toFixed(2)) : sorted[mid];
            const sumVar = acc.freqHistory.reduce((a, b) => a + (b - meanF!) ** 2, 0);
            stdF = parseFloat(Math.sqrt(sumVar / acc.freqHistory.length).toFixed(2));
            minF = sorted[0];
            maxF = sorted[sorted.length - 1];
        }

        setSessionStats({
            startTime: acc.startTime,
            durationSeconds: durSec,
            totalFrames: acc.total,
            validFrames: acc.valid,
            invalidFrames: acc.total - acc.valid,
            validPercent: validRate,
            avgFps: avgFps,
            peakDisplacement: acc.maxDisp,
            rmsDisplacement: rms,
            peakVelocity: acc.maxVel,
            peakAcceleration: acc.maxAccel,
            meanConfidence: meanConf,
            minConfidence: acc.minConf,
            freqMean: meanF,
            freqMedian: medianF,
            freqMin: minF,
            freqMax: maxF,
            freqStd: stdF,
        });
    };

    // Initialize simulation & live service
    useEffect(() => {
        const sim = new VisionSimulator();
        simulatorRef.current = sim;

        const live = new OpticalLiveService("localhost:8000");
        liveServiceRef.current = live;

        const unsubStatus = live.onStatus((connected) => {
            setIsLiveConnected(connected);
        });

        live.connect();

        return () => {
            live.disconnect();
            unsubStatus();
            sim.pause();
        };
    }, []);

    // Switch telemetry source based on mode
    useEffect(() => {
        if (mode === "LIVE") {
            if (simulatorRef.current) {
                simulatorRef.current.pause();
            }

            const live = liveServiceRef.current;
            if (!live) return;

            const unsubLive = live.onTelemetry((t) => {
                setTelemetry(t);
                updateStatsWithTelemetry(t);
            });

            return () => {
                unsubLive();
            };
        } else {
            const sim = simulatorRef.current;
            if (!sim) return;

            sim.start();
            setIsSimRunning(true);

            const unsubTel = sim.subscribeTelemetry((t) => {
                setTelemetry(t);
                setCurrentScenario(t.scenario);
                setAutoScenarioActive(t.autoScenarioActive);
                updateStatsWithTelemetry(t);
            });

            const unsubEvt = sim.subscribeEvents((e) => {
                setEvents((prev) => [e, ...prev].slice(0, 30));
            });

            return () => {
                unsubTel();
                unsubEvt();
            };
        }
    }, [mode]);

    // Live actions
    const handleResetBaseline = async () => {
        if (mode === "LIVE" && liveServiceRef.current) {
            await liveServiceRef.current.resetBaseline();
        } else if (simulatorRef.current) {
            simulatorRef.current.reset();
        }
    };

    const handleCaptureBaseline = async () => {
        if (mode === "LIVE" && liveServiceRef.current) {
            const res = await liveServiceRef.current.captureBaseline();
            if (res) {
                setBaselineRecord(res);
            }
        } else if (telemetry) {
            setBaselineRecord({
                timestamp: telemetry.timestamp,
                baseline_frequency_hz: telemetry.dominantFrequency,
                baseline_displacement_mag_px: telemetry.displacementMagnitude,
                baseline_velocity_mag_px_s: telemetry.velocityMagnitude,
                baseline_confidence: telemetry.confidence,
                captured_at: new Date().toISOString(),
            });
        }
    };

    // Simulation Handlers
    const handleSelectScenario = (sc: ScenarioType) => {
        if (simulatorRef.current) {
            simulatorRef.current.setScenario(sc);
            setCurrentScenario(sc);
            setAutoScenarioActive(simulatorRef.current.getAutoScenario());
        }
    };

    const handleToggleAutoScenario = () => {
        if (simulatorRef.current) {
            const active = simulatorRef.current.toggleAutoScenario();
            setAutoScenarioActive(active);
        }
    };

    const handleStartSim = () => {
        if (simulatorRef.current) {
            simulatorRef.current.start();
            setIsSimRunning(true);
        }
    };

    const handlePauseSim = () => {
        if (simulatorRef.current) {
            simulatorRef.current.pause();
            setIsSimRunning(false);
        }
    };

    const handleResetSim = () => {
        if (simulatorRef.current) {
            simulatorRef.current.reset();
        }
    };

    return (
        <div className="dashboard-container">
            {/* 1. Top Header Bar */}
            <header className="dashboard-header">
                <div className="header-brand">
                    <div className="system-logo">OPTI-IMPACT</div>
                    <div className="header-titles">
                        <h1 className="header-title">OPTICAL STRUCTURAL IMPACT MONITORING</h1>
                        <span className="header-subtitle">
                            {mode === "LIVE"
                                ? "PHYSICAL WEBCAM SENSING • REAL-TIME MJPEG & TELEMETRY STREAM"
                                : "VISION-ONLY SIMULATION MODE • PHYSICAL KINEMATICS ENGINE"}
                        </span>
                    </div>
                </div>

                {/* Primary Dual Operating Mode Switcher */}
                <div className="operating-mode-selector">
                    <button
                        className={`btn-operating-mode btn-live-mode ${mode === "LIVE" ? "active" : ""}`}
                        onClick={() => setMode("LIVE")}
                        title="Switch to physical USB webcam optical pipeline"
                    >
                        <span className="mode-led led-green"></span>
                        ● LIVE CAMERA
                    </button>
                    <button
                        className={`btn-operating-mode btn-sim-mode ${mode === "SIMULATION" ? "active" : ""}`}
                        onClick={() => setMode("SIMULATION")}
                        title="Switch to synthetic kinematic simulation"
                    >
                        <span className="mode-led led-cyan"></span>
                        ○ SIMULATION
                    </button>
                </div>

                <div className="header-status-bar">
                    {mode === "LIVE" ? (
                        <>
                            <div className="status-pill">
                                <span className={`status-dot ${isLiveConnected ? "dot-cyan" : "dot-red"}`}></span>
                                <span className="status-name">WEBSOCKET:</span>
                                <span className="status-val">{isLiveConnected ? "CONNECTED" : "DISCONNECTED"}</span>
                            </div>
                            <div className="status-pill">
                                <span className="status-dot dot-yellow"></span>
                                <span className="status-name">UNITS:</span>
                                <span className="status-val">PIXELS [px]</span>
                            </div>
                        </>
                    ) : (
                        <>
                            <div className="status-pill">
                                <span className="status-dot dot-cyan"></span>
                                <span className="status-name">ENGINE:</span>
                                <span className="status-val">{isSimRunning ? "30 FPS LIVE" : "PAUSED"}</span>
                            </div>
                            <div className={`status-pill pill-scenario scenario-${currentScenario.toLowerCase()}`}>
                                <span className="scenario-label">SIM STATE:</span>
                                <span className="scenario-name">{currentScenario}</span>
                            </div>
                        </>
                    )}
                </div>
            </header>

            {/* 2. Mode-Specific Sub-Nav / Analytics Tabs */}
            <div className="dashboard-subnav-bar">
                <div className="analysis-tabs-group">
                    <button
                        className={`tab-btn ${analysisTab === "TIME_DOMAIN" ? "active" : ""}`}
                        onClick={() => setAnalysisTab("TIME_DOMAIN")}
                    >
                        📈 TIME DOMAIN
                    </button>
                    <button
                        className={`tab-btn ${analysisTab === "FREQUENCY_DOMAIN" ? "active" : ""}`}
                        onClick={() => setAnalysisTab("FREQUENCY_DOMAIN")}
                    >
                        📊 FREQUENCY SPECTRUM (FFT)
                    </button>
                    <button
                        className={`tab-btn ${analysisTab === "SPECTROGRAM" ? "active" : ""}`}
                        onClick={() => setAnalysisTab("SPECTROGRAM")}
                    >
                        🌈 WATERFALL SPECTROGRAM
                    </button>
                    <button
                        className={`tab-btn ${analysisTab === "SESSION_STATS" ? "active" : ""}`}
                        onClick={() => setAnalysisTab("SESSION_STATS")}
                    >
                        📋 SESSION STATS
                    </button>
                    <button
                        className={`tab-btn ${analysisTab === "ALL_VIEWS" ? "active" : ""}`}
                        onClick={() => setAnalysisTab("ALL_VIEWS")}
                    >
                        🗂 COMBINED VIEW
                    </button>
                </div>

                {/* Fast Action Buttons */}
                <div className="subnav-actions">
                    <button
                        className="btn-action-pill"
                        onClick={handleResetBaseline}
                        title="Re-estimate feature center and set current displacement to 0"
                    >
                        🎯 ZERO / RECENTER BASELINE
                    </button>
                    <button
                        className="btn-action-pill pill-highlight"
                        onClick={handleCaptureBaseline}
                        title="Lock current frequency as baseline reference for structural shift calculation"
                    >
                        📌 CAPTURE FREQ BASELINE
                    </button>
                </div>
            </div>

            {/* 3. Main Content Grid */}
            <main className="dashboard-main-grid">
                {/* Left Column: Video/Structural Viewport & Primary Charts */}
                <section className="main-left-column">
                    {mode === "LIVE" ? (
                        <LiveCameraView
                            telemetry={telemetry}
                            isBackendConnected={isLiveConnected}
                            streamUrl="http://localhost:8000/api/video"
                            onSwitchToSimulation={() => setMode("SIMULATION")}
                            onResetBaseline={handleResetBaseline}
                            onCaptureBaseline={handleCaptureBaseline}
                        />
                    ) : (
                        <StructuralViewer telemetry={telemetry || ({} as any)} />
                    )}

                    {/* Active Analysis View */}
                    {(analysisTab === "TIME_DOMAIN" || analysisTab === "ALL_VIEWS") && telemetry && (
                        <DisplacementChart telemetry={telemetry} isLive={mode === "LIVE"} />
                    )}

                    {(analysisTab === "FREQUENCY_DOMAIN" || analysisTab === "ALL_VIEWS") && telemetry && (
                        <FrequencySpectrumChart telemetry={telemetry} />
                    )}

                    {(analysisTab === "SPECTROGRAM" || analysisTab === "ALL_VIEWS") && telemetry && (
                        <SpectrogramChart telemetry={telemetry} />
                    )}

                    {(analysisTab === "SESSION_STATS" || analysisTab === "ALL_VIEWS") && (
                        <SessionStatsPanel stats={sessionStats} unit={mode === "LIVE" ? "px" : "mm"} onResetStats={handleResetSessionStats} />
                    )}

                    {/* Educational / Architectural Panel */}
                    <OpticsEducationalPanel />
                </section>

                {/* Right Column: Telemetry Cards, Baseline Deviation, Quality, Events */}
                <section className="main-right-column">
                    {telemetry && <TelemetryCards telemetry={telemetry} />}

                    {/* Baseline Frequency Shift Panel */}
                    {telemetry && (
                        <BaselineDeviationPanel
                            telemetry={telemetry}
                            baseline={baselineRecord}
                            onCaptureBaseline={handleCaptureBaseline}
                        />
                    )}

                    {telemetry && <OpticalQualityPanel telemetry={telemetry} />}

                    <div className="right-subgrid">
                        {telemetry && <FrequencyChart telemetry={telemetry} />}
                        <OpticalSensorIndicator />
                    </div>

                    {mode === "SIMULATION" ? (
                        <EventTimeline events={events} />
                    ) : (
                        <div className="live-event-banner">
                            <div className="banner-title">PHYSICAL PIPELINE TELEMETRY FEED</div>
                            <div className="banner-desc">
                                Operating optical tracker at ~30 FPS on camera index 1. Invalidation reasons and quality metrics are derived in real-time from OpenCV Lucas–Kanade tracker.
                            </div>
                        </div>
                    )}
                </section>
            </main>

            {/* 4. Simulation Controls (Rendered only in simulation mode) */}
            {mode === "SIMULATION" && (
                <section className="dashboard-controls-section">
                    <ScenarioControls
                        currentScenario={currentScenario}
                        autoScenarioActive={autoScenarioActive}
                        isRunning={isSimRunning}
                        onSelectScenario={handleSelectScenario}
                        onToggleAutoScenario={handleToggleAutoScenario}
                        onStart={handleStartSim}
                        onPause={handlePauseSim}
                        onReset={handleResetSim}
                    />
                </section>
            )}

            {/* 5. Engineering Honesty & Safety Disclaimer Footer */}
            <footer className="dashboard-footer">
                <div className="footer-content">
                    <span className="footer-tag">
                        {mode === "LIVE" ? "PHYSICAL OPTICAL SENSING MODE:" : "VISION SIMULATION MODE:"}
                    </span>
                    <span className="footer-text">
                        {mode === "LIVE"
                            ? "Telemetry streamed from physical webcam via FastAPI adapter using frozen optical/ subsystem. Measurements are reported in camera pixels (px, px/s, px/s²) or calibrated planar coordinates. This system is a laboratory structural response monitoring prototype and does NOT replace licensed structural engineering inspection."
                            : "All telemetry, deflections, and frequencies displayed in simulation mode are synthetic values generated for algorithm visualization. This interface does NOT perform structural safety certification or confirm real-world bridge damage."}
                    </span>
                </div>
            </footer>
        </div>
    );
};

export default App;
