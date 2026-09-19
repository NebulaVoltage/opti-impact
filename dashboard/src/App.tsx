import React, { useState, useEffect, useRef } from "react";
import { VisionSimulator } from "./simulation/visionSimulator";
import type { VisionTelemetry, ScenarioType, SimulationEvent } from "./types/telemetry";
import { StructuralViewer } from "./components/StructuralViewer";
import { TelemetryCards } from "./components/TelemetryCards";
import { OpticalQualityPanel } from "./components/OpticalQualityPanel";
import { DisplacementChart } from "./components/DisplacementChart";
import { FrequencyChart } from "./components/FrequencyChart";
import { ScenarioControls } from "./components/ScenarioControls";
import { EventTimeline } from "./components/EventTimeline";
import { OpticalSensorIndicator } from "./components/OpticalSensorIndicator";
import { OpticsEducationalPanel } from "./components/OpticsEducationalPanel";

export const App: React.FC = () => {
    const simulatorRef = useRef<VisionSimulator | null>(null);
    const [telemetry, setTelemetry] = useState<VisionTelemetry | null>(null);
    const [events, setEvents] = useState<SimulationEvent[]>([]);
    const [isRunning, setIsRunning] = useState<boolean>(true);
    const [currentScenario, setCurrentScenario] = useState<ScenarioType>("NORMAL");
    const [autoScenarioActive, setAutoScenarioActive] = useState<boolean>(false);

    useEffect(() => {
        const sim = new VisionSimulator();
        simulatorRef.current = sim;

        const unsubTel = sim.subscribeTelemetry((t) => {
            setTelemetry(t);
            setCurrentScenario(t.scenario);
            setAutoScenarioActive(t.autoScenarioActive);
        });

        const unsubEvt = sim.subscribeEvents((e) => {
            setEvents((prev) => [e, ...prev].slice(0, 30));
        });

        sim.start();
        setIsRunning(true);

        return () => {
            sim.pause();
            unsubTel();
            unsubEvt();
        };
    }, []);

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

    const handleStart = () => {
        if (simulatorRef.current) {
            simulatorRef.current.start();
            setIsRunning(true);
        }
    };

    const handlePause = () => {
        if (simulatorRef.current) {
            simulatorRef.current.pause();
            setIsRunning(false);
        }
    };

    const handleReset = () => {
        if (simulatorRef.current) {
            simulatorRef.current.reset();
        }
    };

    if (!telemetry) {
        return <div className="loading-screen">INITIALIZING VISION SIMULATION ENGINE...</div>;
    }

    return (
        <div className="dashboard-container">
            {/* 1. Header Bar */}
            <header className="dashboard-header">
                <div className="header-brand">
                    <div className="system-logo">OPTI-IMPACT</div>
                    <div className="header-titles">
                        <h1 className="header-title">OPTICAL STRUCTURAL IMPACT MONITORING</h1>
                        <span className="header-subtitle">
                            VISION-ONLY SIMULATION MODE &bull; PHYSICAL KINEMATICS ENGINE
                        </span>
                    </div>
                </div>

                <div className="header-status-bar">
                    <div className="status-pill">
                        <span className="status-dot dot-cyan"></span>
                        <span className="status-name">ENGINE:</span>
                        <span className="status-val">{isRunning ? "30 FPS LIVE" : "PAUSED"}</span>
                    </div>

                    <div className="status-pill">
                        <span className="status-dot dot-yellow"></span>
                        <span className="status-name">CADENCE:</span>
                        <span className="status-val">{telemetry.timestamp.toFixed(1)}s</span>
                    </div>

                    <div className={`status-pill pill-scenario scenario-${currentScenario.toLowerCase()}`}>
                        <span className="scenario-label">SIMULATED STATE:</span>
                        <span className="scenario-name">{currentScenario}</span>
                    </div>
                </div>
            </header>

            {/* 2. Main Content Grid */}
            <main className="dashboard-main-grid">
                {/* Left Column: Visual Structural Representation & Scrolling Kinematics Waveform */}
                <section className="main-left-column">
                    <StructuralViewer telemetry={telemetry} />
                    <DisplacementChart telemetry={telemetry} />
                    <OpticsEducationalPanel />
                </section>

                {/* Right Column: Telemetry Cards, Optical Quality Breakdown, Frequency Trend, Event Log */}
                <section className="main-right-column">
                    <TelemetryCards telemetry={telemetry} />
                    <OpticalQualityPanel telemetry={telemetry} />

                    <div className="right-subgrid">
                        <FrequencyChart telemetry={telemetry} />
                        <OpticalSensorIndicator />
                    </div>

                    <EventTimeline events={events} />
                </section>
            </main>

            {/* 3. Bottom Controls Panel */}
            <section className="dashboard-controls-section">
                <ScenarioControls
                    currentScenario={currentScenario}
                    autoScenarioActive={autoScenarioActive}
                    isRunning={isRunning}
                    onSelectScenario={handleSelectScenario}
                    onToggleAutoScenario={handleToggleAutoScenario}
                    onStart={handleStart}
                    onPause={handlePause}
                    onReset={handleReset}
                />
            </section>

            {/* 4. Engineering Honesty & Safety Disclaimer Footer */}
            <footer className="dashboard-footer">
                <div className="footer-content">
                    <span className="footer-tag">VISION SIMULATION MODE:</span>
                    <span className="footer-text">
                        All telemetry, deflections, and frequencies displayed in this web dashboard are synthetic values generated for visualization integration.
                        This interface does NOT perform structural safety certification or confirm real-world bridge damage.
                        Physical webcam optical flow is maintained independently in the frozen <code>optical/</code> Python subsystem.
                    </span>
                </div>
            </footer>
        </div>
    );
};

export default App;
