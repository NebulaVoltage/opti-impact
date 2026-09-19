import React from "react";
import type { ScenarioType } from "../types/telemetry";

interface ScenarioControlsProps {
    currentScenario: ScenarioType;
    autoScenarioActive: boolean;
    isRunning: boolean;
    onSelectScenario: (scenario: ScenarioType) => void;
    onToggleAutoScenario: () => void;
    onStart: () => void;
    onPause: () => void;
    onReset: () => void;
}

export const ScenarioControls: React.FC<ScenarioControlsProps> = ({
    currentScenario,
    autoScenarioActive,
    isRunning,
    onSelectScenario,
    onToggleAutoScenario,
    onStart,
    onPause,
    onReset,
}) => {
    return (
        <div className="scenario-controls-panel">
            <div className="controls-group">
                <div className="controls-heading">SIMULATED STRUCTURAL RESPONSE SCENARIO</div>
                <div className="button-group">
                    <button
                        className={`btn-scenario ${currentScenario === "NORMAL" ? "active normal" : ""}`}
                        onClick={() => onSelectScenario("NORMAL")}
                    >
                        <span className="btn-indicator indicator-normal"></span>
                        NORMAL
                        <span className="btn-detail">~0.6 mm | 5.15 Hz</span>
                    </button>

                    <button
                        className={`btn-scenario ${currentScenario === "WARNING" ? "active warning" : ""}`}
                        onClick={() => onSelectScenario("WARNING")}
                    >
                        <span className="btn-indicator indicator-warning"></span>
                        WARNING
                        <span className="btn-detail">~2.8 mm | 4.30 Hz</span>
                    </button>

                    <button
                        className={`btn-scenario ${currentScenario === "CRITICAL" ? "active critical" : ""}`}
                        onClick={() => onSelectScenario("CRITICAL")}
                    >
                        <span className="btn-indicator indicator-critical"></span>
                        CRITICAL
                        <span className="btn-detail">~11.2 mm | 3.50 Hz</span>
                    </button>

                    <button
                        className={`btn-scenario btn-auto-scenario ${autoScenarioActive ? "active-auto" : ""}`}
                        onClick={onToggleAutoScenario}
                        title="Automatically sequences through Normal (10s) -> Warning (8s) -> Critical (6s) -> Warning (8s)"
                    >
                        <span className={`auto-pulse-dot ${autoScenarioActive ? "pulsing" : ""}`}></span>
                        AUTO SEQUENCE
                        <span className="btn-detail">{autoScenarioActive ? "ACTIVE (32s LOOP)" : "DISABLED"}</span>
                    </button>
                </div>
            </div>

            <div className="controls-group playback-group">
                <div className="controls-heading">ENGINE SIMULATION STREAM CONTROLS</div>
                <div className="button-group">
                    {isRunning ? (
                        <button className="btn-action btn-pause" onClick={onPause}>
                            ❚❚ PAUSE
                        </button>
                    ) : (
                        <button className="btn-action btn-start" onClick={onStart}>
                            ▶ RESUME
                        </button>
                    )}

                    <button className="btn-action btn-reset" onClick={onReset}>
                        ↺ RESET BASELINE
                    </button>
                </div>
            </div>
        </div>
    );
};
