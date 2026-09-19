import React from "react";
import type { SimulationEvent } from "../types/telemetry";

interface EventTimelineProps {
    events: SimulationEvent[];
}

export const EventTimeline: React.FC<EventTimelineProps> = ({ events }) => {
    return (
        <div className="event-timeline-panel">
            <div className="panel-header">
                <div className="panel-title">
                    <span className="live-dot dot-blue"></span>
                    SIMULATION EVENT LOG [LIVE STREAM]
                </div>
                <div className="panel-badge">
                    SIMULATION AUDIT
                </div>
            </div>

            <div className="timeline-list">
                {events.length === 0 ? (
                    <div className="timeline-empty">Awaiting simulation events...</div>
                ) : (
                    events.slice(0, 15).map((evt) => (
                        <div key={evt.id} className={`timeline-entry entry-${evt.type}`}>
                            <span className="entry-timestamp">{evt.timestamp}</span>
                            <span className={`entry-badge badge-${evt.type}`}>
                                {evt.type.toUpperCase()}
                            </span>
                            <span className="entry-message">{evt.message}</span>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};
