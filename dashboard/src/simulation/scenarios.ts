/**
 * Structural simulation scenario profiles.
 *
 * NOTE: These ranges are SIMULATION PARAMETERS ONLY for testing the visualization pipeline.
 * They are NOT experimentally validated structural damage thresholds or safety limits.
 */

import type { ScenarioType, TrackingQuality, MeasurementStatus } from "../types/telemetry";

export interface ScenarioProfile {
    name: ScenarioType;
    label: string;
    description: string;
    targetDisplacement: number; // mm
    targetFrequency: number; // Hz
    baseFeatureCount: number;
    nominalTrackingQuality: TrackingQuality;
    nominalMeasurementStatus: MeasurementStatus;
    dampingRatio: number;
    noiseAmplitude: number; // mm
}

export const SCENARIO_PROFILES: Record<ScenarioType, ScenarioProfile> = {
    NORMAL: {
        name: "NORMAL",
        label: "Normal Response",
        description: "Small ambient oscillation, stable baseline natural frequency, high feature retention.",
        targetDisplacement: 0.65, // mm (~0.2-1.0 mm)
        targetFrequency: 5.15, // Hz (~4.6-5.4 Hz)
        baseFeatureCount: 46, // (~25-60)
        nominalTrackingQuality: "GOOD",
        nominalMeasurementStatus: "VALID",
        dampingRatio: 0.05,
        noiseAmplitude: 0.03,
    },
    WARNING: {
        name: "WARNING",
        label: "Abnormal Response / Warning",
        description: "Elevated vibration displacement, noticeable modal frequency downward shift, partial feature drop.",
        targetDisplacement: 2.85, // mm (~1.0-5.0 mm)
        targetFrequency: 4.30, // Hz (~3.9-4.8 Hz)
        baseFeatureCount: 32, // (~15-50)
        nominalTrackingQuality: "GOOD",
        nominalMeasurementStatus: "VALID",
        dampingRatio: 0.04,
        noiseAmplitude: 0.08,
    },
    CRITICAL: {
        name: "CRITICAL",
        label: "Severe Impact / Critical",
        description: "High-amplitude deflection, severe frequency reduction from structural stiffness loss, tracking degradation.",
        targetDisplacement: 11.20, // mm (~5.0-20.0 mm)
        targetFrequency: 3.50, // Hz (~3.0-4.1 Hz)
        baseFeatureCount: 15, // (~5-35)
        nominalTrackingQuality: "DEGRADED",
        nominalMeasurementStatus: "VALID",
        dampingRatio: 0.03,
        noiseAmplitude: 0.25,
    },
};
