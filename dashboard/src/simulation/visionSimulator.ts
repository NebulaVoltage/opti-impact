/**
 * Deterministic Real-Time Vision Telemetry Simulator.
 *
 * Implements physically coupled oscillatory kinematics:
 *   Displacement -> Velocity -> Acceleration -> Resonant Frequency.
 *
 * NOTE: Operates in VISION-ONLY SIMULATION MODE. Values are synthetic and intended for
 * visualization testing and future telemetry adapter integration.
 */

import type {
    VisionTelemetry,
    ScenarioType,
    TrackingQuality,
    MeasurementStatus,
    FeaturePoint,
    SimulationEvent,
} from "../types/telemetry";
import { SCENARIO_PROFILES } from "./scenarios";

export type TelemetrySubscriber = (telemetry: VisionTelemetry) => void;
export type EventSubscriber = (event: SimulationEvent) => void;

export class VisionSimulator {
    private isRunning: boolean = false;
    private timerId: any = null;

    private time: number = 0.0;
    private readonly fps: number = 30.0;
    private readonly dt: number = 1.0 / 30.0;

    private currentScenario: ScenarioType = "NORMAL";

    // Dynamic state with exponential smoothing across transitions
    private currentAmplitude: number = 0.65;
    private currentFrequency: number = 5.15;
    private currentFeatureCount: number = 46;

    // Previous kinematic states for finite differentiation
    private prevDispX: number = 0.0;
    private prevDispY: number = 0.0;
    private prevVelX: number = 0.0;
    private prevVelY: number = 0.0;

    // Feature baseline distribution (50 points on the structural specimen)
    private readonly baseFeatures: Array<{ id: number; baseX: number; baseY: number }> = [];

    // Subscribed listeners
    private telemetrySubscribers: Set<TelemetrySubscriber> = new Set();
    private eventSubscribers: Set<EventSubscriber> = new Set();

    private lastTelemetry: VisionTelemetry;

    constructor() {
        // Initialize 50 irregular electrical tape corner coordinates across the beam specimen
        for (let i = 0; i < 50; i++) {
            const col = i % 10;
            const row = Math.floor(i / 10);
            // Add slight pseudo-random pseudo-irregularity
            const jitterX = Math.sin(i * 1.7) * 3.5;
            const jitterY = Math.cos(i * 2.3) * 4.0;
            const bx = 10 + col * 8.5 + jitterX;
            const by = 20 + row * 14.0 + jitterY;
            this.baseFeatures.push({ id: i + 1, baseX: bx, baseY: by });
        }

        this.lastTelemetry = this.computeStep(0.0);
    }

    /**
     * Set the current simulation scenario. Smoothly transitions amplitude and frequency.
     */
    public setScenario(scenario: ScenarioType): void {
        if (this.currentScenario !== scenario) {
            const old = this.currentScenario;
            this.currentScenario = scenario;
            this.emitEvent({
                id: `evt-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`,
                timestamp: this.formatTime(this.time),
                message: `Scenario transitioned: ${old} -> ${scenario}`,
                type: scenario === "CRITICAL" ? "critical" : scenario === "WARNING" ? "warning" : "info",
            });
        }
    }

    public getScenario(): ScenarioType {
        return this.currentScenario;
    }

    /**
     * Start real-time simulation updates.
     */
    public start(): void {
        if (this.isRunning) return;
        this.isRunning = true;
        this.emitEvent({
            id: `evt-${Date.now()}`,
            timestamp: this.formatTime(this.time),
            message: "Simulation stream started (30 FPS)",
            type: "info",
        });

        this.timerId = setInterval(() => {
            if (!this.isRunning) return;
            const tel = this.step(this.dt);
            this.notifyTelemetry(tel);
        }, Math.round(1000.0 / this.fps));
    }

    /**
     * Pause simulation updates.
     */
    public pause(): void {
        if (!this.isRunning) return;
        this.isRunning = false;
        if (this.timerId) {
            clearInterval(this.timerId);
            this.timerId = null;
        }
        this.emitEvent({
            id: `evt-${Date.now()}`,
            timestamp: this.formatTime(this.time),
            message: "Simulation stream paused",
            type: "info",
        });
    }

    /**
     * Reset simulation clock, position, and kinematics back to initial baseline.
     */
    public reset(): void {
        this.time = 0.0;
        const profile = SCENARIO_PROFILES[this.currentScenario];
        this.currentAmplitude = profile.targetDisplacement;
        this.currentFrequency = profile.targetFrequency;
        this.currentFeatureCount = profile.baseFeatureCount;

        this.prevDispX = 0.0;
        this.prevDispY = 0.0;
        this.prevVelX = 0.0;
        this.prevVelY = 0.0;

        const tel = this.computeStep(0.0);
        this.lastTelemetry = tel;
        this.notifyTelemetry(tel);

        this.emitEvent({
            id: `evt-${Date.now()}`,
            timestamp: this.formatTime(0),
            message: "Simulation reset to baseline zero",
            type: "info",
        });
    }

    public getIsRunning(): boolean {
        return this.isRunning;
    }

    public getTelemetry(): VisionTelemetry {
        return this.lastTelemetry;
    }

    /**
     * Compute a single simulation time step. Useful for headless test execution.
     */
    public step(dt: number = this.dt): VisionTelemetry {
        this.time += dt;
        const tel = this.computeStep(dt);
        this.lastTelemetry = tel;
        return tel;
    }

    private computeStep(dt: number): VisionTelemetry {
        const profile = SCENARIO_PROFILES[this.currentScenario];

        // Smoothly interpolate parameters towards scenario targets (avoid teleportation)
        const alpha = 0.06; // Smoothing factor per frame
        this.currentAmplitude += alpha * (profile.targetDisplacement - this.currentAmplitude);
        this.currentFrequency += alpha * (profile.targetFrequency - this.currentFrequency);
        this.currentFeatureCount += alpha * (profile.baseFeatureCount - this.currentFeatureCount);

        // Physics-based harmonic displacement with light damping and subtle noise
        const omega = 2.0 * Math.PI * this.currentFrequency;
        const phase = omega * this.time;

        // Bounded deterministic micro-noise
        const noiseX = (Math.sin(this.time * 37.1) + Math.cos(this.time * 53.3)) * 0.5 * profile.noiseAmplitude;
        const noiseY = (Math.cos(this.time * 41.7) - Math.sin(this.time * 61.2)) * 0.5 * profile.noiseAmplitude;

        // Primary flexural vibration along horizontal X, coupled secondary deflection along Y
        const dispX = this.currentAmplitude * Math.sin(phase) + noiseX;
        const dispY = 0.22 * this.currentAmplitude * Math.cos(phase) + noiseY;
        const dispMag = Math.sqrt(dispX * dispX + dispY * dispY);

        // Numerically consistent velocity: v = dx/dt (mm/s)
        let velX = 0.0;
        let velY = 0.0;
        if (dt > 1e-5) {
            velX = (dispX - this.prevDispX) / dt;
            velY = (dispY - this.prevDispY) / dt;
        }
        const velMag = Math.sqrt(velX * velX + velY * velY);

        // Numerically consistent acceleration: a = dv/dt / 1000 (m/s²)
        let accelX = 0.0;
        let accelY = 0.0;
        if (dt > 1e-5) {
            accelX = (velX - this.prevVelX) / (dt * 1000.0);
            accelY = (velY - this.prevVelY) / (dt * 1000.0);
        }
        const accelMag = Math.sqrt(accelX * accelX + accelY * accelY);

        this.prevDispX = dispX;
        this.prevDispY = dispY;
        this.prevVelX = velX;
        this.prevVelY = velY;

        // Feature count and tracking quality evaluation
        const activeCount = Math.round(Math.max(0, this.currentFeatureCount));
        let quality: TrackingQuality = profile.nominalTrackingQuality;
        let measStatus: MeasurementStatus = profile.nominalMeasurementStatus;
        let measValid: boolean = true;

        if (activeCount < 10) {
            quality = "LOST";
            measStatus = "TRACKING_LOST";
            measValid = false;
        } else if (activeCount < 25) {
            quality = "DEGRADED";
            measStatus = "VALID";
            measValid = true;
        } else {
            quality = "GOOD";
            measStatus = "VALID";
            measValid = true;
        }

        // Build feature points animated with structural motion
        // Specimen width ~ 400mm conceptual scale: 1mm deflection ~ 0.8% screen offset
        const scaleFactor = 0.65;
        const offsetX = dispX * scaleFactor;
        const offsetY = dispY * scaleFactor;

        const features: FeaturePoint[] = this.baseFeatures.map((bf, idx) => {
            const isValid = idx < activeCount;
            return {
                id: bf.id,
                baseX: bf.baseX,
                baseY: bf.baseY,
                currX: isValid ? bf.baseX + offsetX : bf.baseX,
                currY: isValid ? bf.baseY + offsetY : bf.baseY,
                valid: isValid,
            };
        });

        return {
            timestamp: parseFloat(this.time.toFixed(3)),
            displacementX: parseFloat(dispX.toFixed(3)),
            displacementY: parseFloat(dispY.toFixed(3)),
            displacementMagnitude: parseFloat(dispMag.toFixed(3)),
            velocityX: parseFloat(velX.toFixed(2)),
            velocityY: parseFloat(velY.toFixed(2)),
            velocityMagnitude: parseFloat(velMag.toFixed(2)),
            accelerationX: parseFloat(accelX.toFixed(3)),
            accelerationY: parseFloat(accelY.toFixed(3)),
            accelerationMagnitude: parseFloat(accelMag.toFixed(3)),
            dominantFrequency: parseFloat(this.currentFrequency.toFixed(2)),
            featureCount: activeCount,
            trackingQuality: quality,
            measurementValid: measValid,
            measurementStatus: measStatus,
            scenario: this.currentScenario,
            features,
        };
    }

    public subscribeTelemetry(callback: TelemetrySubscriber): () => void {
        this.telemetrySubscribers.add(callback);
        // Immediately invoke with current telemetry
        callback(this.lastTelemetry);
        return () => this.telemetrySubscribers.delete(callback);
    }

    public subscribeEvents(callback: EventSubscriber): () => void {
        this.eventSubscribers.add(callback);
        return () => this.eventSubscribers.delete(callback);
    }

    private notifyTelemetry(t: VisionTelemetry): void {
        for (const cb of this.telemetrySubscribers) {
            cb(t);
        }
    }

    private emitEvent(e: SimulationEvent): void {
        for (const cb of this.eventSubscribers) {
            cb(e);
        }
    }

    private formatTime(sec: number): string {
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60);
        const ms = Math.floor((sec % 1) * 100);
        return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}.${ms.toString().padStart(2, "0")}`;
    }
}
