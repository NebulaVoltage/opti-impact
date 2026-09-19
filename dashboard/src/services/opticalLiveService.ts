/**
 * Client service connecting the React Dashboard to the physical Python Optical Backend.
 *
 * Manages:
 *   - Real-time WebSocket connection to /ws/optical
 *   - Automatic reconnection on network interruption
 *   - Telemetry normalization (pixel units)
 *   - Baseline capture and recentering REST actions
 */

import type {
    VisionTelemetry,
    LiveTelemetryPacket,
    BaselineRecord,
} from "../types/telemetry";

export type TelemetryCallback = (telemetry: VisionTelemetry) => void;
export type StatusCallback = (connected: boolean, error?: string) => void;

export class OpticalLiveService {
    private ws: WebSocket | null = null;
    private baseUrl: string;
    private wsUrl: string;
    private shouldReconnect: boolean = true;
    private reconnectTimer: any = null;

    private telemetryCallbacks: Set<TelemetryCallback> = new Set();
    private statusCallbacks: Set<StatusCallback> = new Set();

    private isConnected: boolean = false;
    public lastPacketTime: number = 0;

    constructor(host: string = "localhost:8000") {
        this.baseUrl = `http://${host}`;
        this.wsUrl = `ws://${host}/ws/optical`;
    }

    public connect(): void {
        this.shouldReconnect = true;
        this.initWebSocket();
    }

    public disconnect(): void {
        this.shouldReconnect = false;
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.setConnected(false);
    }

    public onTelemetry(cb: TelemetryCallback): () => void {
        this.telemetryCallbacks.add(cb);
        return () => this.telemetryCallbacks.delete(cb);
    }

    public onStatus(cb: StatusCallback): () => void {
        this.statusCallbacks.add(cb);
        cb(this.isConnected);
        return () => this.statusCallbacks.delete(cb);
    }

    public getIsConnected(): boolean {
        return this.isConnected;
    }

    public async checkHealth(): Promise<{ connected: boolean; cameraIndex?: number; frameCount?: number; error?: string }> {
        try {
            const resp = await fetch(`${this.baseUrl}/api/health`, { method: "GET" });
            if (!resp.ok) {
                return { connected: false, error: `HTTP ${resp.status}` };
            }
            const data = await resp.json();
            return {
                connected: data.camera_connected,
                cameraIndex: data.camera_index,
                frameCount: data.frame_count,
                error: data.connection_error,
            };
        } catch (e: any) {
            return { connected: false, error: e.message || "Failed to reach backend." };
        }
    }

    public async resetBaseline(): Promise<boolean> {
        try {
            const resp = await fetch(`${this.baseUrl}/api/reset`, { method: "POST" });
            return resp.ok;
        } catch {
            return false;
        }
    }

    public async captureBaseline(): Promise<BaselineRecord | null> {
        try {
            const resp = await fetch(`${this.baseUrl}/api/baseline`, { method: "POST" });
            if (!resp.ok) return null;
            const data = await resp.json();
            return data.baseline || null;
        } catch {
            return null;
        }
    }

    public async getBaseline(): Promise<BaselineRecord | null> {
        try {
            const resp = await fetch(`${this.baseUrl}/api/baseline`, { method: "GET" });
            if (!resp.ok) return null;
            const data = await resp.json();
            return data.baseline || null;
        } catch {
            return null;
        }
    }

    private initWebSocket(): void {
        if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
            return;
        }

        try {
            this.ws = new WebSocket(this.wsUrl);

            this.ws.onopen = () => {
                this.setConnected(true);
            };

            this.ws.onmessage = (event) => {
                try {
                    const raw: LiveTelemetryPacket = JSON.parse(event.data);
                    this.lastPacketTime = Date.now();
                    const normalized = this.normalizePacket(raw);
                    for (const cb of this.telemetryCallbacks) {
                        cb(normalized);
                    }
                } catch {
                    // Ignore malformed packet
                }
            };

            this.ws.onerror = () => {
                this.setConnected(false, "WebSocket connection error");
            };

            this.ws.onclose = () => {
                this.setConnected(false);
                this.ws = null;
                if (this.shouldReconnect) {
                    this.reconnectTimer = setTimeout(() => this.initWebSocket(), 1500);
                }
            };
        } catch (e: any) {
            this.setConnected(false, e.message);
            if (this.shouldReconnect) {
                this.reconnectTimer = setTimeout(() => this.initWebSocket(), 2000);
            }
        }
    }

    private setConnected(connected: boolean, error?: string): void {
        this.isConnected = connected;
        for (const cb of this.statusCallbacks) {
            cb(connected, error);
        }
    }

    private normalizePacket(raw: LiveTelemetryPacket): VisionTelemetry {
        return {
            timestamp: raw.timestamp,
            displacementX: raw.displacement_x_px,
            displacementY: raw.displacement_y_px,
            displacementMagnitude: raw.displacement_magnitude_px,
            velocityX: raw.velocity_x_px_s,
            velocityY: raw.velocity_y_px_s,
            velocityMagnitude: raw.velocity_magnitude_px_s,
            accelerationX: raw.acceleration_x_px_s2,
            accelerationY: raw.acceleration_y_px_s2,
            accelerationMagnitude: raw.acceleration_magnitude_px_s2,
            dominantFrequency: raw.dominant_frequency_hz,
            featureCount: raw.feature_count,
            inlierCount: raw.inlier_feature_count,
            forwardValidCount: raw.feature_count,
            backwardValidCount: raw.inlier_feature_count,
            rejectedCount: raw.rejected_feature_count,
            retentionRate: raw.retention_rate,
            confidence: raw.confidence,
            madX: raw.mad_x_px,
            madY: raw.mad_y_px,
            trackingQuality: raw.tracking_quality,
            measurementValid: raw.measurement_valid,
            measurementStatus: raw.validity_reason,
            scenario: "NORMAL", // Physical run has no simulated scenario
            autoScenarioActive: false,
            isLive: true,
            fps: raw.fps,
            unit: "px",
            velocityUnit: "px/s",
            accelerationUnit: "px/s²",
            fftFrequencies: raw.fft_frequencies || [],
            fftAmplitudes: raw.fft_amplitudes || [],
            peakFrequencyHz: raw.peak_frequency_hz,
            peakAmplitude: raw.peak_amplitude,
            baselineFrequencyHz: raw.baseline_frequency_hz,
            frequencyDeviationHz: raw.frequency_deviation_hz,
        };
    }
}
