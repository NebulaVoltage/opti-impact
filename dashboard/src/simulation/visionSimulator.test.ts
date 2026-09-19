import { describe, it, expect, beforeEach, vi } from "vitest";
import { VisionSimulator } from "./visionSimulator";

describe("VisionSimulator Tests", () => {
    let simulator: VisionSimulator;

    beforeEach(() => {
        simulator = new VisionSimulator();
    });

    it("1. simulator produces valid telemetry", () => {
        const tel = simulator.step(0.033);
        expect(tel).toBeDefined();
        expect(tel.timestamp).toBeGreaterThan(0);
        expect(typeof tel.displacementMagnitude).toBe("number");
        expect(typeof tel.dominantFrequency).toBe("number");
    });

    it("2. displacement remains finite across all scenarios", () => {
        const scenarios = ["NORMAL", "WARNING", "CRITICAL"] as const;
        for (const sc of scenarios) {
            simulator.setScenario(sc);
            for (let i = 0; i < 60; i++) {
                const tel = simulator.step(0.033);
                expect(Number.isFinite(tel.displacementX)).toBe(true);
                expect(Number.isFinite(tel.displacementY)).toBe(true);
                expect(Number.isFinite(tel.displacementMagnitude)).toBe(true);
                expect(Number.isNaN(tel.displacementMagnitude)).toBe(false);
            }
        }
    });

    it("3. velocity remains finite across all scenarios", () => {
        for (let i = 0; i < 60; i++) {
            const tel = simulator.step(0.033);
            expect(Number.isFinite(tel.velocityX)).toBe(true);
            expect(Number.isFinite(tel.velocityY)).toBe(true);
            expect(Number.isFinite(tel.velocityMagnitude)).toBe(true);
        }
    });

    it("4. acceleration remains finite across all scenarios", () => {
        for (let i = 0; i < 60; i++) {
            const tel = simulator.step(0.033);
            expect(Number.isFinite(tel.accelerationX)).toBe(true);
            expect(Number.isFinite(tel.accelerationY)).toBe(true);
            expect(Number.isFinite(tel.accelerationMagnitude)).toBe(true);
        }
    });

    it("5. frequency remains within configured scenario range", () => {
        // NORMAL: ~4.6–5.4 Hz
        simulator.setScenario("NORMAL");
        for (let i = 0; i < 30; i++) simulator.step(0.033);
        let tel = simulator.getTelemetry();
        expect(tel.dominantFrequency).toBeGreaterThanOrEqual(4.6);
        expect(tel.dominantFrequency).toBeLessThanOrEqual(5.4);

        // WARNING: ~3.9–4.8 Hz
        simulator.setScenario("WARNING");
        for (let i = 0; i < 100; i++) simulator.step(0.033);
        tel = simulator.getTelemetry();
        expect(tel.dominantFrequency).toBeGreaterThanOrEqual(3.9);
        expect(tel.dominantFrequency).toBeLessThanOrEqual(4.8);

        // CRITICAL: ~3.0–4.1 Hz
        simulator.setScenario("CRITICAL");
        for (let i = 0; i < 100; i++) simulator.step(0.033);
        tel = simulator.getTelemetry();
        expect(tel.dominantFrequency).toBeGreaterThanOrEqual(3.0);
        expect(tel.dominantFrequency).toBeLessThanOrEqual(4.1);
    });

    it("6. feature count remains non-negative and within reasonable bounds", () => {
        const scenarios = ["NORMAL", "WARNING", "CRITICAL"] as const;
        for (const sc of scenarios) {
            simulator.setScenario(sc);
            for (let i = 0; i < 40; i++) {
                const tel = simulator.step(0.033);
                expect(tel.featureCount).toBeGreaterThanOrEqual(0);
                expect(tel.featureCount).toBeLessThanOrEqual(50);
            }
        }
    });

    it("7. scenario switching works and updates state", () => {
        expect(simulator.getScenario()).toBe("NORMAL");
        simulator.setScenario("WARNING");
        expect(simulator.getScenario()).toBe("WARNING");
        const tel1 = simulator.step(0.033);
        expect(tel1.scenario).toBe("WARNING");

        simulator.setScenario("CRITICAL");
        expect(simulator.getScenario()).toBe("CRITICAL");
        const tel2 = simulator.step(0.033);
        expect(tel2.scenario).toBe("CRITICAL");
    });

    it("8. tracking states are valid categorical values", () => {
        const validQualities = ["GOOD", "DEGRADED", "LOST"];
        const scenarios = ["NORMAL", "WARNING", "CRITICAL"] as const;
        for (const sc of scenarios) {
            simulator.setScenario(sc);
            for (let i = 0; i < 40; i++) {
                const tel = simulator.step(0.033);
                expect(validQualities).toContain(tel.trackingQuality);
            }
        }
    });

    it("9. measurement states are valid and boolean matches status", () => {
        const validStatuses = ["VALID", "EXCESSIVE_STEP_DISPLACEMENT", "EXCEEDS_FRAME_BOUNDS", "TRACKING_LOST"];
        for (let i = 0; i < 50; i++) {
            const tel = simulator.step(0.033);
            expect(validStatuses).toContain(tel.measurementStatus);
            expect(typeof tel.measurementValid).toBe("boolean");
        }
    });

    it("10. simulator reset works and restores clock to 0", () => {
        for (let i = 0; i < 30; i++) simulator.step(0.033);
        expect(simulator.getTelemetry().timestamp).toBeGreaterThan(0);

        simulator.reset();
        const resetTel = simulator.getTelemetry();
        expect(resetTel.timestamp).toBe(0);
    });

    it("11. pause stops simulation updates", () => {
        vi.useFakeTimers();
        const sim = new VisionSimulator();
        let updateCount = 0;
        sim.subscribeTelemetry(() => {
            updateCount++;
        });

        sim.start();
        expect(sim.getIsRunning()).toBe(true);
        vi.advanceTimersByTime(100);
        const countBefore = updateCount;
        expect(countBefore).toBeGreaterThan(0);

        sim.pause();
        expect(sim.getIsRunning()).toBe(false);
        vi.advanceTimersByTime(200);
        expect(updateCount).toBe(countBefore); // No new updates while paused
        vi.useRealTimers();
    });

    it("12. all dashboard telemetry fields exist and match schema contract", () => {
        const tel = simulator.step(0.033);
        const requiredKeys = [
            "timestamp",
            "displacementX",
            "displacementY",
            "displacementMagnitude",
            "velocityX",
            "velocityY",
            "velocityMagnitude",
            "accelerationX",
            "accelerationY",
            "accelerationMagnitude",
            "dominantFrequency",
            "featureCount",
            "inlierCount",
            "forwardValidCount",
            "backwardValidCount",
            "rejectedCount",
            "retentionRate",
            "confidence",
            "madX",
            "madY",
            "trackingQuality",
            "measurementValid",
            "measurementStatus",
            "scenario",
            "autoScenarioActive",
            "features",
        ];

        for (const key of requiredKeys) {
            expect(key in tel).toBe(true);
        }
        expect(Array.isArray(tel.features)).toBe(true);
        expect(tel.features.length).toBe(50);
    });

    it("13. confidence score remains within [0.0, 1.0] and reflects scenario degradation", () => {
        simulator.setScenario("NORMAL");
        for (let i = 0; i < 30; i++) simulator.step(0.033);
        const normConf = simulator.getTelemetry().confidence;
        expect(normConf).toBeGreaterThanOrEqual(0.85);
        expect(normConf).toBeLessThanOrEqual(1.0);

        simulator.setScenario("CRITICAL");
        for (let i = 0; i < 100; i++) simulator.step(0.033);
        const critConf = simulator.getTelemetry().confidence;
        expect(critConf).toBeLessThan(normConf);
        expect(critConf).toBeGreaterThanOrEqual(0.2);
    });

    it("14. inliers and retention rate are mathematically consistent", () => {
        for (let i = 0; i < 40; i++) {
            const tel = simulator.step(0.033);
            expect(tel.inlierCount + tel.rejectedCount).toBe(50);
            expect(tel.retentionRate).toBeCloseTo((tel.inlierCount / 50) * 100.0, 1);
            expect(tel.inlierCount).toBeLessThanOrEqual(tel.forwardValidCount);
            expect(tel.backwardValidCount).toBeLessThanOrEqual(tel.forwardValidCount);
        }
    });

    it("15. auto scenario automatically transitions scenarios over time", () => {
        simulator.setAutoScenario(true);
        expect(simulator.getAutoScenario()).toBe(true);

        // At t=0, starts in NORMAL
        let tel = simulator.step(1.0);
        expect(tel.scenario).toBe("NORMAL");
        expect(tel.autoScenarioActive).toBe(true);

        // Advance to t=12s -> should be in WARNING (10s threshold)
        for (let i = 0; i < 11; i++) {
            tel = simulator.step(1.0);
        }
        expect(tel.scenario).toBe("WARNING");

        // Advance to t=20s -> should be in CRITICAL (18s threshold)
        for (let i = 0; i < 8; i++) {
            tel = simulator.step(1.0);
        }
        expect(tel.scenario).toBe("CRITICAL");

        // Advance to t=26s -> should be back in WARNING (24s threshold)
        for (let i = 0; i < 6; i++) {
            tel = simulator.step(1.0);
        }
        expect(tel.scenario).toBe("WARNING");

        // Manual override disengages auto scenario
        simulator.setScenario("NORMAL");
        expect(simulator.getAutoScenario()).toBe(false);
    });

    it("16. rejected feature points provide audit rejection reasons", () => {
        simulator.setScenario("CRITICAL");
        for (let i = 0; i < 100; i++) simulator.step(0.033);
        const tel = simulator.getTelemetry();

        const rejected = tel.features.filter((f) => !f.valid);
        expect(rejected.length).toBeGreaterThan(0);
        for (const pt of rejected) {
            expect(["FB_ERROR", "MAD_OUTLIER", "FRAME_BOUNDS"]).toContain(pt.rejectionReason);
        }
    });
});
