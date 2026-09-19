import React, { useRef, useEffect } from "react";
import type { VisionTelemetry } from "../types/telemetry";

interface StructuralViewerProps {
    telemetry: VisionTelemetry;
}

export const StructuralViewer: React.FC<StructuralViewerProps> = ({ telemetry }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const w = canvas.width;
        const h = canvas.height;

        // Clear background
        ctx.fillStyle = "#0c1017";
        ctx.fillRect(0, 0, w, h);

        // 1. Draw Camera Grid / Crosshairs
        ctx.strokeStyle = "rgba(45, 60, 80, 0.4)";
        ctx.lineWidth = 1;
        const gridSize = 40;
        for (let x = 0; x < w; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }
        for (let y = 0; y < h; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }

        // Center crosshair
        ctx.strokeStyle = "rgba(100, 140, 180, 0.5)";
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(w / 2, 0);
        ctx.lineTo(w / 2, h);
        ctx.moveTo(0, h / 2);
        ctx.lineTo(w, h / 2);
        ctx.stroke();
        ctx.setLineDash([]);

        // 2. Specimen ROI Boundary (Yellow Rectangle)
        const roiMarginX = 50;
        const roiMarginY = 40;
        const roiW = w - roiMarginX * 2;
        const roiH = h - roiMarginY * 2;

        ctx.strokeStyle = "#e6b800";
        ctx.lineWidth = 2;
        ctx.strokeRect(roiMarginX, roiMarginY, roiW, roiH);

        ctx.fillStyle = "#e6b800";
        ctx.font = "11px 'JetBrains Mono', monospace";
        ctx.fillText("SPECIMEN ROI [1280x720 VIRTUAL]", roiMarginX + 8, roiMarginY + 16);

        // 3. Draw Simulated Structural Cardboard Specimen
        // Amplified visual displacement for display: 1mm -> 4px
        const visualScale = 3.8;
        const dispPxX = telemetry.displacementX * visualScale;
        const dispPxY = telemetry.displacementY * visualScale;

        // Beam dimensions
        const beamW = roiW * 0.76;
        const beamH = roiH * 0.55;
        const beamX = roiMarginX + (roiW - beamW) / 2 + dispPxX;
        const beamY = roiMarginY + (roiH - beamH) / 2 + dispPxY;

        // Draw structural specimen shadow / baseline origin reference
        ctx.strokeStyle = "rgba(120, 120, 120, 0.25)";
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        const originX = roiMarginX + (roiW - beamW) / 2;
        const originY = roiMarginY + (roiH - beamH) / 2;
        ctx.strokeRect(originX, originY, beamW, beamH);
        ctx.setLineDash([]);

        // Draw Cardboard Specimen Body
        const gradient = ctx.createLinearGradient(beamX, beamY, beamX, beamY + beamH);
        gradient.addColorStop(0, "#b88a58");
        gradient.addColorStop(0.5, "#a67844");
        gradient.addColorStop(1, "#8a5e2f");
        ctx.fillStyle = gradient;
        ctx.fillRect(beamX, beamY, beamW, beamH);

        // Specimen border
        ctx.strokeStyle = "#d4a770";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(beamX, beamY, beamW, beamH);

        // Specimen label
        ctx.fillStyle = "rgba(255, 255, 255, 0.45)";
        ctx.font = "bold 13px 'JetBrains Mono', monospace";
        ctx.fillText("FLEXURAL STRUCTURAL SPECIMEN (CARDBOARD)", beamX + 16, beamY + 28);

        // 4. Draw Irregular Black Electrical Tape Markers
        // 4 prominent irregular tape patches across the beam
        const tapeColor = "#1a1a1a";
        const tapeBorder = "#333333";
        ctx.fillStyle = tapeColor;
        ctx.strokeStyle = tapeBorder;
        ctx.lineWidth = 1;

        const tapePatches = [
            { x: beamX + beamW * 0.12, y: beamY + beamH * 0.25, w: 45, h: 32 },
            { x: beamX + beamW * 0.35, y: beamY + beamH * 0.45, w: 55, h: 28 },
            { x: beamX + beamW * 0.58, y: beamY + beamH * 0.20, w: 40, h: 42 },
            { x: beamX + beamW * 0.78, y: beamY + beamH * 0.50, w: 50, h: 30 },
        ];

        for (const tp of tapePatches) {
            ctx.fillRect(tp.x, tp.y, tp.w, tp.h);
            ctx.strokeRect(tp.x, tp.y, tp.w, tp.h);
        }

        // 5. Draw Tracked Optical Features & Lucas-Kanade Motion Vectors
        const features = telemetry.features || [];
        for (const feat of features) {
            if (!feat.valid) continue;

            const px = roiMarginX + (feat.currX / 100.0) * roiW;
            const py = roiMarginY + (feat.currY / 100.0) * roiH;

            // Green feature circle (Lucas-Kanade tracked corner)
            ctx.beginPath();
            ctx.arc(px, py, 3.5, 0, 2 * Math.PI);
            ctx.fillStyle = telemetry.trackingQuality === "GOOD" ? "#00ff66" : "#ffaa00";
            ctx.fill();
            ctx.strokeStyle = "#003311";
            ctx.lineWidth = 1;
            ctx.stroke();

            // Optical flow motion vector arrow
            const vecScale = 2.2;
            const arrowDx = -telemetry.velocityX * 0.05 * vecScale;
            const arrowDy = -telemetry.velocityY * 0.05 * vecScale;

            if (Math.abs(arrowDx) > 0.8 || Math.abs(arrowDy) > 0.8) {
                ctx.beginPath();
                ctx.moveTo(px, py);
                ctx.lineTo(px + arrowDx, py + arrowDy);
                ctx.strokeStyle = "#ff3333";
                ctx.lineWidth = 1.2;
                ctx.stroke();
            }
        }

        // 6. Live HUD Info in Viewer
        ctx.fillStyle = "rgba(10, 15, 24, 0.85)";
        ctx.fillRect(roiMarginX + 8, roiMarginY + roiH - 50, 280, 42);
        ctx.strokeStyle = "rgba(100, 180, 255, 0.3)";
        ctx.strokeRect(roiMarginX + 8, roiMarginY + roiH - 50, 280, 42);

        ctx.fillStyle = "#aaccff";
        ctx.font = "11px 'JetBrains Mono', monospace";
        ctx.fillText(`Deflection: X=${telemetry.displacementX > 0 ? "+" : ""}${telemetry.displacementX.toFixed(2)} mm  Y=${telemetry.displacementY.toFixed(2)} mm`, roiMarginX + 16, roiMarginY + roiH - 32);
        ctx.fillText(`Active Tracked Features: ${telemetry.featureCount} / 50 corners`, roiMarginX + 16, roiMarginY + roiH - 16);

        // Motion status badge
        const scenarioColor =
            telemetry.scenario === "CRITICAL"
                ? "#ff3b30"
                : telemetry.scenario === "WARNING"
                ? "#ff9500"
                : "#34c759";
        ctx.fillStyle = scenarioColor;
        ctx.font = "bold 11px 'JetBrains Mono', monospace";
        ctx.fillText(`[${telemetry.scenario} RESPONSE]`, roiMarginX + roiW - 160, roiMarginY + 20);

    }, [telemetry]);

    return (
        <div className="structural-viewer-panel">
            <div className="panel-header">
                <div className="panel-title">
                    <span className="live-dot"></span>
                    OPTICAL STRUCTURAL MOTION VIEW (2D TRACKING)
                </div>
                <div className="panel-badge">
                    METHOD: SHI–TOMASI + LUCAS–KANADE (SIMULATED)
                </div>
            </div>
            <div className="canvas-wrapper">
                <canvas
                    ref={canvasRef}
                    width={720}
                    height={400}
                    className="structural-canvas"
                />
            </div>
            <div className="viewer-footer">
                <div className="legend-item">
                    <span className="legend-dot dot-green"></span> Tracked Corner Features
                </div>
                <div className="legend-item">
                    <span className="legend-dot dot-red"></span> Optical Flow Vectors
                </div>
                <div className="legend-item">
                    <span className="legend-dot dot-yellow"></span> Specimen ROI
                </div>
                <div className="legend-item">
                    <span className="legend-dot dot-dashed"></span> Baseline Reference
                </div>
            </div>
        </div>
    );
};
