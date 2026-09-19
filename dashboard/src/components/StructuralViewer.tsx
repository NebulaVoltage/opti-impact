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

        // 1. Clear background & camera sensor noise texture
        ctx.fillStyle = "#080c14";
        ctx.fillRect(0, 0, w, h);

        // Subtle sensor grid
        ctx.strokeStyle = "rgba(35, 50, 75, 0.35)";
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

        // Camera Frame Boundary (1280x720 conceptual aspect ratio)
        const frameMarginX = 18;
        const frameMarginY = 16;
        const frameW = w - frameMarginX * 2;
        const frameH = h - frameMarginY * 2;

        ctx.strokeStyle = "rgba(0, 212, 255, 0.4)";
        ctx.lineWidth = 1.2;
        ctx.strokeRect(frameMarginX, frameMarginY, frameW, frameH);

        // Corner framing brackets
        const bracketSize = 16;
        ctx.strokeStyle = "#00d4ff";
        ctx.lineWidth = 2.5;

        // Top-left
        ctx.beginPath();
        ctx.moveTo(frameMarginX, frameMarginY + bracketSize);
        ctx.lineTo(frameMarginX, frameMarginY);
        ctx.lineTo(frameMarginX + bracketSize, frameMarginY);
        ctx.stroke();

        // Top-right
        ctx.beginPath();
        ctx.moveTo(frameMarginX + frameW - bracketSize, frameMarginY);
        ctx.lineTo(frameMarginX + frameW, frameMarginY);
        ctx.lineTo(frameMarginX + frameW, frameMarginY + bracketSize);
        ctx.stroke();

        // Bottom-left
        ctx.beginPath();
        ctx.moveTo(frameMarginX, frameMarginY + frameH - bracketSize);
        ctx.lineTo(frameMarginX, frameMarginY + frameH);
        ctx.lineTo(frameMarginX + bracketSize, frameMarginY + frameH);
        ctx.stroke();

        // Bottom-right
        ctx.beginPath();
        ctx.moveTo(frameMarginX + frameW - bracketSize, frameMarginY + frameH);
        ctx.lineTo(frameMarginX + frameW, frameMarginY + frameH);
        ctx.lineTo(frameMarginX + frameW, frameMarginY + frameH - bracketSize);
        ctx.stroke();

        // Center Optical Crosshairs
        const cx = w / 2;
        const cy = h / 2;
        ctx.strokeStyle = "rgba(100, 160, 220, 0.3)";
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(cx, frameMarginY);
        ctx.lineTo(cx, frameMarginY + frameH);
        ctx.moveTo(frameMarginX, cy);
        ctx.lineTo(frameMarginX + frameW, cy);
        ctx.stroke();
        ctx.setLineDash([]);

        // Small center reticle circle
        ctx.beginPath();
        ctx.arc(cx, cy, 14, 0, 2 * Math.PI);
        ctx.strokeStyle = "rgba(0, 212, 255, 0.35)";
        ctx.stroke();

        // Camera Framing Header HUD
        ctx.fillStyle = "rgba(0, 212, 255, 0.85)";
        ctx.font = "9px 'JetBrains Mono', monospace";
        ctx.fillText("CAM-01 [1280 × 720 HD] • FOV 78° • COLD CMOS STREAM • 30.0 FPS", frameMarginX + 24, frameMarginY + 12);

        // 2. Specimen ROI Boundary (Yellow Rectangle)
        const roiMarginX = 54;
        const roiMarginY = 48;
        const roiW = w - roiMarginX * 2;
        const roiH = h - roiMarginY * 2;

        ctx.strokeStyle = "rgba(230, 184, 0, 0.85)";
        ctx.lineWidth = 1.8;
        ctx.strokeRect(roiMarginX, roiMarginY, roiW, roiH);

        ctx.fillStyle = "#e6b800";
        ctx.font = "bold 10px 'JetBrains Mono', monospace";
        ctx.fillText("SPECIMEN ROI [1080 × 520 VIRTUAL px]", roiMarginX + 10, roiMarginY + 14);

        // 3. Structural Support Piers (Fixed Base Left, Roller Support Right)
        const pierW = 38;
        const pierTopY = roiMarginY + roiH * 0.76;
        const pierH = roiMarginY + roiH - pierTopY + 8;
        const leftPierX = roiMarginX + roiW * 0.08;
        const rightPierX = roiMarginX + roiW * 0.84;

        // Pier gradient
        const pierGrad = ctx.createLinearGradient(0, pierTopY, 0, pierTopY + pierH);
        pierGrad.addColorStop(0, "#2d3748");
        pierGrad.addColorStop(1, "#1a202c");

        // Left Pier (Fixed Pier)
        ctx.fillStyle = pierGrad;
        ctx.fillRect(leftPierX, pierTopY, pierW, pierH);
        ctx.strokeStyle = "#4a5568";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(leftPierX, pierTopY, pierW, pierH);

        // Fixed pier anchor plate and ground hatching
        ctx.fillStyle = "#1e2430";
        ctx.fillRect(leftPierX - 8, pierTopY + pierH - 6, pierW + 16, 6);
        ctx.strokeStyle = "#718096";
        ctx.strokeRect(leftPierX - 8, pierTopY + pierH - 6, pierW + 16, 6);

        ctx.fillStyle = "#a0aec0";
        ctx.font = "8px 'JetBrains Mono', monospace";
        ctx.textAlign = "center";
        ctx.fillText("FIXED PIER", leftPierX + pierW / 2, pierTopY + pierH + 10);

        // Right Pier (Roller Bearing Support)
        ctx.fillStyle = pierGrad;
        ctx.fillRect(rightPierX, pierTopY, pierW, pierH);
        ctx.strokeStyle = "#4a5568";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(rightPierX, pierTopY, pierW, pierH);

        // Cylindrical Rollers under the beam
        const rollerRadius = 4.5;
        const rollerCenterY = pierTopY - rollerRadius - 1;
        ctx.fillStyle = "#cbd5e0";
        for (let r = 0; r < 3; r++) {
            const rx = rightPierX + 7 + r * 12;
            ctx.beginPath();
            ctx.arc(rx, rollerCenterY, rollerRadius, 0, 2 * Math.PI);
            ctx.fill();
            ctx.strokeStyle = "#4a5568";
            ctx.stroke();
        }

        ctx.fillStyle = "#a0aec0";
        ctx.font = "8px 'JetBrains Mono', monospace";
        ctx.fillText("ROLLER PIER", rightPierX + pierW / 2, pierTopY + pierH + 10);
        ctx.textAlign = "left";

        // 4. Draw Simulated Structural Cardboard Specimen
        // Amplified visual displacement for display: 1mm -> 4px
        const visualScale = 3.6;
        const dispPxX = telemetry.displacementX * visualScale;
        const dispPxY = telemetry.displacementY * visualScale;

        // Beam dimensions
        const beamW = roiW * 0.78;
        const beamH = roiH * 0.46;
        const originX = roiMarginX + (roiW - beamW) / 2;
        const originY = roiMarginY + (roiH - beamH) / 2 - 12;

        const beamX = originX + dispPxX;
        const beamY = originY + dispPxY;

        // Draw structural specimen shadow / baseline origin reference (unbent origin)
        ctx.strokeStyle = "rgba(160, 174, 192, 0.22)";
        ctx.lineWidth = 1.2;
        ctx.setLineDash([3, 3]);
        ctx.strokeRect(originX, originY, beamW, beamH);
        ctx.setLineDash([]);

        ctx.fillStyle = "rgba(160, 174, 192, 0.35)";
        ctx.font = "9px 'JetBrains Mono', monospace";
        ctx.fillText("BASELINE ZERO REFERENCE", originX + 8, originY - 4);

        // Draw Cardboard Specimen Body with realistic material shading
        const beamGrad = ctx.createLinearGradient(beamX, beamY, beamX, beamY + beamH);
        beamGrad.addColorStop(0, "#c29560");
        beamGrad.addColorStop(0.35, "#ab7c47");
        beamGrad.addColorStop(0.75, "#936636");
        beamGrad.addColorStop(1, "#7d5225");

        ctx.fillStyle = beamGrad;
        ctx.fillRect(beamX, beamY, beamW, beamH);

        // Specimen outer border
        ctx.strokeStyle = "#deb887";
        ctx.lineWidth = 1.6;
        ctx.strokeRect(beamX, beamY, beamW, beamH);

        // Neutral axis dashed centerline across the beam
        ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(beamX, beamY + beamH / 2);
        ctx.lineTo(beamX + beamW, beamY + beamH / 2);
        ctx.stroke();
        ctx.setLineDash([]);

        // Specimen identification label
        ctx.fillStyle = "rgba(255, 255, 255, 0.55)";
        ctx.font = "bold 11px 'JetBrains Mono', monospace";
        ctx.fillText("FLEXURAL STRUCTURAL TEST SPECIMEN (CARDBOARD BOX-BEAM)", beamX + 16, beamY + 22);

        // 5. Draw Irregular Black Electrical Tape Markers
        const tapeColor = "#121418";
        const tapeBorder = "#282c34";
        ctx.fillStyle = tapeColor;
        ctx.strokeStyle = tapeBorder;
        ctx.lineWidth = 1.2;

        const tapePatches = [
            { x: beamX + beamW * 0.10, y: beamY + beamH * 0.22, w: 48, h: 36 },
            { x: beamX + beamW * 0.33, y: beamY + beamH * 0.44, w: 60, h: 32 },
            { x: beamX + beamW * 0.56, y: beamY + beamH * 0.18, w: 46, h: 48 },
            { x: beamX + beamW * 0.77, y: beamY + beamH * 0.48, w: 54, h: 34 },
        ];

        for (const tp of tapePatches) {
            ctx.fillRect(tp.x, tp.y, tp.w, tp.h);
            ctx.strokeRect(tp.x, tp.y, tp.w, tp.h);

            // Subtle tape sheen line
            ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
            ctx.beginPath();
            ctx.moveTo(tp.x + 3, tp.y + 4);
            ctx.lineTo(tp.x + tp.w - 3, tp.y + 4);
            ctx.stroke();
        }

        // 6. Draw Tracked Optical Features (Shi–Tomasi Corners) & Lucas–Kanade Motion Vectors
        const features = telemetry.features || [];
        for (const feat of features) {
            const px = roiMarginX + (feat.currX / 100.0) * roiW;
            const py = roiMarginY + (feat.currY / 100.0) * roiH;

            if (feat.valid) {
                // Inlier feature point: vivid green with dark outer ring
                ctx.beginPath();
                ctx.arc(px, py, 3.8, 0, 2 * Math.PI);
                ctx.fillStyle = telemetry.trackingQuality === "GOOD" ? "#00ff66" : "#ffaa00";
                ctx.fill();
                ctx.strokeStyle = "#003311";
                ctx.lineWidth = 1.2;
                ctx.stroke();

                // Small center highlight
                ctx.beginPath();
                ctx.arc(px, py, 1.2, 0, 2 * Math.PI);
                ctx.fillStyle = "#ffffff";
                ctx.fill();

                // Dynamic optical flow motion vector arrow
                const vecScale = 2.4;
                const arrowDx = -telemetry.velocityX * 0.05 * vecScale;
                const arrowDy = -telemetry.velocityY * 0.05 * vecScale;

                if (Math.abs(arrowDx) > 0.6 || Math.abs(arrowDy) > 0.6) {
                    ctx.beginPath();
                    ctx.moveTo(px, py);
                    ctx.lineTo(px + arrowDx, py + arrowDy);
                    ctx.strokeStyle = "#ff3333";
                    ctx.lineWidth = 1.4;
                    ctx.stroke();

                    // Arrowhead
                    const angle = Math.atan2(arrowDy, arrowDx);
                    const headLen = 4;
                    ctx.beginPath();
                    ctx.moveTo(px + arrowDx, py + arrowDy);
                    ctx.lineTo(
                        px + arrowDx - headLen * Math.cos(angle - Math.PI / 6),
                        py + arrowDy - headLen * Math.sin(angle - Math.PI / 6)
                    );
                    ctx.moveTo(px + arrowDx, py + arrowDy);
                    ctx.lineTo(
                        px + arrowDx - headLen * Math.cos(angle + Math.PI / 6),
                        py + arrowDy - headLen * Math.sin(angle + Math.PI / 6)
                    );
                    ctx.stroke();
                }
            } else {
                // Rejected outlier feature: small amber/red indicator
                ctx.beginPath();
                ctx.arc(px, py, 3.0, 0, 2 * Math.PI);
                ctx.fillStyle = "rgba(255, 59, 48, 0.6)";
                ctx.fill();
                ctx.strokeStyle = "#ff3b30";
                ctx.lineWidth = 1;
                ctx.stroke();

                // Small "x" mark
                const s = 2.2;
                ctx.beginPath();
                ctx.moveTo(px - s, py - s);
                ctx.lineTo(px + s, py + s);
                ctx.moveTo(px + s, py - s);
                ctx.lineTo(px - s, py + s);
                ctx.strokeStyle = "#ffffff";
                ctx.stroke();
            }
        }

        // 7. Live HUD Status Overlay in Viewer (Bottom Left)
        const hudW = 310;
        const hudH = 50;
        const hudX = roiMarginX + 8;
        const hudY = roiMarginY + roiH - hudH - 8;

        ctx.fillStyle = "rgba(10, 16, 26, 0.9)";
        ctx.fillRect(hudX, hudY, hudW, hudH);
        ctx.strokeStyle = "rgba(0, 212, 255, 0.35)";
        ctx.lineWidth = 1.2;
        ctx.strokeRect(hudX, hudY, hudW, hudH);

        ctx.fillStyle = "#aaccff";
        ctx.font = "10px 'JetBrains Mono', monospace";
        ctx.fillText(
            `Deflection: X=${telemetry.displacementX > 0 ? "+" : ""}${telemetry.displacementX.toFixed(2)} mm  Y=${telemetry.displacementY.toFixed(2)} mm (Mag: ${telemetry.displacementMagnitude.toFixed(2)} mm)`,
            hudX + 10,
            hudY + 16
        );
        ctx.fillText(
            `Optical Inliers: ${telemetry.inlierCount} / 50 corners (${telemetry.retentionRate.toFixed(1)}% retention)`,
            hudX + 10,
            hudY + 31
        );
        ctx.fillText(
            `Tracking Quality: ${telemetry.trackingQuality} • Confidence: ${(telemetry.confidence * 100).toFixed(1)}%`,
            hudX + 10,
            hudY + 44
        );

        // Motion status badge (Top Right of ROI)
        const scenarioColor =
            telemetry.scenario === "CRITICAL"
                ? "#ff3b30"
                : telemetry.scenario === "WARNING"
                ? "#ff9500"
                : "#34c759";
        ctx.fillStyle = scenarioColor;
        ctx.font = "bold 11px 'JetBrains Mono', monospace";
        ctx.fillText(`[${telemetry.scenario} RESPONSE STATE]`, roiMarginX + roiW - 195, roiMarginY + 20);

    }, [telemetry]);

    return (
        <div className="structural-viewer-panel">
            <div className="panel-header">
                <div className="panel-title">
                    <span className="live-dot"></span>
                    OPTICAL STRUCTURAL MOTION VIEW (2D TRACKING)
                </div>
                <div className="panel-badge">
                    METHOD: SHI–TOMASI + BIDIRECTIONAL LUCAS–KANADE (SIMULATED)
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
                    <span className="legend-dot dot-green"></span> Tracked Corner Inliers
                </div>
                <div className="legend-item">
                    <span className="legend-dot dot-red"></span> LK Motion Flow Vectors
                </div>
                <div className="legend-item">
                    <span className="legend-dot dot-rejected"></span> FB / MAD Rejected Outliers
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
