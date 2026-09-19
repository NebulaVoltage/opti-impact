import React, { useState } from "react";

export const OpticsEducationalPanel: React.FC = () => {
    const [isOpen, setIsOpen] = useState<boolean>(true);

    const steps = [
        {
            num: "01",
            title: "Camera Framing & Calibrated ROI",
            detail: "CMOS optical sensor captures 1280×720 frames at 30 FPS. A bounding Region of Interest (ROI) isolates the structural specimen, discarding background visual noise.",
        },
        {
            num: "02",
            title: "Shi–Tomasi Feature Corner Detection",
            detail: "Good Features to Track analyzes image gradients (minimum eigenvalues) on high-contrast tape markers, identifying stable 2D visual anchor points across the beam.",
        },
        {
            num: "03",
            title: "Bidirectional Lucas–Kanade Flow",
            detail: "Pyramidal optical flow estimates frame-to-frame motion (t → t+1). Reverse tracking (t+1 → recon) validates flow consistency: points with Forward-Backward error > 1.5 px are rejected.",
        },
        {
            num: "04",
            title: "Robust MAD Spatial Outlier Filtering",
            detail: "Feature displacements are aggregated using Median Absolute Deviation (MAD). Outlier points deviating beyond 2.5×MAD from coherent structural motion are pruned.",
        },
        {
            num: "05",
            title: "Reference-Relative Zero-Drift Tracking",
            detail: "Subpixel coordinates are referenced against initial zero-deflection baseline coordinates (P_curr - P_ref), preventing compounding cumulative drift across prolonged monitoring.",
        },
        {
            num: "06",
            title: "Kinematic Differentiation & FFT Modal Analysis",
            detail: "Numerical finite differences calculate velocity (dx/dt) and acceleration (dv/dt). Fast Fourier Transform (FFT) tracks natural resonant frequency shifts indicating stiffness changes.",
        },
    ];

    return (
        <div className="optics-educational-panel">
            <div className="edu-header" onClick={() => setIsOpen(!isOpen)}>
                <div className="edu-title-group">
                    <span className="edu-icon">🔬</span>
                    <span className="edu-title">HOW OPTICAL STRUCTURAL SENSING WORKS</span>
                </div>
                <div className="edu-toggle-btn">
                    {isOpen ? "COLLAPSE ▲" : "EXPAND PIPELINE DETAILS ▼"}
                </div>
            </div>

            {isOpen && (
                <div className="edu-content">
                    <div className="edu-intro-text">
                        Marker-assisted optical structural monitoring converts camera video into precision kinematic telemetry without attaching heavy mechanical sensors directly to the structure.
                    </div>

                    <div className="edu-steps-grid">
                        {steps.map((st) => (
                            <div key={st.num} className="edu-step-card">
                                <div className="step-num-badge">{st.num}</div>
                                <div className="step-body">
                                    <div className="step-title">{st.title}</div>
                                    <div className="step-detail">{st.detail}</div>
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="edu-notice">
                        <strong>Simulated Architecture:</strong> This dashboard demonstrates the visualization pipeline for the Step 6A optical kinematics engine. Telemetry is synthetically simulated for verification; real webcam tracking runs independently in the frozen <code>optical/</code> Python subsystem.
                    </div>
                </div>
            )}
        </div>
    );
};
