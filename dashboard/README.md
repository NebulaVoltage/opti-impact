# Step 6A.2 — Vision-Only Structural Monitoring Dashboard

## 1. Purpose
The **Vision-Only Structural Monitoring Dashboard** is a standalone, browser-based telemetry monitoring application that visualizes structural kinematics and optical flow tracking characteristics in **VISION-ONLY SIMULATION MODE**.

It provides real-time visualization of:
- Structural displacement ($x$, $y$, magnitude)
- Multi-mode kinematics switcher: Displacement ($\text{mm}$), Velocity ($\text{mm/s}$), Acceleration ($\text{m/s}^2$)
- Simulated Confidence & Optical Quality Panel (Confidence meter $0-100\%$, inliers, rejection breakdown, MAD dispersion)
- Dominant structural modal frequency evolution with scenario reference bands
- 2D structural box-beam specimen on fixed and roller support piers with camera framing HUD ($1280 \times 720$)
- Shi–Tomasi corner inliers, FB/MAD rejected points, and dynamic Lucas–Kanade velocity flow vectors
- Educational optics panel explaining camera framing, corner tracking, bidirectional LK flow, MAD filtering, and FFT modal resonance
- Scenario controls (`NORMAL`, `WARNING`, `CRITICAL`), Auto-Scenario sequencing mode, and timestamped event logs

> **SAFETY & SCIENTIFIC DISCLAIMER**:
> This dashboard operates in **SIMULATION MODE ONLY**. All data, waveforms, frequencies, and scenarios are synthetic, physics-inspired engineering simulations. They are **NOT** real bridge measurements, **NOT** safety-certified thresholds, and **NOT** evidence of structural damage detection or safety certification. The dashboard labels reflect *Simulated structural response states*, not real-world structural health assessments.

---

## 2. Architecture & Tech Stack

```
dashboard/
├── index.html                   # HTML entry point with dark technical theme
├── package.json                 # React 19, Lucide React, TypeScript, Vite, Vitest
├── tsconfig.json                # Strict TypeScript configuration
├── vite.config.ts               # Vite bundler configuration
└── src/
    ├── types/
    │   └── telemetry.ts         # Strict TypeScript schemas for telemetry and events
    ├── simulation/
    │   ├── scenarios.ts         # Parameter profiles for NORMAL, WARNING, CRITICAL
    │   ├── visionSimulator.ts   # Deterministic 30 FPS physics simulator with auto-cycling
    │   └── visionSimulator.test.ts # Vitest unit test suite (16 comprehensive tests)
    ├── components/
    │   ├── StructuralViewer.tsx # 2D Canvas rendering specimen oscillation, piers, and flow vectors
    │   ├── TelemetryCards.tsx   # Real-time engineering metric display cards
    │   ├── OpticalQualityPanel.tsx # Step 6A.1-P precision audit metrics & confidence meter
    │   ├── DisplacementChart.tsx# Multi-mode kinematics scrolling waveform (disp / vel / accel)
    │   ├── FrequencyChart.tsx   # Dominant frequency gauge and modal trend line with threshold bands
    │   ├── ScenarioControls.tsx # NORMAL/WARNING/CRITICAL, Auto Sequence & Playback controls
    │   ├── OpticsEducationalPanel.tsx # Step-by-step explainer of optical vision sensing
    │   ├── EventTimeline.tsx    # Scrollable timestamped event log
    │   └── OpticalSensorIndicator.tsx # Virtual sensor specs & active mode badge
    ├── App.tsx                  # Root dashboard layout and telemetry state wiring
    ├── main.tsx                 # React DOM mount point
    └── index.css                # Dark technical engineering UI theme
```

---

## 3. Telemetry Schema

The simulator emits a `VisionTelemetry` packet at 30 FPS ($dt \approx 0.033\text{ s}$):

| Field | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `timestamp` | `number` | s | Elapsed simulation time |
| `displacementX` | `number` | mm | Horizontal displacement relative to baseline |
| `displacementY` | `number` | mm | Vertical displacement relative to baseline |
| `displacementMagnitude` | `number` | mm | Total Euclidean displacement: $\sqrt{x^2 + y^2}$ |
| `velocityX` | `number` | mm/s | Horizontal velocity component |
| `velocityY` | `number` | mm/s | Vertical velocity component |
| `velocityMagnitude` | `number` | mm/s | Total velocity magnitude: $\sqrt{v_x^2 + v_y^2}$ |
| `accelerationX` | `number` | mm/s² | Horizontal acceleration component |
| `accelerationY` | `number` | mm/s² | Vertical acceleration component |
| `accelerationMagnitude` | `number` | mm/s² | Total acceleration magnitude: $\sqrt{a_x^2 + a_y^2}$ |
| `dominantFrequency` | `number` | Hz | Dominant structural response frequency |
| `featureCount` | `number` | count | Count of successfully tracked optical features |
| `trackingQuality` | `TrackingQuality` | categorical | Quality state: `GOOD` ( $\ge 15$ ), `DEGRADED` ( $5-14$ ), `LOST` ( $< 5$ ) |
| `measurementValid` | `boolean` | flag | `true` if displacement is within physically plausible bounds |
| `measurementStatus` | `MeasurementStatus` | categorical | `VALID`, `EXCESSIVE_STEP_DISPLACEMENT`, `EXCEEDS_FRAME_BOUNDS`, `TRACKING_LOST` |
| `scenario` | `ScenarioType` | categorical | Active simulation scenario (`NORMAL`, `WARNING`, `CRITICAL`) |
| `features` | `FeaturePoint[]` | array | 50 tracked corner coordinates and flow vectors for canvas visualization |

---

## 4. Simulation Kinematics & Equations

The simulation engine models damped harmonic structural oscillations driven by scenario parameters:

1. **Displacement**:
   $$x(t) = A_x \cdot \sin(2\pi f t) + 0.15 A_x \cdot \sin(4\pi f t + \phi) + \eta_x(t)$$
   $$y(t) = 0.35 \cdot x(t) + \eta_y(t)$$
   $$R(t) = \sqrt{x(t)^2 + y(t)^2}$$
   where $A_x$ is target displacement, $f$ is dominant modal frequency, and $\eta(t)$ is zero-mean measurement noise.

2. **Velocity**:
   $$v_x(t) = \frac{x(t) - x(t - \Delta t)}{\Delta t}, \quad v_y(t) = \frac{y(t) - y(t - \Delta t)}{\Delta t}$$
   $$V(t) = \sqrt{v_x(t)^2 + v_y(t)^2}$$

3. **Acceleration**:
   $$a_x(t) = \frac{v_x(t) - v_x(t - \Delta t)}{\Delta t}, \quad a_y(t) = \frac{v_y(t) - v_y(t - \Delta t)}{\Delta t}$$
   $$A(t) = \sqrt{a_x(t)^2 + a_y(t)^2}$$

4. **Parameter Transitions**:
   When switching scenarios, target amplitudes and modal frequencies transition smoothly using exponential decay interpolation:
   $$P(t + \Delta t) = P(t) + \alpha \cdot (P_{\text{target}} - P(t)), \quad \alpha = 1 - e^{-k \Delta t}$$

---

## 5. Simulation Scenarios

| Scenario | Target Displacement | Target Frequency | Tracking Quality | Measurement Status | Typical Feature Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NORMAL** | $\sim 0.65\text{ mm}$ ($0.2-1.2\text{ mm}$) | $5.15\text{ Hz}$ ($4.6-5.4\text{ Hz}$) | `GOOD` | `VALID` | $20-25$ |
| **WARNING** | $\sim 2.85\text{ mm}$ ($1.5-4.2\text{ mm}$) | $4.30\text{ Hz}$ ($3.9-4.8\text{ Hz}$) | `GOOD` / `DEGRADED` | `VALID` | $14-22$ |
| **CRITICAL**| $\sim 11.20\text{ mm}$ ($6.0-18.0\text{ mm}$)| $3.50\text{ Hz}$ ($3.0-4.1\text{ Hz}$) | `DEGRADED` / `LOST` | `VALID` / `EXCESSIVE_STEP_DISPLACEMENT` | $4-15$ |

---

## 6. Running the Dashboard

### Prerequisites
- Node.js (v18+ recommended, tested with Node v24.14.1)
- npm (v9+)

### Installation & Development
```bash
# Navigate to dashboard directory
cd dashboard

# Install dependencies
npm install

# Start development server
npm run dev
# Dashboard opens at http://localhost:5173/
```

### Running Tests
```bash
# Execute Vitest unit test suite (16 tests)
npm test
```

### Production Build
```bash
# Compile TypeScript and bundle with Vite
npm run build
```

---

## 7. Future Integration with Live Optical Pipeline

The dashboard is designed with clean decoupling between UI components and telemetry generation:

```
[ Real Webcam + Lucas-Kanade ]  (optical/demo.py)
              │
              ▼ (JSON Telemetry over WebSocket / SSE)
     [ Telemetry Adapter ]
              │
              ▼
   [ React Dashboard State ]
```

To switch from `VisionSimulator` to live webcam telemetry in future iterations:
1. Replace `sim.subscribeTelemetry(...)` in `App.tsx` with a WebSocket or Server-Sent Events subscriber listening to the Python backend endpoint.
2. The payload fields (`displacementX`, `dominantFrequency`, `featureCount`, `measurementValid`, etc.) adhere to the exact same contract defined in `src/types/telemetry.ts`.
3. Zero modifications to `StructuralViewer`, `TelemetryCards`, `DisplacementChart`, or `FrequencyChart` will be required.

---

## 8. Limitations & Scope Constraints

1. **Simulation Mode Only**: The current dashboard visualizes synthetic structural kinematics and corner flow. It does not ingest physical video frames directly in the browser.
2. **Frozen Subsystems**: The existing Python optical pipeline (`optical/*`), signal processing (`signal_processing/*`), and ML models (`ml/*`) remain completely frozen and untouched.
3. **No Sensor Fusion Yet**: Step 6B (ESP32/vibration accelerometer fusion) has not been implemented.
4. **No Structural Safety Claim**: This application does not certify structural safety or detect damage in real civil infrastructure.
