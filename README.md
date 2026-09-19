# Optical Structural Impact Monitoring System (Opti-Impact)

**Multimodal AI-Powered Structural Response & Impact Monitoring Architecture**  
*Combining Physical Optical Flow Kinematics, Mechanical Vibration Response, and Real-Time Machine Learning Decision Systems.*

---

## 📌 Project Overview

**Opti-Impact** is an end-to-end structural condition monitoring framework designed to differentiate ordinary environmental vibrations from abnormal structural response behaviors indicating stiffness degradation or structural impact.

The system combines:
1. **Physical Optical Motion Tracking**: Marker-free Lucas–Kanade pyramidal optical flow on irregular visual surface features (e.g. electrical tape on structural members) using standard USB video cameras.
2. **Synchronized Multimodal Kinematics**: Structural displacement, velocity vectors, Savitzky–Golay smoothed acceleration, and dominant frequency estimation.
3. **Engineering Feature Extraction**: Time-domain (RMS, peak-to-peak, kurtosis, crest factor), frequency-domain (spectral centroids, energy bands, dominant frequency), optical dynamics, and normal baseline z-score deviation.
4. **Machine Learning Classification**: Multi-class structural state classifier (NORMAL, WARNING, CRITICAL) trained with strict anti-leakage stratification.
5. **Real-Time Temporal Engine**: Rolling sliding window inference (5.0s window, 1.0s hop) governed by temporal persistence counters and state hysteresis to reject single-window transient false positives.

---

## 🏗️ Repository Architecture

```
opti-impact/
├── simulation/            # Step 1: Physics-inspired structural response simulator
├── signal_processing/     # Step 2: Signal filtering & multimodal feature extraction
├── dataset/               # Step 3: Controlled variation dataset & normal baseline
├── ml/                    # Step 4: Model benchmark, ablation, and serialization
├── models/                # Production model & preprocessor artifacts
├── realtime/              # Step 5: Streaming inference & temporal decision engine
├── optical/               # Step 6A: Physical USB webcam optical flow sensing & kinematics
├── evaluation/            # Evaluation scripts, ablation benchmarks, and plots
├── docs/                  # Detailed engineering reports (Step 4, Step 5, Step 6A)
└── outputs/               # Realtime inference logs and optical telemetry sessions
```

---

## 🚀 Quick Start

### 1. Installation
```powershell
git clone https://github.com/NebulaVoltage/opti-impact.git
cd opti-impact
pip install -r requirements.txt
```

### 2. Run All Automated Unit & Integration Tests (127 Tests)
```powershell
python -m pytest -v
```

### 3. Run Physical Optical Sensing (External Webcam)
```powershell
# List available video capture devices
python -m optical.demo --list-cameras

# Run live optical tracking with HUD overlay & strip chart
python -m optical.demo --camera 1 --record

# Run headless 10-second optical recording
python -m optical.demo --camera 1 --duration 10 --record --headless
```

### 4. Run Real-Time Multimodal Decision Simulation
```powershell
python -m realtime.demo
```

---

## 🔬 Key Subsystems

### Marker-Free Optical Motion Tracking (`optical/`)
- Uses Shi–Tomasi corner detection within a structural Region of Interest (ROI).
- Tracks features across frames via pyramidal Lucas–Kanade optical flow (`cv2.calcOpticalFlowPyrLK`).
- Aggregates frame displacements using spatial median filtering for outlier rejection.
- Smooths numerical velocity derivatives with Savitzky–Golay filtering ($W=7, p=2$).
- Estimates dominant structural frequency via directional FFT detrended around the primary oscillation coordinate.
- Reports raw units in pixels, px/s, px/s² until metric planar calibration is engaged (`METRIC CALIBRATION: NOT ACTIVE`).

### Real-Time Inference & Temporal Hysteresis (`realtime/`)
- Ingests synchronized vibration and optical streams into a 5-second FIFO window.
- Evaluates instant ML probability distributions with calibrated Logistic Regression.
- Enforces strict transition hysteresis:
  - $\text{NORMAL} \to \text{WARNING}$: Requires 3 consecutive warning predictions.
  - $\text{NORMAL} / \text{WARNING} \to \text{CRITICAL}$: Requires 3 consecutive critical predictions.
  - Recovery to $\text{NORMAL}$: Requires 5 consecutive normal predictions.

---

## 📊 Verification & Benchmarks

- **Automated Tests**: 127/127 passing (0 failures).
- **Physical Webcam Acquisition**: 1280x720 HD @ 28.4–30.1 FPS sustained.
- **Optical Processing Latency**: Mean 27.16 ms per frame (fits within 33.3 ms budget).
- **Stationary Noise Floor**: $< 0.02\text{ px}$ peak-to-peak.
- **ML Final Test Accuracy**: 99.44% across stratified synthetic evaluations.

---

## ⚖️ Engineering Disclosures & Limitations

1. **Proof-of-Concept**: Current physical optical demonstrations use a laboratory cardboard specimen. This system is a proof-of-concept and does **not** constitute real-world bridge safety certification.
2. **Optical Nyquist Limit**: At 30 FPS, the maximum resolvable vibration frequency is 15.0 Hz.
3. **Integration Drift**: Long-duration unreferenced optical flow requires periodic zeroing to prevent sub-pixel integration drift.

---

## 👤 Author
**NebulaVoltage** ([GitHub: NebulaVoltage](https://github.com/NebulaVoltage))
