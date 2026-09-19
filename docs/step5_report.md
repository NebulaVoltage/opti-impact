# Step 5 Technical Report: Real-Time Multimodal Inference & Temporal Decision Engine

**Module:** `realtime`  
**Pipeline Integration:** Steps 1 $\rightarrow$ 2 $\rightarrow$ 4 $\rightarrow$ 5  
**Evaluation Status:** COMPLETE (107/107 Tests Passing)  

---

## 1. Objective
The objective of Step 5 is to transition the offline, event-based machine learning classification pipeline developed in Steps 1–4 into a modular, software-only real-time streaming architecture. 

The system continuously ingests synchronized vibration and optical tracking data, partitions the multi-channel time-series into overlapping streaming windows, extracts the 32 Step 2 physical engineering features, performs instantaneous model inference via the frozen Step 4 bundle, and applies temporal hysteresis to stabilize structural condition classifications (`NORMAL`, `WARNING`, `CRITICAL`).

---

## 2. Existing Architecture Inspection
Prior to implementation, existing project artifacts and interfaces across Steps 1–4 were inspected:
- **Step 1 (`simulation/simulator.py`):** `StructuralSimulator.generate(scenario, seed)` generates `SimulationEvent` objects containing synchronized NumPy arrays (`time`, `true_displacement`, `vibration`, `optical_x`, `optical_y`, `optical_rotation`, `scenario`, `config`).
- **Step 2 (`signal_processing/feature_pipeline.py`):** `FeaturePipeline.extract_from_arrays(vibration, optical_x, optical_y, optical_rotation, sampling_rate)` returns `StructuralFeatures` containing 32 physical metrics (11 vibration time-domain, 6 frequency-domain, 12 optical displacement/kinematics/rotation, 3 cross-modal correlations).
- **Step 3 (`dataset/schema.py`):** Authoritative schema defining `FEATURE_COLUMNS` (32 columns), `METADATA_COLUMNS`, and `TARGET_COLUMN = 'scenario'`.
- **Step 4 (`models/step4/`):** Serialized `step4_v1_model.joblib` (Standardized Logistic Regression), `step4_v1_preprocessor.joblib` (`StandardScaler`), and provenance metadata.

---

## 3. Simulator Integration
To simulate a continuous multi-scenario monitoring deployment, `SyntheticSensorStream` orchestrates sequential runs of the Step 1 `StructuralSimulator` without modifying underlying physics. Each segment (e.g. 20s of `NORMAL`, 20s of `WARNING`, etc.) is synthesized as contiguous structural excitation intervals with continuous, strictly monotonically increasing timestamps ($dt = 0.01\text{ s}$ at $100\text{ Hz}$).

---

## 4. Optical Data Pathway
Optical displacement is a core sensing modality, not a decorative visualization. The optical data pathway follows an unbroken flow:
```
Step 1 Simulator (optical_x, optical_y, optical_rotation)
  ↓
SensorChunk / SensorWindow (synchronized with vibration)
  ↓
Step 2 FeaturePipeline (extracts 12 optical kinematic & deflection metrics)
  ↓
Step 4 Preprocessor (StandardScaler transformation)
  ↓
ML Model & Temporal Engine
```
No secondary optical simulator was created; optical features are never derived from vibration; and the simulated optical channel provides contactless kinematic measurements.

---

## 5. Sensor-Stream Abstraction
A decoupled abstract interface `SensorStream` defines:
- `read_chunk(num_samples: int) -> Optional[SensorChunk]`
- `has_more() -> bool`
- `reset() -> None`

This ensures the downstream streaming window buffer, feature extraction pipeline, and ML classifier have zero knowledge of whether data originates from `SyntheticSensorStream` or future physical sensor drivers (`ArduinoSensorStream`, `WebcamOpticalStream`).

---

## 6. Streaming Window Design
- **Sampling Frequency:** $100\text{ Hz}$
- **Window Duration:** $5.0\text{ s}$ ($500\text{ samples}$)
- **Inference Step / Hop:** $1.0\text{ s}$ ($100\text{ samples}$)
- **Window Overlap:** $80\%$ ($4.0\text{ s}$ overlap between consecutive inference passes)
- **Transition Tagging:** Windows that straddle scenario change boundaries (e.g. $t = 18\text{ s} - 23\text{ s}$) are flagged with `is_transition_window = True` and cataloged with class proportions.

---

## 7. Step 2 Feature-Pipeline Reuse
Zero feature extraction logic is duplicated in Step 5. `RealtimeInferenceEngine` directly invokes the existing Step 2 `FeaturePipeline.extract_from_arrays()`. This guarantees numerical consistency with training data and eliminates discrepancies in RMS calculation ($\text{RMS} = \sqrt{\text{mean}(x^2)}$), Savitzky-Golay kinematics smoothing, FFT spectral centroid, or cross-correlation lag.

---

## 8. Step 4 Model Integration
The finalized Step 4 artifacts are loaded dynamically:
- Model: `models/step4/step4_v1_model.joblib`
- Preprocessor: `models/step4/step4_v1_preprocessor.joblib`
- Metadata: `models/step4/step4_v1_metadata.json`

The engine validates that the preprocessor expects 32 features strictly in `FEATURE_COLUMNS` order and that model classes match `['NORMAL', 'WARNING', 'CRITICAL']`.

---

## 9. Instantaneous Inference
For every 5-second window, `RealtimeInferenceEngine.predict_window()` executes:
1. Feature extraction ($32$ numerical metrics).
2. StandardScaler normalization using the training-fitted preprocessor.
3. Multi-class model inference returning predicted class and calibrated class posterior probabilities:
```json
{
  "instant_prediction": "WARNING",
  "probabilities": {
    "NORMAL": 0.001,
    "WARNING": 0.998,
    "CRITICAL": 0.001
  }
}
```

---

## 10. Temporal State Machine
The system maintains a stable structural state:
$$\mathcal{S} \in \{\text{NORMAL}, \text{WARNING}, \text{CRITICAL}\}$$
Instantaneous predictions update the internal state machine. Transient anomalies or noisy individual windows cannot cause state flapping.

---

## 11. Hysteresis and Persistence
State transitions require confirmed consecutive agreement:
- $\text{NORMAL} \rightarrow \text{WARNING}$: Requires $3$ consecutive `WARNING` window predictions.
- $\text{WARNING} \rightarrow \text{CRITICAL}$: Requires $3$ consecutive `CRITICAL` window predictions.
- $\text{CRITICAL} \rightarrow \text{NORMAL}$ (Recovery): Requires $3$ consecutive `NORMAL` window predictions.

If an ongoing sequence of candidate predictions is interrupted by the current stable state, the candidate counter is reset to 0. Alternating predictions (e.g. `WARNING`, `NORMAL`, `WARNING`, `NORMAL`) leave the stable state undisturbed.

---

## 12. State Transition Examples
During the 100-second demonstration sequence ($0–20\text{s}$ NORMAL, $20–40\text{s}$ WARNING, $40–60\text{s}$ CRITICAL, $60–80\text{s}$ WARNING, $80–100\text{s}$ NORMAL):
1. **$t = 22.0\text{s}$:** Windows at $20\text{s}, 21\text{s}, 22\text{s}$ predict `WARNING` $\rightarrow$ Transition `NORMAL` $\rightarrow$ `WARNING`.
2. **$t = 42.0\text{s}$:** Windows at $40\text{s}, 41\text{s}, 42\text{s}$ predict `CRITICAL` $\rightarrow$ Transition `WARNING` $\rightarrow$ `CRITICAL`.
3. **$t = 62.0\text{s}$:** Windows at $60\text{s}, 61\text{s}, 62\text{s}$ predict `WARNING` $\rightarrow$ Transition `CRITICAL` $\rightarrow$ `WARNING`.
4. **$t = 82.0\text{s}$:** Windows at $80\text{s}, 81\text{s}, 82\text{s}$ predict `NORMAL` $\rightarrow$ Transition `WARNING` $\rightarrow$ `NORMAL`.
5. **$t = 88.0\text{s}, 92.0\text{s}$:** Isolated candidate predictions (`WARNING`) appear during tail relaxation but do not reach the 3-window threshold $\rightarrow$ Stable state remains safely `NORMAL`.

---

## 13. Telemetry
Every window records structured telemetry to `outputs/step5/inference_log.csv`:
- `timestamp`, `window_id`, `instant_prediction`, `stable_state`
- `probability_normal`, `probability_warning`, `probability_critical`
- Physical features: `vibration_rms`, `vibration_peak`, `dominant_frequency`, `optical_displacement`, `optical_velocity`, `optical_acceleration`, `optical_dominant_frequency`, `vibration_optical_correlation`, `cross_correlation_max`, `cross_correlation_lag`
- Performance: component latencies (`feature_extraction_latency_ms`, `preprocessing_latency_ms`, `model_inference_latency_ms`, `temporal_decision_latency_ms`, `total_latency_ms`).

---

## 14. Latency Measurements
Profiling results measured across 96 streaming inference windows on the host machine:

| Metric | Measured Duration |
| :--- | :---: |
| **Mean Window Latency** | $11.40\text{ ms}$ |
| **Median Window Latency** | $11.08\text{ ms}$ |
| **95th Percentile Latency ($P_{95}$)** | $15.41\text{ ms}$ |
| **Maximum Window Latency** | $30.24\text{ ms}$ |
| **Available Budget per Window Hop** | $1,000.00\text{ ms}$ |
| **CPU Utilization Ratio** | $\approx 1.14\%$ |

The entire feature extraction, scaling, inference, and temporal engine pipeline executes in $\sim 11\text{ ms}$, operating at roughly $90\times$ faster than real-time requirements.

---

## 15. Instantaneous Evaluation
Evaluated against ground truth on stationary, clean windows (excluding mixed boundary-straddling windows):
- **Accuracy:** $0.9625$
- **Macro F1:** $0.9687$
- **NORMAL Recall:** $0.9062$
- **WARNING Recall:** $1.0000$
- **CRITICAL Recall:** $1.0000$

---

## 16. Temporal Evaluation
Evaluated on the stabilized temporal state output across clean windows:
- **Accuracy:** $0.9625$
- **Macro F1:** $0.9592$
- **NORMAL Recall:** $0.9688$ (improved stability and recovery over raw instant predictions)
- **WARNING Recall:** $0.9375$
- **CRITICAL Recall:** $1.0000$ (zero false negatives for critical structural response)

---

## 17. Visualizations
Generated and archived in `evaluation/plots/realtime/`:
1. **`state_timeline.png`:** Compares Ground Truth, Instantaneous Model Predictions, and the Stabilized Temporal State across time, highlighting mixed transition regions.
2. **`probability_timeline.png`:** Plots continuous multi-class posterior probability curves $P(\text{NORMAL}), P(\text{WARNING}), P(\text{CRITICAL})$.
3. **`feature_timeline.png`:** Displays synchronized multi-panel trajectories for Vibration RMS, Dominant Frequency, Optical Displacement, and Cross-Modal Correlation.

---

## 18. Test Results
The comprehensive test suite verifies all layers:
- **Step 5 Real-Time Tests:** 26 tests (Model loading, sensor streams, windowing, inference, temporal state machine, hysteresis, serialization).
- **Step 1–4 Tests:** 81 tests.
- **Total Passing Tests:** **107 / 107** ($100\%$ pass rate, $0$ failures).

---

## 19. Hardware-Readiness Architecture
The decoupled architecture enables dropping in physical hardware in subsequent phases without changing downstream code:
- To add accelerometers: implement `ArduinoSensorStream(SensorStream)` reading serial packets into `SensorChunk`.
- To add camera tracking: implement optical displacement extraction via OpenCV ArUco tracking returning `optical_x`, `optical_y`, `optical_rotation`.
- The window manager, feature pipeline, preprocessor, model, temporal engine, and telemetry remain 100% untouched.

---

## 20. Limitations
1. **Synthetic Dynamics:** Responses are generated using physics-inspired synthetic modal equations without non-linear damping or multi-span boundary interactions.
2. **Optical Noise Model:** Simulated optical displacements include Gaussian noise and baseline drift, which does not capture camera occlusion, low-light blur, or lens distortion.
3. **Persistence Parameters:** The 3-window persistence threshold is a software prototype parameter for demonstration, not a civil engineering standard.

---

## 21. Synthetic-Data Disclaimer
> **MANDATORY NOTICE:**
> The real-time inference and temporal evaluation were performed using physics-inspired synthetic structural-response data. These results demonstrate software-pipeline feasibility only and do not establish real-world bridge damage-detection accuracy, structural safety thresholds, or deployment readiness. The optical measurements are still simulated; real camera/ArUco/optical-flow measurements have not yet been integrated.

---

## 22. Recommended Next Stage
- **Step 6:** Live streaming API (FastAPI backend with WebSocket streaming telemetry) and interactive real-time dashboard (React / Vite) displaying synchronized waveforms, probability gauges, and temporal status indicators.
