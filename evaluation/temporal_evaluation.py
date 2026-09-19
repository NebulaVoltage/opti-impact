"""Temporal and real-time evaluation module.

Evaluates:
  1. Instantaneous single-window ML predictions vs. ground truth.
  2. Temporal stabilized system states vs. ground truth.
  3. Generates 3 time-series diagnostics in evaluation/plots/realtime/:
     - state_timeline.png
     - probability_timeline.png
     - feature_timeline.png

NOTE: Transition windows that straddle scenario switches are processed in real-time
but excluded from clean stationary class-level metrics.
"""

from pathlib import Path
from typing import Any, Dict, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.metrics import compute_classification_metrics
from realtime.config import RealtimePipelineConfig
from realtime.data_stream import SyntheticSensorStream
from realtime.runtime import RealtimeMonitoringRuntime

STATE_COLOR_MAP = {
    "NORMAL": "#2b5c8f",
    "WARNING": "#e9b872",
    "CRITICAL": "#e84a5f",
}

STATE_NUMERIC_MAP = {
    "NORMAL": 0,
    "WARNING": 1,
    "CRITICAL": 2,
}


def run_or_load_inference_log(
    csv_path: Path = Path("outputs/step5/inference_log.csv"),
) -> pd.DataFrame:
    """Load existing inference log or execute a standard 100-second session."""
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if len(df) > 0:
            return df

    # Run session
    schedule = [
        ("NORMAL", 20.0),
        ("WARNING", 20.0),
        ("CRITICAL", 20.0),
        ("WARNING", 20.0),
        ("NORMAL", 20.0),
    ]
    cfg = RealtimePipelineConfig()
    stream = SyntheticSensorStream(scenario_schedule=schedule, sampling_rate=cfg.stream.sampling_rate, base_seed=42)
    runtime = RealtimeMonitoringRuntime(config=cfg, stream=stream)
    runtime.run_all()
    return pd.read_csv(csv_path)


def evaluate_instantaneous_and_temporal(
    df_log: pd.DataFrame,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Compute classification metrics for instantaneous predictions and temporal stable state.

    Args:
        df_log: Telemetry DataFrame containing instant_prediction, stable_state,
                ground_truth_scenario, and is_transition_window.

    Returns:
        Tuple of (clean_instant_metrics, clean_temporal_metrics, all_windows_instant_metrics).
    """
    # Exclude transition-crossing boundary windows for clean stationary evaluation
    df_clean = df_log[df_log["is_transition_window"] == False].copy()

    y_true_clean = df_clean["ground_truth_scenario"].values
    y_pred_instant_clean = df_clean["instant_prediction"].values
    y_pred_temporal_clean = df_clean["stable_state"].values

    instant_metrics_clean = compute_classification_metrics(y_true_clean, y_pred_instant_clean)
    temporal_metrics_clean = compute_classification_metrics(y_true_clean, y_pred_temporal_clean)

    # Also compute on all windows including transitions for transparent reporting
    instant_metrics_all = compute_classification_metrics(
        df_log["ground_truth_scenario"].values,
        df_log["instant_prediction"].values,
    )

    return instant_metrics_clean, temporal_metrics_clean, instant_metrics_all


def plot_state_timeline(
    df_log: pd.DataFrame,
    output_path: Path = Path("evaluation/plots/realtime/state_timeline.png"),
) -> Path:
    """Plot ground-truth scenario vs instantaneous predictions vs stable temporal state."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t = df_log["timestamp"].values
    gt_num = df_log["ground_truth_scenario"].map(STATE_NUMERIC_MAP).values
    inst_num = df_log["instant_prediction"].map(STATE_NUMERIC_MAP).values
    stab_num = df_log["stable_state"].map(STATE_NUMERIC_MAP).values

    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

    # Shaded ground truth background
    ax.step(t, gt_num, label="Ground Truth Condition", where="post", color="#888888", linewidth=2.5, linestyle="--")
    ax.scatter(t, inst_num, label="Instant ML Prediction", color="#3b9ab2", alpha=0.6, s=25, zorder=3)
    ax.step(t, stab_num, label="Stable Temporal State (Hysteresis)", where="post", color="#e84a5f", linewidth=3.0, zorder=4)

    # Highlight transition boundary windows
    trans_t = df_log[df_log["is_transition_window"] == True]["timestamp"].values
    if len(trans_t) > 0:
        for tt in trans_t:
            ax.axvspan(tt - 0.5, tt + 0.5, color="#ffecb3", alpha=0.3)
        ax.plot([], [], color="#ffecb3", alpha=0.6, linewidth=6, label="Transition Windows (Mixed)")

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["NORMAL", "WARNING", "CRITICAL"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Structural Condition", fontsize=12, fontweight="bold")
    ax.set_title("Real-Time Structural State Timeline: Instantaneous vs Temporal Hysteresis", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_probability_timeline(
    df_log: pd.DataFrame,
    output_path: Path = Path("evaluation/plots/realtime/probability_timeline.png"),
) -> Path:
    """Plot continuous model-estimated class probabilities over time."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t = df_log["timestamp"].values
    p_norm = df_log["probability_normal"].values
    p_warn = df_log["probability_warning"].values
    p_crit = df_log["probability_critical"].values

    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

    ax.plot(t, p_norm, label="P(NORMAL)", color=STATE_COLOR_MAP["NORMAL"], linewidth=2.2)
    ax.plot(t, p_warn, label="P(WARNING)", color=STATE_COLOR_MAP["WARNING"], linewidth=2.2)
    ax.plot(t, p_crit, label="P(CRITICAL)", color=STATE_COLOR_MAP["CRITICAL"], linewidth=2.2)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Estimated Class Probability", fontsize=12, fontweight="bold")
    ax.set_title("Multimodal Classifier Posterior Probability Trajectories", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="center right", frameon=True, fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_feature_timeline(
    df_log: pd.DataFrame,
    output_path: Path = Path("evaluation/plots/realtime/feature_timeline.png"),
) -> Path:
    """Plot multi-panel physical feature dynamics over time."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t = df_log["timestamp"].values

    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True, dpi=300)

    # 1. Vibration RMS
    axs[0].plot(t, df_log["vibration_rms"].values, color="#2b5c8f", linewidth=2.0)
    axs[0].set_ylabel("Vib RMS (m/s²)", fontsize=10, fontweight="bold")
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].set_title("Physical Engineering Feature Dynamics Over Time", fontsize=13, fontweight="bold", pad=12)

    # 2. Dominant Vibration Frequency
    axs[1].plot(t, df_log["dominant_frequency"].values, color="#3b9ab2", linewidth=2.0)
    axs[1].set_ylabel("Vib Freq (Hz)", fontsize=10, fontweight="bold")
    axs[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Optical Max Displacement
    axs[2].plot(t, df_log["optical_displacement"].values, color="#e84a5f", linewidth=2.0)
    axs[2].set_ylabel("Opt Disp (mm)", fontsize=10, fontweight="bold")
    axs[2].grid(True, linestyle="--", alpha=0.5)

    # 4. Vibration-Optical Correlation
    axs[3].plot(t, df_log["vibration_optical_correlation"].values, color="#6a4c93", linewidth=2.0)
    axs[3].set_ylabel("Opt/Vib Corr", fontsize=10, fontweight="bold")
    axs[3].set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    axs[3].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def run_temporal_evaluation() -> None:
    """Execute full real-time and temporal evaluation."""
    print("\n" + "=" * 64)
    print("RUNNING TEMPORAL EVALUATION & TIMELINE PLOTTING")
    print("=" * 64)

    df_log = run_or_load_inference_log()
    print(f"Loaded {len(df_log)} inference windows from log.")

    m_inst_clean, m_temp_clean, m_inst_all = evaluate_instantaneous_and_temporal(df_log)

    print("\n--- INSTANTANEOUS ML PREDICTION PERFORMANCE (Clean Windows) ---")
    print(f"  Accuracy:         {m_inst_clean['accuracy']:.4f}")
    print(f"  Macro F1:         {m_inst_clean['macro_f1']:.4f}")
    print(f"  NORMAL Recall:    {m_inst_clean['normal_recall']:.4f}")
    print(f"  WARNING Recall:   {m_inst_clean['warning_recall']:.4f}")
    print(f"  CRITICAL Recall:  {m_inst_clean['critical_recall']:.4f}")

    print("\n--- STABLE TEMPORAL STATE PERFORMANCE (Clean Windows) ---")
    print(f"  Accuracy:         {m_temp_clean['accuracy']:.4f}")
    print(f"  Macro F1:         {m_temp_clean['macro_f1']:.4f}")
    print(f"  NORMAL Recall:    {m_temp_clean['normal_recall']:.4f}")
    print(f"  WARNING Recall:   {m_temp_clean['warning_recall']:.4f}")
    print(f"  CRITICAL Recall:  {m_temp_clean['critical_recall']:.4f}")

    p_state = plot_state_timeline(df_log)
    p_prob = plot_probability_timeline(df_log)
    p_feat = plot_feature_timeline(df_log)

    print(f"\nGenerated Plots:")
    print(f"  State Timeline:       {p_state}")
    print(f"  Probability Timeline: {p_prob}")
    print(f"  Feature Timeline:     {p_feat}")
    print("=" * 64)


if __name__ == "__main__":
    run_temporal_evaluation()
