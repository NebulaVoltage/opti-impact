"""Visualization script for structural response simulations.

Generates comprehensive comparative plots:
1. Vibration (acceleration) vs time
2. True structural displacement vs time
3. Optical displacement (X & Y) vs time
4. Scenario comparison (NORMAL vs WARNING vs CRITICAL)
5. Frequency-domain preview using FFT (spectral density)

Usage:
    python simulation/visualize.py
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path when running this file directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import matplotlib.pyplot as plt
import numpy as np

from simulation.simulator import StructuralSimulator


def compute_fft(signal_data: np.ndarray, sampling_rate: float):
    """Compute single-sided FFT amplitude spectrum."""
    n = len(signal_data)
    # Detrend by removing mean
    detrended = signal_data - np.mean(signal_data)
    # Apply Hanning window to mitigate leakage
    window = np.hanning(n)
    fft_vals = np.fft.rfft(detrended * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    amplitude = np.abs(fft_vals) * 2.0 / np.sum(window)
    return freqs, amplitude


def plot_simulation_comparison(output_path: Path, seed: int = 42):
    """Generate and save 5-panel comparative visualization."""
    sim = StructuralSimulator(seed=seed)

    scenarios = ["NORMAL", "WARNING", "CRITICAL"]
    colors = {
        "NORMAL": "#2ecc71",    # Emerald green
        "WARNING": "#f39c12",   # Amber orange
        "CRITICAL": "#e74c3c",  # Crimson red
    }

    events = {s: sim.generate(s, seed=seed) for s in scenarios}

    # Setup 5-panel layout (3 rows x 2 cols, bottom row span or 5 subplots)
    fig = plt.figure(figsize=(16, 12), dpi=150)
    gs = fig.add_gridspec(3, 2, hspace=0.32, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])  # True displacement vs time
    ax2 = fig.add_subplot(gs[0, 1])  # Vibration vs time
    ax3 = fig.add_subplot(gs[1, 0])  # Optical X displacement vs time
    ax4 = fig.add_subplot(gs[1, 1])  # Optical vs True Displacement Tracking
    ax5 = fig.add_subplot(gs[2, :])  # FFT Frequency Spectrum

    # 1. True Displacement vs Time
    for s in scenarios:
        ev = events[s]
        ax1.plot(ev.time, ev.true_displacement, label=s, color=colors[s], alpha=0.85, linewidth=1.3)
    ax1.set_title("1. True Primary Structural Displacement vs. Time", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time (s)", fontsize=10)
    ax1.set_ylabel("Displacement (mm)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.axvline(x=2.0, color="gray", linestyle=":", alpha=0.7, label="Impact Time (t=2.0s)")
    ax1.legend(loc="upper right", framealpha=0.9)

    # 2. Vibration vs Time
    for s in scenarios:
        ev = events[s]
        ax2.plot(ev.time, ev.vibration, label=s, color=colors[s], alpha=0.75, linewidth=1.0)
    ax2.set_title("2. Structural Vibration (Acceleration) vs. Time", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time (s)", fontsize=10)
    ax2.set_ylabel("Vibration / Acceleration (m/s²)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.axvline(x=2.0, color="gray", linestyle=":", alpha=0.7)
    ax2.legend(loc="upper right", framealpha=0.9)

    # 3. Optical X Displacement vs Time
    for s in scenarios:
        ev = events[s]
        ax3.plot(ev.time, ev.optical_x, label=f"{s} (Optical X)", color=colors[s], alpha=0.85, linewidth=1.2)
    ax3.set_title("3. Optical Sensor Measured Displacement vs. Time", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Time (s)", fontsize=10)
    ax3.set_ylabel("Optical X (mm)", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", framealpha=0.9)

    # 4. Zoomed Tracking Detail: Optical X vs True Displacement (Transient Ringing)
    # Zoom in around impact ringing from t = 2.0s to 3.5s for CRITICAL and NORMAL
    t_mask = (events["NORMAL"].time >= 1.9) & (events["NORMAL"].time <= 3.8)
    sub_t = events["NORMAL"].time[t_mask]
    for s in ["NORMAL", "CRITICAL"]:
        ev = events[s]
        ax4.plot(
            sub_t,
            ev.true_displacement[t_mask],
            label=f"{s} True",
            color=colors[s],
            linestyle="--",
            alpha=0.7,
            linewidth=1.2,
        )
        ax4.plot(
            sub_t,
            ev.optical_x[t_mask],
            label=f"{s} Optical (noisy+quantized)",
            color=colors[s],
            alpha=0.9,
            linewidth=1.4,
        )
    ax4.set_title("4. Sensor Synchronization Detail (t = 1.9s to 3.8s)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Time (s)", fontsize=10)
    ax4.set_ylabel("Displacement (mm)", fontsize=10)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # 5. Frequency Domain Preview (Structural Vibration FFT Spectrum)
    for s in scenarios:
        ev = events[s]
        freqs, amp = compute_fft(ev.vibration, ev.config.sampling_rate)
        # Focus on dynamic structural resonance band (0.5 to 12 Hz)
        f_mask = (freqs >= 0.5) & (freqs <= 12.0)
        peak_idx = np.argmax(amp[f_mask])
        peak_freq = freqs[f_mask][peak_idx]
        ax5.plot(
            freqs[f_mask],
            amp[f_mask],
            label=f"{s} (Resonance: {peak_freq:.2f} Hz, Base $f_n$: {ev.config.effective_frequency:.1f} Hz)",
            color=colors[s],
            linewidth=1.8,
        )
        ax5.axvline(x=peak_freq, color=colors[s], linestyle=":", alpha=0.6)

    ax5.set_title("5. Frequency-Domain Vibration FFT Spectrum (Structural Natural Frequency Identification)", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Frequency (Hz)", fontsize=10)
    ax5.set_ylabel("Vibration FFT Amplitude (m/s²)", fontsize=10)
    ax5.set_xlim(0.5, 12.0)
    ax5.grid(True, linestyle="--", alpha=0.5)
    ax5.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.suptitle(
        "AI Optical Structural Impact Monitoring — Synthetic Structural Response Simulator\n"
        "(NORMAL vs WARNING vs CRITICAL Dynamics)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Comparative plot successfully saved to: {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize structural response simulations.")
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/simulation_comparison.png",
        help="Target output image file path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for simulation reproducibility.",
    )
    args = parser.parse_args()

    plot_simulation_comparison(Path(args.output), seed=args.seed)
