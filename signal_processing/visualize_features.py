"""Diagnostic and comparative visualization script for signal processing.

Generates:
1. plots/signal_processing/diagnostic_features.png:
   6-panel diagnostic figure containing:
   - Raw vibration vs time
   - Filtered / preprocessed vibration vs time
   - FFT magnitude spectrum
   - Optical displacement magnitude and components
   - Optical velocity vs time
   - Optical acceleration vs time
2. plots/signal_processing/spectral_comparison.png:
   Comparative frequency-domain spectra for NORMAL, WARNING, and CRITICAL.

Usage:
    python signal_processing/visualize_features.py
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from simulation.simulator import StructuralSimulator
from signal_processing.filters import preprocess_signal
from signal_processing.frequency_features import compute_spectrum, get_dominant_frequency
from signal_processing.optical_features import (
    calculate_relative_displacement,
    compute_optical_kinematics,
)


def plot_diagnostic_features(event, output_path: Path):
    """Generate 6-panel diagnostic visualization for a single event."""
    fs = float(event.config.sampling_rate)
    time = event.time

    # 1. Vibration filtering (raw vs processed with DC removal and mild bandpass)
    raw_vib, proc_vib = preprocess_signal(
        event.vibration,
        sampling_rate=fs,
        remove_mean=True,
        filter_type="bandpass",
        lowcut=0.5,
        highcut=25.0,
        order=4,
    )

    # 2. FFT spectrum
    freqs, amp = compute_spectrum(proc_vib, sampling_rate=fs, window="hann")
    dom_freq, dom_amp = get_dominant_frequency(proc_vib, sampling_rate=fs, min_freq=0.5)

    # 3. Optical displacement & kinematics (using Savitzky-Golay smoothing to avoid differentiation noise)
    x_disp, y_disp, r_disp = calculate_relative_displacement(event.optical_x, event.optical_y)
    vel_mag, accel_mag = compute_optical_kinematics(
        x_disp, y_disp, sampling_rate=fs, smoothing_method="savgol", smoothing_window=11
    )

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), dpi=150)
    fig.subplots_adjust(hspace=0.35, wspace=0.22)

    # 1. Raw Vibration
    axes[0, 0].plot(time, raw_vib, color="#e67e22", alpha=0.85, linewidth=1.0)
    axes[0, 0].set_title("1. Raw Structural Vibration Signal", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Time (s)", fontsize=10)
    axes[0, 0].set_ylabel("Vibration (m/s²)", fontsize=10)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # 2. Filtered Vibration
    axes[0, 1].plot(time, proc_vib, color="#2980b9", alpha=0.85, linewidth=1.0)
    axes[0, 1].set_title("2. Preprocessed Vibration (Zero-DC & Bandpass [0.5-25 Hz])", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Time (s)", fontsize=10)
    axes[0, 1].set_ylabel("Filtered Vibration (m/s²)", fontsize=10)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # 3. FFT Magnitude Spectrum
    f_mask = freqs <= 15.0
    axes[1, 0].plot(freqs[f_mask], amp[f_mask], color="#8e44ad", linewidth=1.6, label="Spectrum")
    axes[1, 0].axvline(
        dom_freq,
        color="#c0392b",
        linestyle="--",
        label=f"Dominant Mode: {dom_freq:.2f} Hz ({dom_amp:.3f} m/s²)",
    )
    axes[1, 0].set_title("3. FFT Magnitude Spectrum (Physical Calibrated Amplitude)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Frequency (Hz)", fontsize=10)
    axes[1, 0].set_ylabel("Amplitude (m/s²)", fontsize=10)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend(loc="upper right", fontsize=9)

    # 4. Optical Displacement
    axes[1, 1].plot(time, x_disp, label="Relative X", color="#27ae60", alpha=0.8, linewidth=1.1)
    axes[1, 1].plot(time, y_disp, label="Relative Y", color="#16a085", alpha=0.7, linewidth=1.0)
    axes[1, 1].plot(time, r_disp, label="Magnitude R(t)", color="#2c3e50", alpha=0.85, linewidth=1.3)
    axes[1, 1].set_title("4. Optical Relative Displacement & Magnitude R(t)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Time (s)", fontsize=10)
    axes[1, 1].set_ylabel("Displacement (mm)", fontsize=10)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend(loc="upper right", fontsize=9)

    # 5. Optical Velocity
    axes[2, 0].plot(time, vel_mag, color="#d35400", linewidth=1.1)
    axes[2, 0].set_title("5. Optical Velocity Magnitude |v(t)| = sqrt(vx² + vy²)", fontsize=11, fontweight="bold")
    axes[2, 0].set_xlabel("Time (s)", fontsize=10)
    axes[2, 0].set_ylabel("Velocity (mm/s)", fontsize=10)
    axes[2, 0].grid(True, linestyle="--", alpha=0.5)

    # 6. Optical Acceleration
    axes[2, 1].plot(time, accel_mag, color="#c0392b", linewidth=1.0)
    axes[2, 1].set_title("6. Optical Acceleration Magnitude |a(t)| = sqrt(ax² + ay²)", fontsize=11, fontweight="bold")
    axes[2, 1].set_xlabel("Time (s)", fontsize=10)
    axes[2, 1].set_ylabel("Acceleration (mm/s²)", fontsize=10)
    axes[2, 1].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        f"Diagnostic Signal Processing Features — Scenario: {event.scenario}",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved diagnostic plot to: {output_path.resolve()}")


def plot_spectral_comparison(sim: StructuralSimulator, output_path: Path):
    """Generate comparative frequency spectrum plot for NORMAL, WARNING, CRITICAL."""
    scenarios = ["NORMAL", "WARNING", "CRITICAL"]
    palette = {"NORMAL": "#2ecc71", "WARNING": "#f39c12", "CRITICAL": "#e74c3c"}

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    for scn in scenarios:
        ev = sim.generate(scn, seed=42)
        fs = float(ev.config.sampling_rate)
        # Preprocess: remove DC
        _, proc_vib = preprocess_signal(ev.vibration, sampling_rate=fs, remove_mean=True)
        freqs, amp = compute_spectrum(proc_vib, sampling_rate=fs, window="hann")
        dom_freq, dom_amp = get_dominant_frequency(proc_vib, sampling_rate=fs, min_freq=0.5)

        f_mask = (freqs >= 0.5) & (freqs <= 12.0)
        ax.plot(
            freqs[f_mask],
            amp[f_mask],
            label=f"{scn} (Dominant: {dom_freq:.2f} Hz, Amplitude: {dom_amp:.3f} m/s²)",
            color=palette[scn],
            linewidth=2.0,
        )
        ax.axvline(dom_freq, color=palette[scn], linestyle=":", alpha=0.7)

    ax.set_title(
        "Frequency Spectra Comparison — NORMAL vs WARNING vs CRITICAL\n"
        "(Progressive Resonance Frequency Shift & Decreased Damping)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Vibration Amplitude Spectrum (m/s²)", fontsize=11)
    ax.set_xlim(0.5, 12.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved spectral comparison plot to: {output_path.resolve()}")


def main():
    sim = StructuralSimulator(seed=42)
    plot_dir = REPO_ROOT / "plots" / "signal_processing"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Diagnostic plot for WARNING scenario event
    warning_event = sim.generate("WARNING", seed=42)
    plot_diagnostic_features(warning_event, plot_dir / "diagnostic_features.png")

    # 2. Spectral comparison across all scenarios
    plot_spectral_comparison(sim, plot_dir / "spectral_comparison.png")


if __name__ == "__main__":
    main()
