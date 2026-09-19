"""Demonstration CLI script for real-time multimodal inference and temporal decision engine.

Simulates a 100-second sequence of structural response transitions:
    0–20 s:  NORMAL
    20–40 s: WARNING
    40–60 s: CRITICAL
    60–80 s: WARNING
    80–100 s: NORMAL

Prints formatted, judge-friendly window diagnostics and state transition banners.
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Optional

from realtime.config import RealtimePipelineConfig
from realtime.data_stream import SyntheticSensorStream
from realtime.runtime import RealtimeMonitoringRuntime
from realtime.state_machine import StateTransitionEvent
from realtime.telemetry import InferenceTelemetry


def run_demo(sleep_interval: float = 0.0, seed: int = 42) -> None:
    """Run interactive CLI demonstration across 100 seconds of simulated structural data.

    Args:
        sleep_interval: Optional simulated real-time delay per window (seconds). Default 0.0.
        seed: Deterministic random seed.
    """
    print("=" * 64)
    print(" OPTICAL STRUCTURAL RESPONSE MONITOR")
    print(" REAL-TIME MULTIMODAL INFERENCE & TEMPORAL DECISION ENGINE")
    print("=" * 64)
    print("Simulated Multi-Scenario Sequence:")
    print("  [00.0s - 20.0s] : NORMAL")
    print("  [20.0s - 40.0s] : WARNING")
    print("  [40.0s - 60.0s] : CRITICAL")
    print("  [60.0s - 80.0s] : WARNING")
    print("  [80.0s - 100.0s]: NORMAL")
    print("-" * 64)

    # 1. Setup multi-segment schedule
    schedule = [
        ("NORMAL", 20.0),
        ("WARNING", 20.0),
        ("CRITICAL", 20.0),
        ("WARNING", 20.0),
        ("NORMAL", 20.0),
    ]

    config = RealtimePipelineConfig()
    stream = SyntheticSensorStream(
        scenario_schedule=schedule,
        sampling_rate=config.stream.sampling_rate,
        base_seed=seed,
    )

    print(f"Generated continuous stream: {stream.total_duration:.1f}s ({stream.total_samples} samples)")
    print(f"Window Duration: {config.stream.window_duration:.1f}s | Step: {config.stream.inference_step:.1f}s")
    print(f"Persistence Threshold: 3 consecutive windows")
    print("=" * 64 + "\n")

    runtime = RealtimeMonitoringRuntime(config=config, stream=stream)

    def print_callback(telemetry: InferenceTelemetry, transition: Optional[StateTransitionEvent]) -> None:
        win_str = f"{telemetry.window_id:03d}"
        t_range = f"[{telemetry.timestamp - config.stream.window_duration:.1f}s - {telemetry.timestamp:.1f}s]"

        print(f"Window: {win_str}  Time: {t_range}")
        print(f"Instant Prediction: {telemetry.instant_prediction}")
        print(f"P(NORMAL):   {telemetry.probability_normal:.3f}")
        print(f"P(WARNING):  {telemetry.probability_warning:.3f}")
        print(f"P(CRITICAL): {telemetry.probability_critical:.3f}")

        # Persistence status
        if telemetry.candidate_state is not None and telemetry.candidate_count > 0:
            req = runtime.temporal_engine.get_persistence_threshold(telemetry.candidate_state)
            pers_str = f"{telemetry.candidate_state} persistence: {telemetry.candidate_count}/{req}"
        else:
            pers_str = "Stable"

        print(f"Stable State: {telemetry.stable_state}  ({pers_str})")
        print(
            f"Vib RMS: {telemetry.vibration_rms:.3f} m/s^2 | "
            f"Freq: {telemetry.dominant_frequency:.2f} Hz | "
            f"Opt Disp: {telemetry.optical_displacement:.2f} mm | "
            f"Corr: {telemetry.vibration_optical_correlation:.2f}"
        )

        if transition is not None:
            print("\n" + "*" * 50)
            print(f"  STATE TRANSITION CONFIRMED: {transition.previous_state} -> {transition.new_state}")
            print(f"  Reason: {transition.reason} ({transition.consecutive_windows} agreeing windows)")
            print("*" * 50)

        print("-" * 64)

        if sleep_interval > 0:
            time.sleep(sleep_interval)

    # Run session
    summary = runtime.run_all(callback=print_callback)

    print("\n" + "=" * 64)
    print("DEMO EXECUTION COMPLETE")
    print("=" * 64)
    print(f"Total Windows Evaluated:     {summary['number_of_windows']}")
    print(f"State Transitions Triggered: {summary['number_of_state_transitions']}")
    print(f"Final Stable State:          {summary['final_state']}")
    print(f"Mean Latency per Window:     {summary['mean_latency']:.2f} ms")
    print(f"Median Latency:              {summary['median_latency']:.2f} ms")
    print(f"95th Percentile Latency:     {summary['p95_latency']:.2f} ms")
    print(f"Max Latency:                 {summary['max_latency']:.2f} ms")
    print(f"Telemetry Log:               {config.output_dir / config.inference_log_file}")
    print(f"Transition Events Log:       {config.output_dir / config.state_events_file}")
    print(f"Session Summary:             {config.output_dir / config.summary_file}")
    print("=" * 64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run real-time monitoring demonstration.")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional simulated delay per window in seconds (default: 0.0 for immediate execution).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed (default: 42).")
    args = parser.parse_args()
    run_demo(sleep_interval=args.delay, seed=args.seed)
