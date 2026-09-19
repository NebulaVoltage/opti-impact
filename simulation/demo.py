"""Demonstration script showcasing the StructuralSimulator API.

Generates:
1. Single events for NORMAL, WARNING, CRITICAL.
2. Direct inspection of attributes and DataFrame conversion.
3. Export of sample events to dataset/raw/.
4. Multi-event batch dataset generation.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.simulator import StructuralSimulator


def run_demonstration():
    print("=" * 70)
    print("AI Optical Structural Impact Monitoring — Simulation Demonstration")
    print("=" * 70)

    sim = StructuralSimulator(seed=42)
    output_dir = REPO_ROOT / "dataset" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = ["NORMAL", "WARNING", "CRITICAL"]

    for scn in scenarios:
        print(f"\n--> Generating single event for scenario: {scn}")
        event = sim.generate(scn, seed=42)

        print(f"    Scenario:              {event.scenario}")
        print(f"    Duration:              {event.config.duration} s")
        print(f"    Sampling Rate:         {event.config.sampling_rate} Hz ({len(event.time)} samples)")
        print(f"    Effective fn:          {event.config.effective_frequency:.2f} Hz")
        print(f"    Damping ratio (zeta):  {event.config.damping_ratio:.4f}")
        print(f"    Peak True Disp:        {event.true_displacement.max():.3f} mm (min: {event.true_displacement.min():.3f} mm)")
        print(f"    Peak Optical X:        {event.optical_x.max():.3f} mm (min: {event.optical_x.min():.3f} mm)")
        print(f"    Peak Vibration:        {event.vibration.max():.3f} m/s² (min: {event.vibration.min():.3f} m/s²)")

        # Save single sample CSV
        out_file = output_dir / f"{scn.lower()}_event_001.csv"
        saved_path = event.to_csv(out_file)
        print(f"    [+] Saved sample to:    {saved_path.name}")

    print("\n--> Generating batch dataset (3 events per scenario)...")
    for scn in scenarios:
        files = sim.generate_dataset(scenario=scn, count=3, output_dir=output_dir, base_seed=100)
        print(f"    [+] {scn}: Created {len(files)} files: {[f.name for f in files]}")

    print("\n[+] Demonstration completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_demonstration()
