"""Automated test suite for the synthetic structural response simulator.

Verifies:
1. Generation of all three scenarios (NORMAL, WARNING, CRITICAL).
2. Exact expected number of samples (duration * sampling_rate).
3. Absence of NaN values across all signals.
4. Absence of infinite values across all signals.
5. Correct sampling and monotonicity of the time vector.
6. Strong correlation between optical measurements and true structural displacement.
7. CSV export formatting, headers, and file integrity.
8. Different random seeds produce distinct signal realizations.
9. Identical random seeds produce exact bitwise reproducible signals.
10. Physical consistency (amplitude scaling and decay rate differences across scenarios).
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pytest

from simulation.config import ScenarioConfig
from simulation.scenarios import get_scenario_config
from simulation.simulator import StructuralSimulator, generate_dataset


@pytest.fixture
def simulator() -> StructuralSimulator:
    """Fixture providing a StructuralSimulator instance initialized with seed 42."""
    return StructuralSimulator(seed=42)


@pytest.mark.parametrize("scenario_name", ["NORMAL", "WARNING", "CRITICAL"])
def test_all_three_scenarios_generate(simulator: StructuralSimulator, scenario_name: str):
    """Verify that NORMAL, WARNING, and CRITICAL scenarios can be generated without error."""
    event = simulator.generate(scenario_name)
    assert event is not None
    assert event.scenario == scenario_name
    assert hasattr(event, "time")
    assert hasattr(event, "vibration")
    assert hasattr(event, "true_displacement")
    assert hasattr(event, "optical_x")
    assert hasattr(event, "optical_y")
    assert hasattr(event, "optical_rotation")


@pytest.mark.parametrize("scenario_name", ["NORMAL", "WARNING", "CRITICAL"])
def test_signals_have_expected_sample_count(simulator: StructuralSimulator, scenario_name: str):
    """Verify that all generated signals have duration * sampling_rate samples."""
    event = simulator.generate(scenario_name)
    expected_samples = int(event.config.duration * event.config.sampling_rate)

    assert len(event.time) == expected_samples
    assert len(event.vibration) == expected_samples
    assert len(event.true_displacement) == expected_samples
    assert len(event.optical_x) == expected_samples
    assert len(event.optical_y) == expected_samples
    assert len(event.optical_rotation) == expected_samples


@pytest.mark.parametrize("scenario_name", ["NORMAL", "WARNING", "CRITICAL"])
def test_no_nan_values(simulator: StructuralSimulator, scenario_name: str):
    """Verify that no NaN values exist in any signal."""
    event = simulator.generate(scenario_name)

    assert not np.isnan(event.time).any(), "NaN found in time vector"
    assert not np.isnan(event.vibration).any(), "NaN found in vibration"
    assert not np.isnan(event.true_displacement).any(), "NaN found in true_displacement"
    assert not np.isnan(event.optical_x).any(), "NaN found in optical_x"
    assert not np.isnan(event.optical_y).any(), "NaN found in optical_y"
    assert not np.isnan(event.optical_rotation).any(), "NaN found in optical_rotation"


@pytest.mark.parametrize("scenario_name", ["NORMAL", "WARNING", "CRITICAL"])
def test_no_inf_values(simulator: StructuralSimulator, scenario_name: str):
    """Verify that no infinite values exist in any signal."""
    event = simulator.generate(scenario_name)

    assert not np.isinf(event.time).any(), "Inf found in time vector"
    assert not np.isinf(event.vibration).any(), "Inf found in vibration"
    assert not np.isinf(event.true_displacement).any(), "Inf found in true_displacement"
    assert not np.isinf(event.optical_x).any(), "Inf found in optical_x"
    assert not np.isinf(event.optical_y).any(), "Inf found in optical_y"
    assert not np.isinf(event.optical_rotation).any(), "Inf found in optical_rotation"


def test_time_vector_correctly_sampled(simulator: StructuralSimulator):
    """Verify that the time vector starts at 0.0 and has uniform delta_t = 1/sampling_rate."""
    event = simulator.generate("NORMAL")
    dt = 1.0 / event.config.sampling_rate

    assert event.time[0] == 0.0
    diffs = np.diff(event.time)
    assert np.allclose(diffs, dt, atol=1e-9), "Time steps are not uniformly 1/sampling_rate"
    assert np.all(diffs > 0), "Time vector must be strictly monotonically increasing"


@pytest.mark.parametrize("scenario_name", ["NORMAL", "WARNING", "CRITICAL"])
def test_optical_measurements_correlated_with_displacement(
    simulator: StructuralSimulator, scenario_name: str
):
    """Verify that optical_x measurement strongly correlates with underlying true_displacement."""
    event = simulator.generate(scenario_name, seed=123)

    corr_matrix = np.corrcoef(event.true_displacement, event.optical_x)
    pearson_r = corr_matrix[0, 1]

    # Ground truth and optical sensor must be strongly correlated (r > 0.85)
    assert pearson_r > 0.85, (
        f"Expected Pearson r > 0.85 for scenario {scenario_name}, got {pearson_r:.4f}"
    )


def test_csv_export(simulator: StructuralSimulator, tmp_path: Path):
    """Verify that CSV export works, creates a valid file, and preserves required columns."""
    event = simulator.generate("NORMAL", seed=42)
    export_file = tmp_path / "test_export.csv"

    returned_path = event.to_csv(export_file)
    assert export_file.exists()
    assert returned_path == export_file

    df = pd.read_csv(export_file)
    expected_columns = [
        "timestamp",
        "true_displacement",
        "vibration",
        "optical_x",
        "optical_y",
        "optical_rotation",
        "scenario",
    ]
    assert list(df.columns) == expected_columns
    assert len(df) == len(event.time)
    assert df["scenario"].iloc[0] == "NORMAL"
    assert np.allclose(df["timestamp"].values, event.time, atol=1e-5)
    assert np.allclose(df["true_displacement"].values, event.true_displacement, atol=1e-5)


def test_batch_dataset_generation(tmp_path: Path):
    """Verify generate_dataset function creates expected files with distinct runs."""
    files = generate_dataset("NORMAL", count=3, output_dir=tmp_path, base_seed=100)
    assert len(files) == 3

    for f in files:
        assert f.exists()
        assert f.name.startswith("normal_event_")
        assert f.suffix == ".csv"

    # Verify files generated from different seeds contain different data
    df1 = pd.read_csv(files[0])
    df2 = pd.read_csv(files[1])
    assert not np.allclose(df1["true_displacement"], df2["true_displacement"])


def test_different_seeds_produce_different_events(simulator: StructuralSimulator):
    """Verify that different random seeds generate distinct stochastic events."""
    event1 = simulator.generate("WARNING", seed=101)
    event2 = simulator.generate("WARNING", seed=202)

    assert not np.array_equal(event1.true_displacement, event2.true_displacement)
    assert not np.array_equal(event1.vibration, event2.vibration)
    assert not np.array_equal(event1.optical_x, event2.optical_x)


def test_same_seed_produces_reproducible_results(simulator: StructuralSimulator):
    """Verify that identical random seeds produce identical, bitwise-reproducible signals."""
    event1 = simulator.generate("CRITICAL", seed=999)
    event2 = simulator.generate("CRITICAL", seed=999)

    np.testing.assert_array_equal(event1.time, event2.time)
    np.testing.assert_array_equal(event1.true_displacement, event2.true_displacement)
    np.testing.assert_array_equal(event1.vibration, event2.vibration)
    np.testing.assert_array_equal(event1.optical_x, event2.optical_x)
    np.testing.assert_array_equal(event1.optical_y, event2.optical_y)
    np.testing.assert_array_equal(event1.optical_rotation, event2.optical_rotation)


def test_physical_consistency_across_scenarios(simulator: StructuralSimulator):
    """Verify that amplitude increases from NORMAL to WARNING to CRITICAL."""
    event_normal = simulator.generate("NORMAL", seed=42)
    event_warning = simulator.generate("WARNING", seed=42)
    event_critical = simulator.generate("CRITICAL", seed=42)

    max_disp_normal = np.max(np.abs(event_normal.true_displacement))
    max_disp_warning = np.max(np.abs(event_warning.true_displacement))
    max_disp_critical = np.max(np.abs(event_critical.true_displacement))

    assert max_disp_normal < max_disp_warning < max_disp_critical, (
        f"Peak displacements should follow normal < warning < critical. "
        f"Got: normal={max_disp_normal:.3f}, warning={max_disp_warning:.3f}, critical={max_disp_critical:.3f}"
    )


def test_invalid_scenario_raises_value_error(simulator: StructuralSimulator):
    """Verify that requesting an invalid scenario name raises a descriptive ValueError."""
    with pytest.raises(ValueError, match="Unknown scenario 'INVALID'"):
        simulator.generate("INVALID")
