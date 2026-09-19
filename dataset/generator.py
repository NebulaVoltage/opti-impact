"""Dataset generation engine.

Orchestrates parameterized simulation, signal preprocessing, and feature extraction
to construct balanced, non-trivially overlapping machine learning datasets.
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Ensure repository root is available
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.simulator import StructuralSimulator
from signal_processing.feature_pipeline import FeaturePipeline
from dataset.variation import DEFAULT_VARIATION_PRESETS, ScenarioVariationConfig, get_variation_config
from dataset.schema import FEATURE_COLUMNS, METADATA_COLUMNS, TARGET_COLUMN, VALID_SCENARIOS


class DatasetGenerator:
    """Generates synthetic tabular feature datasets for machine learning."""

    def __init__(
        self,
        seed: int = 42,
        variation_presets: Optional[Dict[str, ScenarioVariationConfig]] = None,
        pipeline: Optional[FeaturePipeline] = None,
    ):
        """Initialize generator with reproducible seed and pipeline configuration.

        Args:
            seed: Base random seed for reproducible stochastic dataset synthesis.
            variation_presets: Optional dictionary of ScenarioVariationConfig instances.
            pipeline: Optional FeaturePipeline instance (defaults to standard pipeline with Savitzky-Golay).
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.presets = variation_presets or DEFAULT_VARIATION_PRESETS
        self.pipeline = pipeline or FeaturePipeline(
            remove_mean=True,
            min_freq=0.5,
            smoothing_method="savgol",
            smoothing_window=11,
            polynomial_order=3,
        )
        self.simulator = StructuralSimulator(seed=seed)

    def generate_single_record(
        self,
        scenario_name: str,
        event_index: int,
        seed: int,
    ) -> Dict[str, object]:
        """Generate one independent structural simulation event and extract its feature record.

        Args:
            scenario_name: 'NORMAL', 'WARNING', or 'CRITICAL'.
            event_index: Integer sequence index for tracking and unique event_id creation.
            seed: Distinct reproducible seed for this specific event realization.

        Returns:
            Dictionary containing measurable features, tracking metadata, and target scenario label.
        """
        event_rng = np.random.default_rng(seed)
        var_config = get_variation_config(scenario_name, self.presets)

        # 1. Sample continuous physical parameters with controlled boundary overlap
        sampled_scenario_config = var_config.sample_config(event_rng)

        # 2. Simulate physical time-series
        event = self.simulator.generate(sampled_scenario_config, seed=seed)

        # 3. Extract measurable engineering features via Step 2 pipeline
        features = self.pipeline.extract(event)
        feat_dict = features.to_dict()

        # Remove scenario from feature dict to handle it strictly as target
        feat_dict.pop("scenario", None)

        # 4. Construct metadata record (audit parameters, strictly segregated from features)
        record_id = f"{scenario_name.lower()}_{event_index:05d}"
        metadata = {
            "event_id": record_id,
            "random_seed": seed,
            "sim_natural_frequency": float(sampled_scenario_config.effective_frequency),
            "sim_amplitude": float(sampled_scenario_config.amplitude),
            "sim_damping_ratio": float(sampled_scenario_config.damping_ratio),
            "sim_impact_strength": float(sampled_scenario_config.impact_strength),
            "sim_noise_level": float(sampled_scenario_config.noise_level),
            "sim_optical_noise": float(sampled_scenario_config.optical_noise),
            "sim_sampling_rate": float(sampled_scenario_config.sampling_rate),
            "sim_duration": float(sampled_scenario_config.duration),
        }

        # Combine: features + metadata + target
        full_record = {**feat_dict, **metadata, TARGET_COLUMN: scenario_name}
        return full_record

    def generate_dataset(
        self,
        samples_per_class: int = 1000,
        scenarios: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Generate a complete balanced synthetic feature dataset across scenarios.

        Args:
            samples_per_class: Number of events per scenario (default 1000, total 3000).
            scenarios: List of scenarios to generate (defaults to ['NORMAL', 'WARNING', 'CRITICAL']).
            verbose: Whether to print generation progress.

        Returns:
            pandas DataFrame containing all generated records.
        """
        target_scenarios = scenarios or VALID_SCENARIOS
        total_events = samples_per_class * len(target_scenarios)

        if verbose:
            print(f"[DatasetGenerator] Generating {total_events} events ({samples_per_class} per class across {target_scenarios})...")
            print(f"[DatasetGenerator] Using base seed: {self.seed}")

        records: List[Dict[str, object]] = []
        global_idx = 1

        for scn in target_scenarios:
            if verbose:
                print(f"  -> Synthesizing class: {scn} ({samples_per_class} samples)...")

            for i in range(1, samples_per_class + 1):
                # Deterministic unique seed per event derived from base seed
                event_seed = int(self.rng.integers(0, 2**31 - 1))
                rec = self.generate_single_record(
                    scenario_name=scn,
                    event_index=global_idx,
                    seed=event_seed,
                )
                records.append(rec)
                global_idx += 1

        df = pd.DataFrame(records)

        # Enforce column ordering: metadata first, then features, then target
        desired_cols = METADATA_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
        # Only retain columns that exist
        ordered_cols = [c for c in desired_cols if c in df.columns]
        df = df[ordered_cols]

        if verbose:
            print(f"[DatasetGenerator] Completed generation. DataFrame shape: {df.shape}")

        return df
