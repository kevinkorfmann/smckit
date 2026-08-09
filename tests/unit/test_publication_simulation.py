"""Tests for every frozen publication simulation scenario."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import tskit

SCRIPT = (
    Path(__file__).resolve().parents[2] / "workflow" / "publication" / "scripts" / "simulate.py"
)
SPEC = importlib.util.spec_from_file_location("smckit_publication_simulate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SIMULATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SIMULATE
SPEC.loader.exec_module(SIMULATE)


@pytest.mark.parametrize(
    "scenario",
    [
        "constant",
        "bottleneck",
        "expansion",
        "split_with_migration",
        "structure",
        "selfing_dormancy",
    ],
)
def test_publication_scenario_generates_tree_and_truth(scenario, tmp_path) -> None:
    tree_path = tmp_path / f"{scenario}.trees"
    holdout_path = tmp_path / f"{scenario}.holdout.trees"
    truth_path = tmp_path / f"{scenario}.truth.json"
    payload = SIMULATE.simulate_scenario(
        scenario_name=scenario,
        replicate=1,
        seed=100,
        sequence_length=20_000,
        recombination_rate=1e-8,
        mutation_rate=1.25e-8,
        tree_output=tree_path,
        truth_output=truth_path,
        holdout_tree_output=holdout_path,
    )

    trees = tskit.load(tree_path)
    holdout = tskit.load(holdout_path)
    assert truth_path.is_file()
    assert trees.sequence_length == 20_000
    assert payload["scenario"] == scenario
    assert payload["tree_sequence"]["sha256"]
    assert payload["holdout_tree_sequence"]["sha256"]
    assert payload["holdout_ancestry_seed"] == 102
    assert payload["holdout_mutation_seed"] == 103
    assert holdout.sequence_length == trees.sequence_length
    assert np.isfinite(payload["diversity"])


def test_selfing_dormancy_uses_vendored_esmc2_transform() -> None:
    scenario = SIMULATE.demographic_scenario("selfing_dormancy", 1e-8)
    truth = scenario.truth
    expected_f = 0.8 / (2.0 - 0.8)
    assert truth["inbreeding_coefficient"] == pytest.approx(expected_f)
    assert truth["effective_population_size"] == pytest.approx(24_000)
    assert scenario.recombination_rate == pytest.approx(1e-8 * 0.5 * (1 - expected_f))


def test_unknown_publication_scenario_fails() -> None:
    with pytest.raises(ValueError, match="Unknown scenario"):
        SIMULATE.demographic_scenario("unknown")
