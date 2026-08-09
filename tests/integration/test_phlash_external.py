"""Real-package smoke test for the maintained PHLASH integration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import msprime
import numpy as np
import pytest

from smckit.tl import phlash

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]


def _load_publication_script(name: str):
    path = ROOT / "workflow" / "publication" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"smckit_integration_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(importlib.util.find_spec("phlash") is None, reason="PHLASH is not installed")
def test_installed_phlash_psmcfa_smoke(tmp_path) -> None:
    source = tmp_path / "synthetic.psmcfa"
    sequence = np.full(5_000, "T", dtype="<U1")
    sequence[::50] = "K"
    source.write_text(
        ">synthetic\n" + "".join(sequence.tolist()) + "\n",
        encoding="utf-8",
    )

    data = phlash(
        [source],
        input_kind="psmcfa",
        hold_out=False,
        random_seed=17,
        grid_size=16,
        niter=1,
        num_particles=4,
        num_workers=1,
        max_samples=1,
        overlap=20,
        progress=False,
    )
    result = data.results["phlash"]

    assert result["upstream"]["version"] == "1.0.6"
    assert result["n_posterior_samples"] == 4
    assert result["posterior_ne"].shape == (4, 16)
    assert np.all(np.isfinite(result["posterior_ne"]))
    assert result["provenance"]["seed"] == 17


def _simulated_tree_sequence() -> msprime.TreeSequence:
    ancestry = msprime.sim_ancestry(
        samples=4,
        ploidy=2,
        sequence_length=500_000,
        recombination_rate=1e-8,
        population_size=10_000,
        random_seed=811,
    )
    return msprime.sim_mutations(ancestry, rate=2e-8, random_seed=1811)


def _fit_options() -> dict[str, object]:
    return {
        "hold_out": False,
        "random_seed": 29,
        "window_size": 100,
        "grid_size": 16,
        "niter": 1,
        "num_particles": 4,
        "num_workers": 1,
        "max_samples": 1,
        "overlap": 20,
        "progress": False,
    }


def _assert_real_input_result(result: dict[str, object], input_kind: str) -> None:
    assert result["upstream"]["version"] == "1.0.6"
    assert result["provenance"]["arguments"]["input_kind"] == input_kind
    assert result["provenance"]["arguments"]["window_size"] == 100
    assert result["n_posterior_samples"] == 4
    posterior = np.asarray(result["posterior_ne"])
    assert posterior.shape == (4, 16)
    assert np.all(np.isfinite(posterior))


@pytest.mark.skipif(importlib.util.find_spec("phlash") is None, reason="PHLASH is not installed")
def test_installed_phlash_tree_sequence_smoke(tmp_path: Path) -> None:
    tree_sequence = _simulated_tree_sequence()
    source = tmp_path / "synthetic.trees"
    tree_sequence.dump(source)
    nodes = [tuple(individual.nodes) for individual in tree_sequence.individuals()]

    data = phlash(
        [source],
        input_kind="tree_sequence",
        samples=nodes,
        **_fit_options(),
    )

    _assert_real_input_result(data.results["phlash"], "tree_sequence")


@pytest.mark.skipif(importlib.util.find_spec("phlash") is None, reason="PHLASH is not installed")
def test_installed_phlash_indexed_vcf_smoke(tmp_path: Path) -> None:
    import pysam

    tree_sequence = _simulated_tree_sequence()
    source = tmp_path / "synthetic.vcf"
    sample_names = [f"sample-{index}" for index in range(tree_sequence.num_individuals)]
    with source.open("w", encoding="utf-8") as handle:
        tree_sequence.write_vcf(handle, contig_id="chr1", individual_names=sample_names)
    compressed = tmp_path / "synthetic.vcf.gz"
    pysam.tabix_compress(str(source), str(compressed), force=True)
    pysam.tabix_index(str(compressed), preset="vcf", force=True)

    data = phlash(
        [compressed],
        input_kind="vcf",
        samples=sample_names,
        region="chr1:1-500000",
        **_fit_options(),
    )

    _assert_real_input_result(data.results["phlash"], "vcf")


@pytest.mark.skipif(importlib.util.find_spec("phlash") is None, reason="PHLASH is not installed")
def test_installed_phlash_publication_runner_smoke(tmp_path: Path) -> None:
    """Exercise the evidence runner with real independent train/holdout inputs."""
    simulator = _load_publication_script("simulate")
    runner = _load_publication_script("run_phlash_accuracy")
    tree_path = tmp_path / "constant.trees"
    holdout_path = tmp_path / "constant.holdout.trees"
    truth_path = tmp_path / "constant.truth.json"
    truth = simulator.simulate_scenario(
        scenario_name="constant",
        replicate=1,
        seed=1_001,
        sequence_length=500_000,
        recombination_rate=1e-8,
        mutation_rate=1.25e-8,
        tree_output=tree_path,
        holdout_tree_output=holdout_path,
        truth_output=truth_path,
    )
    protocol = {
        "protocol_id": "smoke-test-not-publication-evidence",
        "source": {"sha256": "smoke-test"},
        "config": {
            "replicates": 1,
            "mutation_rate": 1.25e-8,
            "phlash": {
                "scenarios": ["constant"],
                "window_size": 100,
                "grid_size": 16,
                "credible_level": 0.95,
                "niter": 1,
                "num_particles": 4,
                "num_workers": 1,
                "max_samples": 1,
                "overlap": 20,
                "hold_out": True,
                "inference_seed": 2_718,
                "evaluation_min_generations": 100,
                "evaluation_max_generations": 1_000_000,
            },
        },
    }
    output_path = tmp_path / "constant.validation.json"
    record = runner.run_phlash_accuracy(
        protocol=protocol,
        truth_payload=truth,
        truth_path=truth_path,
        tree_path=tree_path,
        holdout_tree_path=holdout_path,
        artifact_prefix=tmp_path / "constant",
        output_path=output_path,
    )

    assert record["method"] == "phlash"
    assert record["protocol_id"] == "smoke-test-not-publication-evidence"
    assert record["inference"]["phlash_version"] == "1.0.6"
    assert record["inference"]["arguments"]["hold_out"] is True
    assert record["posterior"]["n_samples"] == 4
    assert np.isfinite(record["metrics"]["log_integrated_trajectory_error"])
    assert output_path.is_file()
