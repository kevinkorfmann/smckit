"""Generate frozen msprime simulations and machine-readable demographic truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msprime


@dataclass(frozen=True)
class Scenario:
    demography: msprime.Demography
    samples: dict[str, int]
    recombination_rate: float
    truth: dict[str, Any]


def demographic_scenario(name: str, recombination_rate: float = 1e-8) -> Scenario:
    """Return one publication scenario with explicit truth metadata."""
    if recombination_rate <= 0:
        raise ValueError("recombination_rate must be positive.")
    demography = msprime.Demography()

    if name == "split_with_migration":
        demography.add_population(name="ancestral", initial_size=10_000)
        demography.add_population(name="p1", initial_size=10_000)
        demography.add_population(name="p2", initial_size=12_000)
        migration_rate = 1e-5
        demography.set_symmetric_migration_rate(["p1", "p2"], rate=migration_rate)
        demography.add_population_split(
            time=2_000,
            derived=["p1", "p2"],
            ancestral="ancestral",
        )
        return Scenario(
            demography=demography,
            samples={"p1": 4, "p2": 4},
            recombination_rate=recombination_rate,
            truth={
                "kind": "split_with_migration",
                "population_sizes": {"p1": 10_000, "p2": 12_000, "ancestral": 10_000},
                "split_time_generations": 2_000,
                "symmetric_migration_rate": migration_rate,
            },
        )

    if name == "structure":
        demography.add_population(name="deme_a", initial_size=8_000)
        demography.add_population(name="deme_b", initial_size=12_000)
        migration_rate = 2e-5
        demography.set_symmetric_migration_rate(
            ["deme_a", "deme_b"],
            rate=migration_rate,
        )
        return Scenario(
            demography=demography,
            samples={"deme_a": 4, "deme_b": 4},
            recombination_rate=recombination_rate,
            truth={
                "kind": "persistent_two_deme_structure",
                "population_sizes": {"deme_a": 8_000, "deme_b": 12_000},
                "symmetric_migration_rate": migration_rate,
            },
        )

    if name == "selfing_dormancy":
        census_size = 10_000
        sigma = 0.8
        beta = 0.5
        inbreeding = sigma / (2.0 - sigma)
        effective_size = census_size * (1.0 - 0.5 * sigma) / beta**2
        effective_recombination = recombination_rate * beta * (1.0 - inbreeding)
        demography.add_population(name="population", initial_size=effective_size)
        return Scenario(
            demography=demography,
            samples={"population": 8},
            recombination_rate=effective_recombination,
            truth={
                "kind": "esmc2_coalescent_equivalent",
                "census_size": census_size,
                "sigma_selfing": sigma,
                "beta_germination": beta,
                "inbreeding_coefficient": inbreeding,
                "effective_population_size": effective_size,
                "baseline_recombination_rate": recombination_rate,
                "effective_recombination_rate": effective_recombination,
                "transform_source": (
                    "vendored eSMC2 Tutorial_2_git(simulation)/mspts_SS_TS.py"
                ),
            },
        )

    demography.add_population(name="population", initial_size=10_000)
    events: list[dict[str, Any]] = []
    if name == "bottleneck":
        demography.add_population_parameters_change(time=500, initial_size=1_000)
        demography.add_population_parameters_change(time=1_000, initial_size=10_000)
        events = [
            {"time_generations": 0, "population_size": 10_000},
            {"time_generations": 500, "population_size": 1_000},
            {"time_generations": 1_000, "population_size": 10_000},
        ]
    elif name == "expansion":
        demography.add_population_parameters_change(time=500, initial_size=2_000)
        events = [
            {"time_generations": 0, "population_size": 10_000},
            {"time_generations": 500, "population_size": 2_000},
        ]
    elif name == "constant":
        events = [{"time_generations": 0, "population_size": 10_000}]
    else:
        raise ValueError(f"Unknown scenario: {name}")
    return Scenario(
        demography=demography,
        samples={"population": 8},
        recombination_rate=recombination_rate,
        truth={"kind": name, "population_size_epochs": events},
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def simulate_scenario(
    *,
    scenario_name: str,
    replicate: int,
    seed: int,
    sequence_length: float,
    recombination_rate: float,
    mutation_rate: float,
    tree_output: Path,
    truth_output: Path,
) -> dict[str, Any]:
    """Simulate and persist one publication replicate."""
    if replicate < 1:
        raise ValueError("replicate must be positive.")
    if seed <= 0:
        raise ValueError("seed must be positive.")
    if sequence_length <= 0 or mutation_rate <= 0:
        raise ValueError("sequence_length and mutation_rate must be positive.")
    scenario = demographic_scenario(scenario_name, recombination_rate)
    trees = msprime.sim_ancestry(
        samples=scenario.samples,
        demography=scenario.demography,
        sequence_length=sequence_length,
        recombination_rate=scenario.recombination_rate,
        random_seed=seed,
    )
    mutated = msprime.sim_mutations(
        trees,
        rate=mutation_rate,
        random_seed=seed + 1,
    )
    tree_output.parent.mkdir(parents=True, exist_ok=True)
    truth_output.parent.mkdir(parents=True, exist_ok=True)
    mutated.dump(tree_output)
    payload = {
        "schema_version": 1,
        "scenario": scenario_name,
        "replicate": replicate,
        "ancestry_seed": seed,
        "mutation_seed": seed + 1,
        "sequence_length": float(mutated.sequence_length),
        "mutation_rate": mutation_rate,
        "recombination_rate": scenario.recombination_rate,
        "samples": scenario.samples,
        "n_sample_nodes": mutated.num_samples,
        "n_trees": mutated.num_trees,
        "n_sites": mutated.num_sites,
        "diversity": float(mutated.diversity()),
        "truth": scenario.truth,
        "software": {"msprime": msprime.__version__},
        "tree_sequence": {
            "path": str(tree_output.resolve()),
            "sha256": sha256_file(tree_output),
        },
    }
    truth_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sequence-length", type=float, default=1_000_000)
    parser.add_argument("--recombination-rate", type=float, default=1e-8)
    parser.add_argument("--mutation-rate", type=float, default=1.25e-8)
    parser.add_argument("--tree-output", required=True)
    parser.add_argument("--truth-output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    simulate_scenario(
        scenario_name=args.scenario,
        replicate=args.replicate,
        seed=args.seed,
        sequence_length=args.sequence_length,
        recombination_rate=args.recombination_rate,
        mutation_rate=args.mutation_rate,
        tree_output=Path(args.tree_output),
        truth_output=Path(args.truth_output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
