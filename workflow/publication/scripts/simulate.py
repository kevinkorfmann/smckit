from __future__ import annotations

import argparse
import json
from pathlib import Path

import msprime


def demographic_model(name: str) -> msprime.Demography:
    demography = msprime.Demography()
    if name == "split_with_migration":
        demography.add_population(name="ancestral", initial_size=10_000)
        demography.add_population(name="p1", initial_size=10_000)
        demography.add_population(name="p2", initial_size=12_000)
        demography.add_population_split(time=2_000, derived=["p1", "p2"], ancestral="ancestral")
        demography.set_symmetric_migration_rate(["p1", "p2"], rate=1e-5)
        return demography
    demography.add_population(name="population", initial_size=10_000)
    if name == "bottleneck":
        demography.add_population_parameters_change(time=500, initial_size=1_000)
        demography.add_population_parameters_change(time=1_000, initial_size=10_000)
    elif name == "expansion":
        demography.add_population_parameters_change(time=500, initial_size=2_000)
    elif name == "structure":
        raise NotImplementedError("Structure protocol awaits frozen population definition.")
    elif name == "selfing_dormancy":
        raise NotImplementedError("Selfing/dormancy requires the reviewed eSMC2 simulation model.")
    elif name != "constant":
        raise ValueError(f"Unknown scenario: {name}")
    return demography


parser = argparse.ArgumentParser()
parser.add_argument("--scenario", required=True)
parser.add_argument("--replicate", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
demography = demographic_model(args.scenario)
samples = {"p1": 4, "p2": 4} if args.scenario == "split_with_migration" else {"population": 8}
trees = msprime.sim_ancestry(
    samples=samples,
    demography=demography,
    sequence_length=1_000_000,
    recombination_rate=1e-8,
    random_seed=args.seed,
)
mutated = msprime.sim_mutations(trees, rate=1.25e-8, random_seed=args.seed + 1)
target = Path(args.output)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(
    json.dumps(
        {
            "scenario": args.scenario,
            "replicate": args.replicate,
            "seed": args.seed,
            "sequence_length": mutated.sequence_length,
            "n_samples": mutated.num_samples,
            "n_sites": mutated.num_sites,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
