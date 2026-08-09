"""Persistent fitted exponential-growth worker for diCal2 benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import msprime
import numpy as np

from smckit.io import read_dical2
from smckit.tl import dical2

DATASET = "exponential-growth-msprime-seed211-v1"
SEQUENCE_LENGTH = 5_000
SEED = 211
START_POINT = (0.8, 2.0)
BOUNDS = "0.01,20;0.05,50"


def _simulate() -> msprime.TreeSequence:
    demography = msprime.Demography()
    demography.add_population(
        name="p0",
        initial_size=9_771,
        growth_rate=1e-4,
    )
    demography.add_population_parameters_change(
        time=2_000,
        initial_size=8_000,
        growth_rate=0.0,
        population="p0",
    )
    demography.add_population_parameters_change(
        time=8_000,
        initial_size=10_000,
        growth_rate=0.0,
        population="p0",
    )
    demography.sort_events()
    ancestry = msprime.sim_ancestry(
        samples={"p0": 5},
        demography=demography,
        sequence_length=SEQUENCE_LENGTH,
        recombination_rate=1.25e-8,
        random_seed=SEED,
    )
    return msprime.sim_mutations(
        ancestry,
        rate=2e-7,
        model=msprime.BinaryMutationModel(),
        discrete_genome=True,
        random_seed=SEED + 1,
    )


def _write_fixture(directory: Path) -> tuple[Path, Path]:
    tree_sequence = _simulate()
    reference_path = directory / "exponential-growth.fa"
    reference_path.write_text("A" * SEQUENCE_LENGTH + "\n", encoding="utf-8")

    sample_nodes = np.asarray(tree_sequence.samples(), dtype=np.int64)
    sample_names = [f"sample_{idx}" for idx in range(len(sample_nodes) // 2)]
    rows = [
        "##fileformat=VCFv4.2",
        '##FILTER=<ID=PASS,Description="All filters passed">',
        f"##contig=<ID=1,length={SEQUENCE_LENGTH}>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "\t".join(
            ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
            + sample_names
        ),
    ]
    used_positions: set[int] = set()
    for variant in tree_sequence.variants(samples=sample_nodes):
        position = int(variant.site.position) + 1
        if position in used_positions:
            continue
        used_positions.add(position)
        genotypes = np.asarray(variant.genotypes, dtype=np.int8)
        diploids = [
            f"{int(genotypes[idx])}|{int(genotypes[idx + 1])}"
            for idx in range(0, len(genotypes), 2)
        ]
        rows.append(
            "\t".join(
                ["1", str(position), str(variant.site.id), "A", "C", ".", "PASS", ".", "GT"]
                + diploids
            )
        )
    if len(rows) == 5:
        raise RuntimeError("The deterministic growth benchmark produced no variants.")

    vcf_path = directory / "exponential-growth.vcf"
    vcf_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return vcf_path, reference_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "implementation": result["implementation"],
        "log_likelihood": float(result["log_likelihood"]),
        "best_params": np.asarray(result["best_params"], dtype=np.float64).tolist(),
        "core_type": result.get("core_type"),
        "number_iterations_mstep": int(result["resolved_options"]["number_iterations_mstep"]),
    }


def _write(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def run_worker(*, implementation: str) -> int:
    with tempfile.TemporaryDirectory(prefix="smckit-dical2-growth-benchmark-") as directory_name:
        directory = Path(directory_name)
        vcf_path, reference_path = _write_fixture(directory)
        example_root = Path("vendor/diCal2/examples/expGrowth").resolve()
        data_kwargs = {
            "sequences": vcf_path,
            "param_file": example_root / "mutRec.param",
            "demo_file": example_root / "exp_growth.demo",
            "rates_file": example_root / "exp_growth.rates",
            "config_file": example_root / "exp_growth.config",
            "reference_file": reference_path,
            "filter_pass_string": "PASS",
        }
        options = {
            "interval_type": "logUniform",
            "interval_params": "8,0.01,4",
        }
        prepared = None
        _write(
            {
                "event": "ready",
                "method": "dical2",
                "implementation": implementation,
                "dataset": DATASET,
                "vcf_sha256": _sha256(vcf_path),
                "reference_sha256": _sha256(reference_path),
            }
        )
        for raw in sys.stdin:
            request = json.loads(raw)
            if request.get("event") == "close":
                _write({"event": "closed"})
                return 0
            if request.get("event") == "prepare" and isinstance(request.get("repetition"), int):
                prepared = read_dical2(**data_kwargs)
                _write({"event": "prepared", "repetition": request["repetition"]})
                continue
            if request.get("event") != "run" or not isinstance(request.get("repetition"), int):
                raise ValueError("Expected a run event with an integer repetition.")
            if prepared is None:
                raise ValueError("Each run event must be preceded by a prepare event.")
            implementation_options = (
                {"native_options": options}
                if implementation == "native"
                else {"upstream_options": options}
            )
            fitted = dical2(
                prepared,
                implementation=implementation,
                n_em_iterations=1,
                start_point=START_POINT,
                seed=SEED,
                loci_per_hmm_step=50,
                composite_mode="pcl",
                bounds=BOUNDS,
                **implementation_options,
            ).results["dical2"]
            prepared = None
            _write(
                {
                    "event": "result",
                    "repetition": request["repetition"],
                    "result": _summary(fitted),
                }
            )
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("native", "upstream"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_worker(implementation=args.implementation)


if __name__ == "__main__":
    raise SystemExit(main())
