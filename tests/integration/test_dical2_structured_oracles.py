"""Independent simulated structured-demography oracles for native diCal2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import smckit
from smckit.io import read_dical2
from smckit.tl import dical2

msprime = pytest.importorskip("msprime")

pytestmark = [
    pytest.mark.oracle,
    pytest.mark.skipif(
        not smckit.upstream.status("dical2")["runtime_ready"],
        reason="Java runtime not available",
    ),
]

EXAMPLES = Path("vendor/diCal2/examples")
SEQUENCE_LENGTH = 5_000
STRUCTURED_LL_ABS_TOL = 1e-5


@dataclass(frozen=True)
class StructuredScenario:
    name: str
    example_dir: str
    demo_file: str
    config_file: str
    start_point: tuple[float, ...]
    demography: str
    seed: int


SCENARIOS = [
    StructuredScenario(
        name="clean_split",
        example_dir="cleanSplit",
        demo_file="clean_split.demo",
        config_file="clean_split.config",
        start_point=(0.2, 0.25, 0.25, 1.0),
        demography="split",
        seed=101,
    ),
    StructuredScenario(
        name="migration_window",
        example_dir="islolationMigrationWindow",
        demo_file="isolation_migration_window.demo",
        config_file="isolation_migration_window.config",
        start_point=(0.1, 0.2, 0.25, 0.25, 0.1, 1.0),
        demography="migration_window",
        seed=103,
    ),
    StructuredScenario(
        name="three_populations",
        example_dir="threePopulations",
        demo_file="three_populations.demo",
        config_file="three_populations.config",
        start_point=(0.2, 0.4),
        demography="three_populations",
        seed=107,
    ),
    StructuredScenario(
        name="introgression",
        example_dir="introgression",
        demo_file="introgression.demo",
        config_file="introgression.config",
        start_point=(0.05, 0.03),
        demography="introgression",
        seed=109,
    ),
]


def _two_population_demography(*, migration_window: bool) -> msprime.Demography:
    demography = msprime.Demography()
    demography.add_population(name="p0", initial_size=5_000)
    demography.add_population(name="p1", initial_size=5_000)
    demography.add_population(name="ancestral", initial_size=10_000)
    if migration_window:
        demography.set_symmetric_migration_rate(["p0", "p1"], rate=1e-4)
        demography.add_symmetric_migration_rate_change(
            time=2_000,
            populations=["p0", "p1"],
            rate=0.0,
        )
    demography.add_population_split(
        time=4_000,
        derived=["p0", "p1"],
        ancestral="ancestral",
    )
    demography.sort_events()
    return demography


def _three_population_demography(*, introgression: bool) -> msprime.Demography:
    demography = msprime.Demography()
    for name in ("p0", "p1", "p2", "p01", "ancestral"):
        demography.add_population(name=name, initial_size=10_000)
    if introgression:
        demography.add_mass_migration(
            time=1_000,
            source="p0",
            dest="p2",
            proportion=0.03,
        )
    demography.add_population_split(
        time=4_000,
        derived=["p0", "p1"],
        ancestral="p01",
    )
    demography.add_population_split(
        time=8_000,
        derived=["p01", "p2"],
        ancestral="ancestral",
    )
    demography.sort_events()
    return demography


def _simulate(scenario: StructuredScenario):
    if scenario.demography == "split":
        demography = _two_population_demography(migration_window=False)
        samples = {"p0": 2, "p1": 2}
    elif scenario.demography == "migration_window":
        demography = _two_population_demography(migration_window=True)
        samples = {"p0": 2, "p1": 2}
    elif scenario.demography in {"three_populations", "introgression"}:
        demography = _three_population_demography(
            introgression=scenario.demography == "introgression"
        )
        samples = {"p0": 2, "p1": 2, "p2": 2}
    else:  # pragma: no cover - guarded by the frozen scenario table
        raise AssertionError(f"Unknown structured scenario: {scenario.demography}")

    ancestry = msprime.sim_ancestry(
        samples=samples,
        demography=demography,
        sequence_length=SEQUENCE_LENGTH,
        recombination_rate=1.25e-8,
        random_seed=scenario.seed,
    )
    return msprime.sim_mutations(
        ancestry,
        rate=2e-7,
        model=msprime.BinaryMutationModel(),
        discrete_genome=True,
        random_seed=scenario.seed + 1,
    )


def _write_vcf_and_reference(ts, directory: Path, stem: str) -> tuple[Path, Path]:
    reference_path = directory / f"{stem}.fa"
    reference_path.write_text("A" * SEQUENCE_LENGTH + "\n", encoding="utf-8")

    sample_nodes = np.asarray(ts.samples(), dtype=np.int64)
    if len(sample_nodes) % 2:
        raise AssertionError("Structured diCal2 fixtures require diploid sample pairs.")
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
    for variant in ts.variants(samples=sample_nodes):
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
        raise AssertionError(f"Simulation {stem} produced no variant records.")

    vcf_path = directory / f"{stem}.vcf"
    vcf_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return vcf_path, reference_path


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.name)
def test_native_structured_fixed_point_matches_upstream(
    scenario: StructuredScenario,
    tmp_path: Path,
) -> None:
    ts = _simulate(scenario)
    vcf_path, reference_path = _write_vcf_and_reference(ts, tmp_path, scenario.name)
    root = EXAMPLES / scenario.example_dir
    data_kwargs = {
        "sequences": vcf_path,
        "param_file": root / "mutRec.param",
        "demo_file": root / scenario.demo_file,
        "config_file": root / scenario.config_file,
        "reference_file": reference_path,
        "filter_pass_string": "PASS",
    }
    common = {
        "n_em_iterations": 0,
        "start_point": scenario.start_point,
        "seed": scenario.seed,
        "loci_per_hmm_step": 50,
        "composite_mode": "lol",
    }
    options = {
        "interval_type": "logUniform",
        "interval_params": "8,0.01,4",
    }
    upstream_options = dict(options)
    if scenario.demography == "introgression":
        # diCal2's ODE core deliberately rejects pulse migration.
        upstream_options["cli_args"] = ["--useEigenCore"]
    upstream = dical2(
        read_dical2(**data_kwargs),
        implementation="upstream",
        upstream_options=upstream_options,
        **common,
    ).results["dical2"]
    native = dical2(
        read_dical2(**data_kwargs),
        implementation="native",
        native_options=options,
        **common,
    ).results["dical2"]

    assert np.isfinite(upstream["log_likelihood"])
    assert np.isfinite(native["log_likelihood"])
    assert native["log_likelihood"] == pytest.approx(
        upstream["log_likelihood"],
        abs=STRUCTURED_LL_ABS_TOL,
    )
    assert abs(native["log_likelihood"] - upstream["log_likelihood"]) / SEQUENCE_LENGTH < 1e-6
    np.testing.assert_allclose(
        np.asarray(native["best_params"]),
        np.asarray(upstream["best_params"]),
        rtol=0.0,
        atol=1e-14,
    )


def test_native_introgression_one_step_matches_upstream(tmp_path: Path) -> None:
    scenario = next(item for item in SCENARIOS if item.name == "introgression")
    ts = _simulate(scenario)
    vcf_path, reference_path = _write_vcf_and_reference(ts, tmp_path, scenario.name)
    root = EXAMPLES / scenario.example_dir
    data_kwargs = {
        "sequences": vcf_path,
        "param_file": root / "mutRec.param",
        "demo_file": root / scenario.demo_file,
        "config_file": root / scenario.config_file,
        "reference_file": reference_path,
        "filter_pass_string": "PASS",
    }
    common = {
        "n_em_iterations": 1,
        "start_point": scenario.start_point,
        "seed": scenario.seed,
        "loci_per_hmm_step": 50,
        "composite_mode": "lol",
    }
    options = {
        "interval_type": "logUniform",
        "interval_params": "8,0.01,4",
    }
    upstream = dical2(
        read_dical2(**data_kwargs),
        implementation="upstream",
        upstream_options={**options, "cli_args": ["--useEigenCore"]},
        **common,
    ).results["dical2"]
    native = dical2(
        read_dical2(**data_kwargs),
        implementation="native",
        native_options=options,
        **common,
    ).results["dical2"]

    assert upstream["resolved_options"]["number_iterations_mstep"] == 1
    assert native["resolved_options"]["number_iterations_mstep"] == 1
    assert native["core_type"] == "eigen"
    assert native["log_likelihood"] == pytest.approx(
        upstream["log_likelihood"],
        abs=STRUCTURED_LL_ABS_TOL,
    )
    np.testing.assert_allclose(
        np.asarray(native["best_params"]),
        np.asarray(upstream["best_params"]),
        rtol=0.0,
        atol=1e-14,
    )
