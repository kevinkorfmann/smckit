"""Independent simulated structured-demography oracles for native diCal2."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import smckit
from smckit.io import read_dical2
from smckit.tl import dical2
from smckit.tl._dical2 import _dical2_upstream, _resolve_dical2_options

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
    bounds: str
    demography: str
    seed: int


SCENARIOS = [
    StructuredScenario(
        name="clean_split",
        example_dir="cleanSplit",
        demo_file="clean_split.demo",
        config_file="clean_split.config",
        start_point=(0.2, 0.25, 0.25, 1.0),
        bounds="0.002,20;0.01,20;0.01,20;0.01,20",
        demography="split",
        seed=101,
    ),
    StructuredScenario(
        name="migration_window",
        example_dir="islolationMigrationWindow",
        demo_file="isolation_migration_window.demo",
        config_file="isolation_migration_window.config",
        start_point=(0.1, 0.2, 0.25, 0.25, 0.1, 1.0),
        bounds="0.02,20;0.002,20;0.01,20;0.01,20;0.01,100;0.01,20",
        demography="migration_window",
        seed=103,
    ),
    StructuredScenario(
        name="three_populations",
        example_dir="threePopulations",
        demo_file="three_populations.demo",
        config_file="three_populations.config",
        start_point=(0.2, 0.4),
        bounds="0.02,20;0.02,20",
        demography="three_populations",
        seed=107,
    ),
    StructuredScenario(
        name="introgression",
        example_dir="introgression",
        demo_file="introgression.demo",
        config_file="introgression.config",
        start_point=(0.05, 0.03),
        bounds="0.001,20;0.0001,0.9999",
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


def _simulate_exponential_growth():
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
        random_seed=211,
    )
    return msprime.sim_mutations(
        ancestry,
        rate=2e-7,
        model=msprime.BinaryMutationModel(),
        discrete_genome=True,
        random_seed=212,
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


def test_native_introgression_ancient_states_with_trunk_refinement_match_upstream(
    tmp_path: Path,
) -> None:
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
        "n_em_iterations": 0,
        "start_point": scenario.start_point,
        "seed": scenario.seed,
        "loci_per_hmm_step": 50,
        "composite_mode": "lol",
    }
    options = {
        "ancient_deme_states": True,
        "add_trunk_intervals": 2,
        "trunk_style": "migratingEthan",
        "cake_style": "average",
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

    assert native["core_type"] == "eigen"
    assert native["log_likelihood"] == pytest.approx(
        upstream["log_likelihood"], abs=STRUCTURED_LL_ABS_TOL
    )


@pytest.mark.parametrize(
    "scenario",
    [item for item in SCENARIOS if item.demography != "introgression"],
    ids=lambda scenario: scenario.name,
)
def test_native_structured_one_step_matches_upstream(
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
        "n_em_iterations": 1,
        "start_point": scenario.start_point,
        "seed": scenario.seed,
        "loci_per_hmm_step": 50,
        "composite_mode": "lol",
        "bounds": scenario.bounds,
    }
    options = {
        "interval_type": "logUniform",
        "interval_params": "8,0.01,4",
    }
    upstream = dical2(
        read_dical2(**data_kwargs),
        implementation="upstream",
        upstream_options=options,
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
    assert native["core_type"] == "ode"
    assert native["log_likelihood"] == pytest.approx(
        upstream["log_likelihood"],
        abs=STRUCTURED_LL_ABS_TOL,
    ), (native["best_params"], upstream["best_params"])
    np.testing.assert_allclose(
        np.asarray(native["best_params"]),
        np.asarray(upstream["best_params"]),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "scenario",
    SCENARIOS[:3],
    ids=lambda scenario: scenario.name,
)
def test_native_structured_pac_one_step_matches_upstream(
    scenario: StructuredScenario,
    tmp_path: Path,
) -> None:
    ts = _simulate(scenario)
    vcf_path, reference_path = _write_vcf_and_reference(
        ts,
        tmp_path,
        f"{scenario.name}-pac",
    )
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
        "composite_mode": "pac",
        "bounds": scenario.bounds,
    }
    options = {
        "interval_type": "logUniform",
        "interval_params": "8,0.01,4",
        "number_iterations_mstep": 1,
        "num_permutations": 2,
        "num_csds_per_permutation": 2,
    }
    upstream = dical2(
        read_dical2(**data_kwargs),
        implementation="upstream",
        upstream_options=options,
        **common,
    ).results["dical2"]
    native = dical2(
        read_dical2(**data_kwargs),
        implementation="native",
        native_options=options,
        **common,
    ).results["dical2"]

    permutations = native["permutations"]["per_contig"][0]
    assert len(permutations) == 2
    for permutation in permutations:
        assert sorted(permutation) == list(range(len(permutation)))
        row = "\t".join(str(value) for value in permutation)
        assert f"# HAP PERMUTATION:\t{row}\t" in upstream["upstream"]["stdout"]
    np.testing.assert_allclose(
        np.asarray(native["best_params"]),
        np.asarray(upstream["best_params"]),
        rtol=0.0,
        atol=1e-12,
    )
    assert native["log_likelihood"] == pytest.approx(
        upstream["log_likelihood"],
        abs=STRUCTURED_LL_ABS_TOL,
    )


def test_native_clean_split_file_per_contig_pac_one_step_matches_upstream(
    tmp_path: Path,
) -> None:
    scenario = SCENARIOS[0]
    ts = _simulate(scenario)
    vcf_path, reference_path = _write_vcf_and_reference(
        ts,
        tmp_path,
        "clean-split-file-pac",
    )
    permutation_files = [tmp_path / "chunk-1.perm", tmp_path / "chunk-2.perm"]
    permutation_files[0].write_text(
        "0 1 2 3\n3 2 1 0\n",
        encoding="utf-8",
    )
    permutation_files[1].write_text(
        "1 0 3 2\n2 3 0 1\n",
        encoding="utf-8",
    )
    root = EXAMPLES / scenario.example_dir
    data_kwargs = {
        "sequences": [vcf_path, vcf_path],
        "param_file": root / "mutRec.param",
        "demo_file": root / scenario.demo_file,
        "config_file": root / scenario.config_file,
        "reference_file": [reference_path, reference_path],
        "filter_pass_string": "PASS",
    }
    common = {
        "n_em_iterations": 1,
        "start_point": scenario.start_point,
        "seed": scenario.seed,
        "loci_per_hmm_step": 50,
        "composite_mode": "pac",
        "bounds": scenario.bounds,
    }
    options = {
        "interval_type": "logUniform",
        "interval_params": "8,0.01,4",
        "number_iterations_mstep": 1,
        "permutation_files": permutation_files,
        "different_permutations_per_contig": True,
        "num_csds_per_permutation": 2,
    }
    upstream = dical2(
        read_dical2(**data_kwargs),
        implementation="upstream",
        upstream_options=options,
        **common,
    ).results["dical2"]
    native = dical2(
        read_dical2(**data_kwargs),
        implementation="native",
        native_options=options,
        **common,
    ).results["dical2"]

    assert native["permutations"]["per_contig"] == [
        [[0, 1, 2, 3], [3, 2, 1, 0]],
        [[1, 0, 3, 2], [2, 3, 0, 1]],
    ]
    assert len(native["permutations"]["files"]) == 2
    np.testing.assert_allclose(
        np.asarray(native["best_params"]),
        np.asarray(upstream["best_params"]),
        rtol=0.0,
        atol=1e-12,
    )
    assert native["log_likelihood"] == pytest.approx(
        upstream["log_likelihood"],
        abs=STRUCTURED_LL_ABS_TOL,
    )


@pytest.mark.parametrize(
    ("objective_options", "objective_mode", "upstream_failure"),
    [
        (
            {"condOnTransitionType": True},
            "condition_lineage_transition_type",
            False,
        ),
        ({"marginalKL": True}, "marginal_kl", True),
    ],
)
def test_native_structured_objective_matches_upstream(
    objective_options: dict[str, bool],
    objective_mode: str,
    upstream_failure: bool,
    tmp_path: Path,
) -> None:
    scenario = SCENARIOS[0]
    ts = _simulate(scenario)
    vcf_path, reference_path = _write_vcf_and_reference(
        ts,
        tmp_path,
        f"{scenario.name}-{objective_mode}",
    )
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
        "bounds": scenario.bounds,
    }
    options = {
        "interval_type": "logUniform",
        "interval_params": "8,0.01,4",
        **objective_options,
    }
    if upstream_failure:
        with pytest.raises(RuntimeError, match="mutationRate.*null"):
            dical2(
                read_dical2(**data_kwargs),
                implementation="upstream",
                upstream_options=options,
                **common,
            )
        upstream = None
    else:
        upstream = dical2(
            read_dical2(**data_kwargs),
            implementation="upstream",
            upstream_options=options,
            **common,
        ).results["dical2"]
    native = dical2(
        read_dical2(**data_kwargs),
        implementation="native",
        native_options=options,
        **common,
    ).results["dical2"]

    assert native["resolved_options"]["objective_mode"] == objective_mode
    assert native["objective_mode"] == objective_mode
    if upstream is None:
        np.testing.assert_allclose(
            np.asarray(native["best_params"]),
            np.array([0.32, 0.35, 0.2, 0.6]),
            rtol=0.0,
            atol=1e-12,
        )
        assert native["log_likelihood"] == pytest.approx(
            -59.781214042356474,
            abs=1e-10,
        )
        return
    assert upstream["resolved_options"]["objective_mode"] == objective_mode
    assert native["log_likelihood"] == pytest.approx(
        upstream["log_likelihood"],
        abs=STRUCTURED_LL_ABS_TOL,
    ), (native["best_params"], upstream["best_params"])
    np.testing.assert_allclose(
        np.asarray(native["best_params"]),
        np.asarray(upstream["best_params"]),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("composite_mode", "composite_options"),
    [
        ("lol", {}),
        ("pac", {"num_permutations": 2, "num_csds_per_permutation": 2}),
    ],
)
def test_native_marginal_kl_matches_repaired_source_oracle(
    tmp_path: Path,
    composite_mode: str,
    composite_options: dict[str, int],
) -> None:
    scenario = SCENARIOS[0]
    ts = _simulate(scenario)
    vcf_path, reference_path = _write_vcf_and_reference(
        ts,
        tmp_path,
        f"{scenario.name}-marginal-kl-repaired-{composite_mode}",
    )
    root = EXAMPLES / scenario.example_dir
    data_kwargs = {
        "sequences": vcf_path,
        "param_file": root / "mutRec.param",
        "demo_file": root / scenario.demo_file,
        "config_file": root / scenario.config_file,
        "reference_file": reference_path,
        "filter_pass_string": "PASS",
    }
    options = {
        "interval_type": "logUniform",
        "interval_params": "8,0.01,4",
        "marginalKL": True,
        **composite_options,
    }
    resolved = _resolve_dical2_options(
        n_intervals=11,
        max_t=4.0,
        alpha=0.1,
        n_em_iterations=1,
        composite_mode=composite_mode,
        loci_per_hmm_step=50,
        start_point=scenario.start_point,
        meta_start_file=None,
        meta_num_iterations=1,
        meta_keep_best=1,
        meta_num_points=None,
        bounds=scenario.bounds,
        seed=scenario.seed,
        method_options=options,
    )
    repaired_jar = tmp_path / "dical2-marginal-kl-repaired.jar"
    build_process = subprocess.run(
        [
            sys.executable,
            "scripts/build_dical2_repaired_oracle.py",
            "--output",
            str(repaired_jar),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if build_process.returncode != 0:
        if "Could not locate the JDK" in build_process.stderr:
            pytest.skip(build_process.stderr)
        pytest.fail(build_process.stderr)
    repair_metadata = json.loads(build_process.stdout)

    repaired = _dical2_upstream(
        read_dical2(**data_kwargs),
        resolved=resolved,
        cli_args=[],
        implementation_requested="upstream",
        jar_override=repaired_jar,
        oracle_repair=repair_metadata,
    ).results["dical2"]
    native = dical2(
        read_dical2(**data_kwargs),
        implementation="native",
        n_em_iterations=1,
        start_point=scenario.start_point,
        seed=scenario.seed,
        loci_per_hmm_step=50,
        composite_mode=composite_mode,
        bounds=scenario.bounds,
        native_options=options,
    ).results["dical2"]

    assert repaired["backend"] == "upstream_repaired_oracle"
    assert (
        repaired["upstream"]["oracle_repair"]["output_sha256"] == repair_metadata["output_sha256"]
    )
    np.testing.assert_allclose(
        np.asarray(native["best_params"]),
        np.asarray(repaired["best_params"]),
        rtol=0.0,
        atol=1e-12,
    )
    assert native["log_likelihood"] == pytest.approx(
        repaired["log_likelihood"],
        abs=STRUCTURED_LL_ABS_TOL,
    )


def test_native_exponential_growth_one_step_matches_upstream(tmp_path: Path) -> None:
    ts = _simulate_exponential_growth()
    vcf_path, reference_path = _write_vcf_and_reference(
        ts,
        tmp_path,
        "exponential_growth",
    )
    root = EXAMPLES / "expGrowth"
    data_kwargs = {
        "sequences": vcf_path,
        "param_file": root / "mutRec.param",
        "demo_file": root / "exp_growth.demo",
        "rates_file": root / "exp_growth.rates",
        "config_file": root / "exp_growth.config",
        "reference_file": reference_path,
        "filter_pass_string": "PASS",
    }
    common = {
        "n_em_iterations": 1,
        "start_point": (0.8, 2.0),
        "seed": 211,
        "loci_per_hmm_step": 50,
        "composite_mode": "pcl",
        "bounds": "0.01,20;0.05,50",
    }
    options = {
        "interval_type": "logUniform",
        "interval_params": "8,0.01,4",
    }
    upstream = dical2(
        read_dical2(**data_kwargs),
        implementation="upstream",
        upstream_options=options,
        **common,
    ).results["dical2"]
    native = dical2(
        read_dical2(**data_kwargs),
        implementation="native",
        native_options=options,
        **common,
    ).results["dical2"]

    assert native["core_type"] == "ode"
    assert native["log_likelihood"] == pytest.approx(
        upstream["log_likelihood"],
        abs=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(native["best_params"]),
        np.asarray(upstream["best_params"]),
        rtol=0.0,
        atol=1e-14,
    )
