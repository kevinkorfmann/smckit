"""Integration coverage for the upstream-backed PSMC wrapper."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

import smckit
from smckit.io import read_psmcfa
from smckit.pp import psmcfa_from_consensus
from smckit.tl import psmc

pytestmark = pytest.mark.skipif(
    shutil.which("make") is None or shutil.which("cc") is None,
    reason="Upstream PSMC build toolchain is not available",
)


def test_psmc_upstream_backend_runs_end_to_end() -> None:
    smckit.upstream.bootstrap("psmc")

    data = read_psmcfa("tests/data/NA12878_chr22.psmcfa")
    result = psmc(
        data,
        pattern="1+1+1",
        n_iterations=1,
        max_t=5.0,
        tr_ratio=5.0,
        mu=1e-8,
        generation_time=1.0,
        implementation="upstream",
    )

    res = result.results["psmc"]
    assert res["implementation"] == "upstream"
    assert res["backend"] == "upstream"
    assert np.all(res["ne"] > 0)
    assert np.isfinite(res["theta"])
    assert np.isfinite(res["rho"])
    assert set(res["upstream"]) >= {"tool", "binary", "input_path", "effective_args"}


def test_psmc_upstream_preserves_decode_and_output_artifact(tmp_path) -> None:
    smckit.upstream.bootstrap("psmc")
    output = tmp_path / "upstream.psmc"

    res = psmc(
        read_psmcfa("tests/data/NA12878_chr22.psmcfa"),
        pattern="1+1+1",
        n_iterations=0,
        max_t=5.0,
        decode="posterior",
        transition_cap=1,
        output_path=output,
        implementation="upstream",
    ).results["psmc"]

    assert output.is_file()
    assert res["decode"]["mode"] == "posterior"
    assert res["decode"]["segments"]
    assert res["provenance"]["artifacts"][0]["sha256"]


def test_psmc_native_full_decode_matches_upstream_on_missing_multirecord_fixture(
    tmp_path,
) -> None:
    smckit.upstream.bootstrap("psmc")
    fixture = tmp_path / "multi.psmcfa"
    fixture.write_text(
        ">chr1\nTTKNTTTKTTTT\n>chr2\nTTTTKNTTT\n",
        encoding="utf-8",
    )
    common = {
        "pattern": "1+1+1",
        "n_iterations": 0,
        "max_t": 5.0,
        "tr_ratio": 5.0,
        "random_init": 0.0,
        "decode": "full",
    }

    native = psmc(
        read_psmcfa(fixture),
        implementation="native",
        **common,
    ).results["psmc"]
    upstream = psmc(
        read_psmcfa(fixture),
        implementation="upstream",
        **common,
    ).results["psmc"]

    assert native["theta"] == pytest.approx(upstream["theta"], rel=2e-6)
    assert native["rho"] == pytest.approx(upstream["rho"], rel=2e-6)
    np.testing.assert_allclose(native["time"], upstream["time"], rtol=5e-7)
    np.testing.assert_allclose(native["lambda"], upstream["lambda"], rtol=1e-12)
    native_rows = [
        row
        for record in native["decode"]["records"]
        for row in zip(
            record["recombination_probability"],
            record["posterior"],
            strict=True,
        )
    ]
    upstream_rows = upstream["decode"]["rows"]
    assert len(native_rows) == len(upstream_rows)
    np.testing.assert_allclose(
        [row[0] for row in native_rows],
        [row["recombination_probability"] for row in upstream_rows],
        atol=5e-7,
    )
    np.testing.assert_allclose(
        [row[1] for row in native_rows],
        [row["posterior"] for row in upstream_rows],
        atol=5e-5,
    )


@pytest.mark.parametrize(
    ("mutation_filter", "flag"),
    [
        (None, None),
        ("transversions", "-v"),
        ("transitions", "-n"),
        ("cpg", "-c"),
        ("exclude_cpg", "-C"),
    ],
)
def test_native_consensus_preprocessing_matches_original_fq2psmcfa(
    tmp_path,
    mutation_filter,
    flag,
) -> None:
    smckit.upstream.bootstrap("psmc")
    fixture = tmp_path / "diploid.fq"
    fixture.write_text(
        "@chr1\nARAAAAAAAAMAAAAAAAAACRAAAAAAAA\n+\nIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n",
        encoding="utf-8",
    )
    args = ["-g", "1", "-s", "10", "-q", "10"]
    if flag is not None:
        args.append(flag)
    args.append(str(fixture))
    original = smckit.upstream.run(
        "psmc",
        args,
        entrypoint="fq2psmcfa",
        output_dir=tmp_path / "upstream-run",
    )
    assert original.returncode == 0, original.stderr
    original_path = tmp_path / "original.psmcfa"
    original_path.write_text(original.stdout, encoding="utf-8")

    native = psmcfa_from_consensus(
        fixture,
        min_good_bases=1,
        min_quality=10,
        block_size=10,
        mutation_filter=mutation_filter,
    )
    upstream = read_psmcfa(original_path)
    np.testing.assert_array_equal(
        native.uns["records"][0]["codes"],
        upstream.uns["records"][0]["codes"],
    )


def test_psmc_upstream_preserves_probability_and_simulation_records(tmp_path) -> None:
    smckit.upstream.bootstrap("psmc")
    fixture = tmp_path / "input.psmcfa"
    fixture.write_text(">chr1\nTTKNTTTKTTTT\n", encoding="utf-8")

    result = psmc(
        read_psmcfa(fixture),
        pattern="1+1+1",
        n_iterations=0,
        random_init=0,
        sequence_probability=True,
        simulate=True,
        implementation="upstream",
    ).results["psmc"]

    assert len(result["sequence_probabilities"][0]["scale"]) == 12
    assert result["simulated_records"][0]["name"] == "chr1"
    assert len(result["simulated_records"][0]["sequence"]) == 12
    assert result["simulated_records"][0]["sequence"][3] == "N"
