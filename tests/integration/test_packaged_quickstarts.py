"""Smoke tests for packaged quickstart fixtures on installed-style paths."""

from __future__ import annotations

from pathlib import Path

import pytest

import smckit
from smckit.tl._implementation import NativeTrustWarning


def test_packaged_quickstart_fixtures_resolve_from_isolated_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    psmc = smckit.io.example_path("psmc/NA12878_chr22.psmcfa")
    msmc2 = smckit.io.example_path("msmc2/msmc2_test.multihetsep")
    msmc_im = smckit.io.example_path("msmc_im/Yoruba_French.8haps.combined.msmc2.final.txt")
    dical2_vcf = smckit.io.example_path("dical2/test.vcf")
    dical2_param = smckit.io.example_path("dical2/test.param")
    dical2_demo = smckit.io.example_path("dical2/exp.demo")
    dical2_config = smckit.io.example_path("dical2/exp.config")
    dical2_ref = smckit.io.example_path("dical2/test.fa")
    asmc_root = smckit.io.example_prefix(
        "asmc/exampleFile.n300.array",
        (".hap.gz", ".samples", ".map.gz"),
    )
    asmc_dq = smckit.io.example_path("asmc/30-100-2000_CEU.decodingQuantities.gz")
    smcpp = smckit.io.example_path("smcpp/example.smc.gz")

    assert psmc.exists()
    assert msmc2.exists()
    assert msmc_im.exists()
    assert dical2_vcf.exists()
    assert dical2_param.exists()
    assert dical2_demo.exists()
    assert dical2_config.exists()
    assert dical2_ref.exists()
    assert Path(str(asmc_root) + ".hap.gz").exists()
    assert Path(str(asmc_root) + ".samples").exists()
    assert Path(str(asmc_root) + ".map.gz").exists()
    assert asmc_dq.exists()
    assert smcpp.exists()


def test_packaged_psmc_quickstart_runs_native(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    example = smckit.io.example_path("psmc/NA12878_chr22.psmcfa")
    data = smckit.io.read_psmcfa(example)
    data = smckit.tl.psmc(
        data,
        pattern="4+5*3+4",
        n_iterations=2,
        max_t=15.0,
        tr_ratio=4.0,
        mu=1.25e-8,
        generation_time=25.0,
        implementation="native",
        seed=1,
    )

    res = data.results["psmc"]
    assert res["implementation"] == "native"
    assert res["theta"] > 0
    assert res["rho"] > 0


def test_packaged_msmc2_quickstart_reads_example(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    example = smckit.io.example_path("msmc2/msmc2_test.multihetsep")
    data = smckit.io.read_multihetsep(example)
    data = smckit.tl.msmc2(
        data,
        time_pattern="1*2+25*1+1*2+1*3",
        n_iterations=1,
        mu=1.25e-8,
        generation_time=25.0,
        implementation="native",
    )

    assert data.results["msmc2"]["implementation"] == "native"


def test_packaged_msmc_im_quickstart_runs_native(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    example = smckit.io.example_path("msmc_im/Yoruba_French.8haps.combined.msmc2.final.txt")
    data = smckit.tl.msmc_im(
        example,
        pattern="1*2+25*1+1*2+1*3",
        mu=1.25e-8,
        beta=(1e-8, 1e-6),
        implementation="native",
    )

    res = data.results["msmc_im"]
    assert res["implementation"] == "native"
    assert "split_time_quantiles" in res


def test_packaged_asmc_quickstart_runs_native(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    root = smckit.io.example_prefix(
        "asmc/exampleFile.n300.array",
        (".hap.gz", ".samples", ".map.gz"),
    )
    dq = smckit.io.example_path("asmc/30-100-2000_CEU.decodingQuantities.gz")
    data = smckit.io.read_asmc(root, dq)
    data = smckit.tl.asmc(
        data,
        pairs=[(0, 1)],
        mode="array",
        implementation="native",
    )

    res = data.results["asmc"]
    assert res["implementation"] == "native"
    assert res["n_pairs_decoded"] == 1


def test_packaged_dical2_quickstart_reads_example_bundle(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    data = smckit.io.read_dical2(
        sequences=smckit.io.example_path("dical2/test.vcf"),
        param_file=smckit.io.example_path("dical2/test.param"),
        demo_file=smckit.io.example_path("dical2/exp.demo"),
        config_file=smckit.io.example_path("dical2/exp.config"),
        reference_file=smckit.io.example_path("dical2/test.fa"),
        filter_pass_string=".",
    )
    with pytest.warns(NativeTrustWarning):
        data = smckit.tl.dical2(
            data,
            n_intervals=11,
            max_t=4.0,
            n_em_iterations=1,
            composite_mode="pac",
            implementation="native",
        )

    assert data.results["dical2"]["implementation"] == "native"


def test_packaged_smcpp_quickstart_runs_native(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    example = smckit.io.example_path("smcpp/example.smc.gz")
    data = smckit.io.read_smcpp_input(example)
    data = smckit.tl.smcpp(
        data,
        n_intervals=4,
        max_iterations=1,
        regularization=10.0,
        mu=1.25e-8,
        generation_time=25.0,
        implementation="native",
        seed=42,
    )

    res = data.results["smcpp"]
    assert res["implementation"] == "native"
    assert len(res["ne"]) > 0


def test_packaged_esmc2_quickstart_runs_native(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    example = smckit.io.example_path("psmc/NA12878_chr22.psmcfa")
    data = smckit.io.read_psmcfa(example)
    with pytest.warns(NativeTrustWarning):
        data = smckit.tl.esmc2(
            data,
            n_states=20,
            n_iterations=1,
            estimate_beta=True,
            beta=0.8,
            mu=1.25e-8,
            generation_time=1.0,
            implementation="native",
        )

    res = data.results["esmc2"]
    assert res["implementation"] == "native"
    assert "ne" in res
