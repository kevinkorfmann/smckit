"""Tests for native VCF-to-SMC++ preparation and multi-population I/O."""

from __future__ import annotations

import gzip

import pytest

from smckit.io import read_smcpp_input, write_smcpp_input
from smckit.pp import smcpp_from_vcf


def _write_vcf(path) -> None:
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=20>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n"
        "chr1\t2\t.\tA\tG\t.\tPASS\t.\tGT\t0|1\t0|0\n"
        "chr1\t5\t.\tC\tT\t.\tPASS\t.\tGT\t1|1\t1|1\n"
        "chr1\t10\t.\tG\tA\t.\tPASS\t.\tGT\t0|1\t.|1\n"
        "chr1\t12\t.\tA\tAT\t.\tPASS\t.\tGT\t0|1\t0|1\n",
        encoding="utf-8",
    )


def test_one_population_vcf_conversion_and_compression(tmp_path) -> None:
    vcf = tmp_path / "input.vcf"
    output = tmp_path / "output.smc.gz"
    _write_vcf(vcf)

    data = smcpp_from_vcf(
        vcf,
        output,
        contig="chr1",
        populations={"pop": ["s1", "s2"]},
        distinguished=[("s1", 0), ("s1", 1)],
    )

    assert output.is_file()
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        assert handle.readline().startswith("# SMC++ ")
    assert data.uns["n_populations"] == 1
    assert data.uns["n_undist"] == 2
    assert data.uns["total_sites"] == 20
    assert data.uns["preprocessing"]["source_sha256"]
    observations = data.uns["records"][0]["observations"]
    assert (1, 1, 0) in observations
    assert any(a == -1 for _, a, _ in observations)


def test_bed_mask_removes_variants_and_marks_missing_span(tmp_path) -> None:
    vcf = tmp_path / "input.vcf"
    mask = tmp_path / "mask.bed"
    output = tmp_path / "masked.smc"
    _write_vcf(vcf)
    mask.write_text("chr1\t4\t7\n", encoding="utf-8")

    data = smcpp_from_vcf(
        vcf,
        output,
        contig="chr1",
        populations={"pop": ["s1", "s2"]},
        mask_path=mask,
    )

    observations = data.uns["records"][0]["observations"]
    assert (3, -1, -1) in observations
    assert data.uns["preprocessing"]["mask_sha256"]


def test_missing_cutoff_marks_long_unobserved_segments(tmp_path) -> None:
    vcf = tmp_path / "input.vcf"
    output = tmp_path / "cutoff.smc"
    _write_vcf(vcf)

    data = smcpp_from_vcf(
        vcf,
        output,
        contig="chr1",
        populations={"pop": ["s1", "s2"]},
        missing_cutoff=3,
    )

    assert any(
        span > 3 and a == -1
        for span, a, _ in data.uns["records"][0]["observations"]
    )


def test_two_population_columns_round_trip_without_loss(tmp_path) -> None:
    vcf = tmp_path / "input.vcf"
    output = tmp_path / "two-pop.smc"
    roundtrip = tmp_path / "two-pop-roundtrip.smc.gz"
    _write_vcf(vcf)

    data = smcpp_from_vcf(
        vcf,
        output,
        contig="chr1",
        populations={"p1": ["s1"], "p2": ["s2"]},
        distinguished=[("s1", 0), ("s2", 0)],
    )

    assert data.uns["n_populations"] == 2
    assert data.uns["populations"] == ["p1", "p2"]
    assert data.uns["n_undist_by_population"] == [1, 1]
    assert data.uns["records"] == []
    assert all(len(populations) == 2 for _, populations in data.uns["joint_observations"])

    write_smcpp_input(data, roundtrip)
    reread = read_smcpp_input(roundtrip)
    assert reread.uns["joint_observations"] == data.uns["joint_observations"]
    assert reread.uns["smcpp_header"] == data.uns["smcpp_header"]


def test_vcf_preprocessing_validates_population_and_mask_controls(tmp_path) -> None:
    vcf = tmp_path / "input.vcf"
    mask = tmp_path / "mask.bed"
    _write_vcf(vcf)
    mask.touch()

    with pytest.raises(ValueError, match="mutually exclusive"):
        smcpp_from_vcf(
            vcf,
            tmp_path / "bad.smc",
            contig="chr1",
            populations={"pop": ["s1"]},
            mask_path=mask,
            missing_cutoff=10,
        )
    with pytest.raises(ValueError, match="absent from VCF"):
        smcpp_from_vcf(
            vcf,
            tmp_path / "bad.smc",
            contig="chr1",
            populations={"pop": ["missing"]},
        )


def test_smcpp_reader_rejects_malformed_population_rows(tmp_path) -> None:
    source = tmp_path / "malformed.smc"
    source.write_text(
        "# SMC++ {\"pids\": [\"p1\", \"p2\"], \"undist\": [[], []], \"dist\": [[], []]}\n"
        "10\t0\t0\t2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="population count"):
        read_smcpp_input(source)
