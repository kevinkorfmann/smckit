"""Tests for native, callability-aware multihetsep preparation."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from smckit.pp import multihetsep_from_vcf


def _write_vcf(
    path: Path,
    rows: list[tuple[int, str, str, str]],
    *,
    sample_names: tuple[str, ...] = ("sample",),
    contig: str = "chr1",
) -> None:
    lines = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID={contig},length=1000000000>",
        "\t".join(
            [
                "#CHROM",
                "POS",
                "ID",
                "REF",
                "ALT",
                "QUAL",
                "FILTER",
                "INFO",
                "FORMAT",
                *sample_names,
            ]
        ),
    ]
    for position, reference, alternate, genotype_fields in rows:
        lines.append(
            "\t".join(
                [
                    contig,
                    str(position),
                    ".",
                    reference,
                    alternate,
                    ".",
                    "PASS",
                    ".",
                    "DP:GT",
                    *genotype_fields.split(","),
                ]
            )
        )
    text = "\n".join(lines) + "\n"
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        path.write_text(text, encoding="utf-8")


def test_callability_mask_is_required_by_default(tmp_path: Path) -> None:
    vcf = tmp_path / "sample.vcf"
    _write_vcf(vcf, [(2, "A", "G", "8:0|1")])

    with pytest.raises(ValueError, match="variant-only VCFs"):
        multihetsep_from_vcf(vcf, tmp_path / "sample.mhs")


def test_pinned_msmc_tools_semantics_and_provenance(tmp_path: Path) -> None:
    vcf = tmp_path / "sample.vcf.gz"
    mask = tmp_path / "callable.bed"
    output = tmp_path / "sample.mhs"
    _write_vcf(
        vcf,
        [
            (2, "A", "G", "8:0|1"),
            (5, "C", "T", "8:0/1"),
            (7, "A", ".", "8:0|0"),
            (9, "G", "A", "8:0/1"),
            (12, "T", "C", "8:1|1"),
        ],
    )
    mask.write_text("chr1\t0\t4\nchr1\t6\t15\n", encoding="utf-8")

    data = multihetsep_from_vcf(vcf, output, mask_paths=[mask])

    assert output.read_text() == "chr1\t2\t2\tAG\nchr1\t9\t5\tAG,GA\n"
    preprocessing = data.uns["preprocessing"]
    assert preprocessing["variant_positions_read"] == 5
    assert preprocessing["segregating_sites_emitted"] == 2
    assert preprocessing["callable_bases_through_last_site"] == 7
    assert preprocessing["output_sha256"]
    oracle = preprocessing["compatibility_oracle"]
    assert oracle["commit"] == "4d1f05f39f7b4f8c205e602c180b44a7c68a7bba"
    assert oracle["entrypoint_sha256"] == (
        "caaa87a07e0fe2dc7228f30c9aff759cf86e9f61b8332aabd41398399ea6331b"
    )
    assert oracle["redistributed"] is False


def test_positive_masks_intersect_and_negative_masks_subtract(tmp_path: Path) -> None:
    vcf = tmp_path / "sample.vcf"
    first = tmp_path / "first.bed"
    second = tmp_path / "second.mask"
    negative = tmp_path / "negative.bed"
    output = tmp_path / "masked.mhs"
    _write_vcf(vcf, [(3, "A", "G", "8:0|1"), (7, "C", "T", "8:0|1")])
    first.write_text("chr1\t0\t10\n", encoding="utf-8")
    second.write_text("2\t8\n", encoding="utf-8")
    negative.write_text("chr1\t4\t6\n", encoding="utf-8")

    multihetsep_from_vcf(
        vcf,
        output,
        mask_paths=[first, second],
        negative_mask_paths=[negative],
    )

    assert output.read_text() == "chr1\t3\t2\tAG\nchr1\t7\t2\tCT\n"


def test_large_callable_gap_does_not_require_per_base_expansion(tmp_path: Path) -> None:
    vcf = tmp_path / "large.vcf"
    output = tmp_path / "large.mhs"
    _write_vcf(
        vcf,
        [(2, "A", "G", "8:0|1"), (1_000_000_000, "C", "T", "8:0|1")],
    )

    multihetsep_from_vcf(
        vcf,
        output,
        assume_all_sites_callable=True,
        contig_length=1_000_000_000,
    )

    assert output.read_text() == ("chr1\t2\t2\tAG\nchr1\t1000000000\t999999998\tCT\n")


def test_multiple_single_sample_vcfs_are_stream_joined(tmp_path: Path) -> None:
    first = tmp_path / "first.vcf"
    second = tmp_path / "second.vcf.gz"
    mask = tmp_path / "mask.bed"
    output = tmp_path / "joined.mhs"
    _write_vcf(first, [(2, "A", "G", "8:0|1")], sample_names=("first",))
    _write_vcf(second, [(4, "C", "T", "8:0|1")], sample_names=("second",))
    mask.write_text("chr1\t0\t10\n", encoding="utf-8")

    data = multihetsep_from_vcf([first, second], output, mask_paths=[mask])

    assert output.read_text() == "chr1\t2\t2\tAGAA\nchr1\t4\t2\tCCCT\n"
    assert data.uns["n_haplotypes"] == 4
    assert data.uns["preprocessing"]["samples"] == ["first", "second"]


def test_trio_phasing_removes_child_haplotypes(tmp_path: Path) -> None:
    paths = [tmp_path / f"{name}.vcf" for name in ("child", "father", "mother")]
    mask = tmp_path / "mask.bed"
    output = tmp_path / "trio.mhs"
    _write_vcf(paths[0], [(2, "A", "G", "8:0/1")], sample_names=("child",))
    _write_vcf(paths[1], [(2, "A", "G", "8:0|0")], sample_names=("father",))
    _write_vcf(paths[2], [(2, "A", "G", "8:1|1")], sample_names=("mother",))
    mask.write_text("chr1\t0\t10\n", encoding="utf-8")

    data = multihetsep_from_vcf(
        paths,
        output,
        mask_paths=[mask],
        trios=[(0, 1, 2)],
    )

    assert output.read_text() == "chr1\t2\t2\tAAGG\n"
    assert data.uns["n_haplotypes"] == 4


def test_multisample_vcf_requires_and_records_selection(tmp_path: Path) -> None:
    vcf = tmp_path / "cohort.vcf"
    mask = tmp_path / "mask.bed"
    _write_vcf(
        vcf,
        [(2, "A", "G", "8:0|0,8:0|1")],
        sample_names=("reference", "selected"),
    )
    mask.write_text("chr1\t0\t10\n", encoding="utf-8")

    with pytest.raises(ValueError, match="select exactly one"):
        multihetsep_from_vcf(vcf, tmp_path / "missing-selection.mhs", mask_paths=[mask])

    data = multihetsep_from_vcf(
        vcf,
        tmp_path / "selected.mhs",
        mask_paths=[mask],
        samples="selected",
    )
    assert data.uns["preprocessing"]["samples"] == ["selected"]


@pytest.mark.parametrize(
    ("reference", "alternate", "genotype", "message"),
    [
        ("A", "AT", "8:0|1", "requires A/C/G/T SNVs"),
        ("A", "G", "8:.|1", "missing GT alleles"),
        ("A", "G", "8:0|2", "unavailable allele"),
    ],
)
def test_invalid_called_site_vcf_is_rejected(
    tmp_path: Path,
    reference: str,
    alternate: str,
    genotype: str,
    message: str,
) -> None:
    vcf = tmp_path / "invalid.vcf"
    mask = tmp_path / "mask.bed"
    _write_vcf(vcf, [(2, reference, alternate, genotype)])
    mask.write_text("chr1\t0\t10\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        multihetsep_from_vcf(vcf, tmp_path / "invalid.mhs", mask_paths=[mask])


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    vcf = tmp_path / "sample.vcf"
    mask = tmp_path / "mask.bed"
    output = tmp_path / "existing.mhs"
    _write_vcf(vcf, [(2, "A", "G", "8:0|1")])
    mask.write_text("chr1\t0\t10\n", encoding="utf-8")
    output.write_text("keep me\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        multihetsep_from_vcf(vcf, output, mask_paths=[mask])

    assert output.read_text() == "keep me\n"


def test_contig_length_cannot_silently_drop_records(tmp_path: Path) -> None:
    vcf = tmp_path / "sample.vcf"
    _write_vcf(vcf, [(11, "A", "G", "8:0|1")])

    with pytest.raises(ValueError, match="exceeds contig_length"):
        multihetsep_from_vcf(
            vcf,
            tmp_path / "outside.mhs",
            assume_all_sites_callable=True,
            contig_length=10,
        )
