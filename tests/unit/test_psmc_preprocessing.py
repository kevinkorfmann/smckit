"""Tests for production PSMC consensus preprocessing."""

from __future__ import annotations

import gzip

import numpy as np
import pytest

from smckit.io import read_psmcfa, write_psmcfa
from smckit.pp import psmcfa_from_consensus


def test_consensus_fastq_quality_missingness_and_multirecord_output(tmp_path):
    fastq = tmp_path / "diploid.fq"
    fastq.write_text(
        "@chr1\nAAAAARAAAANNNNNNNNNN\n+\nIIIIIIIIIIIIIIIIIIII\n@chr2\nAAAAARAAAA\n+\nIIIII!IIII\n",
        encoding="utf-8",
    )
    output = tmp_path / "result.psmcfa.gz"

    data = psmcfa_from_consensus(
        fastq,
        output_path=output,
        block_size=10,
        min_good_bases=1,
        min_quality=10,
    )

    assert [record["name"] for record in data.uns["records"]] == ["chr1", "chr2"]
    np.testing.assert_array_equal(data.uns["records"][0]["codes"], [1, 2])
    np.testing.assert_array_equal(data.uns["records"][1]["codes"], [0])
    assert data.uns["sum_L"] == 2
    assert data.uns["sum_n"] == 1
    round_trip = read_psmcfa(output)
    np.testing.assert_array_equal(round_trip.uns["records"][0]["codes"], [1, 2])


@pytest.mark.parametrize(
    ("mutation_filter", "expected"),
    [
        (None, [1, 1, 1]),
        ("transversions", [0, 1, 0]),
        ("transitions", [1, 0, 0]),
        ("cpg", [0, 0, 1]),
        ("exclude_cpg", [1, 1, 0]),
    ],
)
def test_consensus_mutation_class_filters(tmp_path, mutation_filter, expected):
    # R is an A/G transition, M is an A/C transversion, and CR represents
    # a CpG transition at the second base.
    fasta = tmp_path / "classes.fa"
    fasta.write_text(
        ">chr1\nARAAAAAAAAMAAAAAAAAACRAAAAAAAA\n",
        encoding="utf-8",
    )

    data = psmcfa_from_consensus(
        fasta,
        block_size=10,
        min_good_bases=1,
        mutation_filter=mutation_filter,
    )

    np.testing.assert_array_equal(data.uns["records"][0]["codes"], expected)


def test_consensus_custom_mask_uses_zero_based_half_open_coordinates(tmp_path):
    fasta = tmp_path / "masked.fa"
    fasta.write_text(">chr1\nAAAARAAAAA\n", encoding="utf-8")

    data = psmcfa_from_consensus(
        fasta,
        block_size=10,
        min_good_bases=1,
        masks={"chr1": [(4, 5)]},
    )

    np.testing.assert_array_equal(data.uns["records"][0]["codes"], [0])


def test_write_psmcfa_supports_gzip_and_validates_line_width(tmp_path):
    source = tmp_path / "source.psmcfa"
    source.write_text(">chr1\nTKNT\n", encoding="utf-8")
    data = read_psmcfa(source)
    output = tmp_path / "copy.psmcfa.gz"

    write_psmcfa(data, output, line_width=2)
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        assert handle.read() == ">chr1\nTK\nNT\n"
    with pytest.raises(ValueError, match="line_width"):
        write_psmcfa(data, tmp_path / "bad.psmcfa", line_width=0)


def test_consensus_rejects_invalid_filter_and_all_filtered_records(tmp_path):
    fasta = tmp_path / "empty.fa"
    fasta.write_text(">chr1\nNNNN\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mutation_filter"):
        psmcfa_from_consensus(fasta, mutation_filter="unknown")
    with pytest.raises(ValueError, match="passed the callable-base filters"):
        psmcfa_from_consensus(fasta, min_good_bases=1)
