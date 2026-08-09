"""Native PSMC+ preprocessing against frozen upstream outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smckit.backends._psmcplus_preprocessing import prepare_psmcplus_sequence

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/data/psmcplus"
ORACLE_PATH = FIXTURES / "preprocessing_oracle_v1.npz"
UPSTREAM_COMMIT = "032168f2ceed3c0e46b7f214f890faf83dff41ae"


def _prepare():
    return prepare_psmcplus_sequence(
        FIXTURES / "preprocessing_masked.mhs",
        bin_size=10,
        mutation_map_path=FIXTURES / "preprocessing_mutation.bed",
        recombination_map_path=FIXTURES / "preprocessing_recombination.bed",
    )


def test_native_preprocessing_matches_frozen_upstream() -> None:
    sequence = _prepare()
    with np.load(ORACLE_PATH) as oracle:
        assert str(oracle["oracle_commit"]) == UPSTREAM_COMMIT
        assert sequence.sequence_length == int(oracle["sequence_length"])
        assert sequence.number_heterozygotes == int(oracle["number_heterozygotes"])
        assert sequence.number_masked_bases == int(oracle["number_masked_bases"])
        assert sequence.maximum_heterozygotes == int(oracle["maximum_heterozygotes"])
        for name in (
            "heterozygotes",
            "masked_bases",
            "mutation_indices",
            "mutation_factors",
            "recombination_indices",
            "recombination_factors",
        ):
            np.testing.assert_array_equal(getattr(sequence, name), oracle[name])
        np.testing.assert_array_equal(
            sequence.mutation_factor_sequence(), oracle["mutation_factor_sequence"]
        )
        np.testing.assert_array_equal(
            sequence.recombination_factor_sequence(), oracle["recombination_factor_sequence"]
        )


def test_default_rate_maps_are_constant_one() -> None:
    sequence = prepare_psmcplus_sequence(
        FIXTURES / "preprocessing_masked.mhs",
        bin_size=10,
    )
    np.testing.assert_array_equal(sequence.mutation_indices, np.zeros(5, dtype=np.int64))
    np.testing.assert_array_equal(sequence.recombination_indices, np.zeros(5, dtype=np.int64))
    np.testing.assert_array_equal(sequence.mutation_factors, np.ones(1))
    np.testing.assert_array_equal(sequence.recombination_factors, np.ones(1))


def test_short_map_is_padded_and_variable_recombination_is_truncated(tmp_path: Path) -> None:
    rate_map = tmp_path / "short.bed"
    rate_map.write_text("chr1\t0\t15\t0.019\n", encoding="utf-8")
    sequence = prepare_psmcplus_sequence(
        FIXTURES / "preprocessing_masked.mhs",
        bin_size=10,
        recombination_map_path=rate_map,
    )
    np.testing.assert_array_equal(
        sequence.recombination_factor_sequence(),
        np.array([0.01, 0.5, 1.0, 1.0, 1.0]),
    )


def test_rate_map_binning_preserves_upstream_float_reduction_order(tmp_path: Path) -> None:
    multihetsep = tmp_path / "long.mhs"
    multihetsep.write_text("chr1\t25\t25\tAT\nchr1\t110\t85\tAG\n", encoding="utf-8")
    rate_map = tmp_path / "rates.bed"
    rate_map.write_text(
        "chr1\t0\t50\t0.4\nchr1\t50\t100\t1.35\nchr1\t100\t210\t1.0\n",
        encoding="utf-8",
    )

    sequence = prepare_psmcplus_sequence(
        multihetsep,
        bin_size=50,
        recombination_map_path=rate_map,
    )

    # The preserved implementation expands each bin before averaging. Its
    # float64 reduction yields 0.399... and 1.349..., then truncates those
    # values to 0.39 and 1.34 rather than the algebraic 0.40 and 1.35.
    np.testing.assert_array_equal(
        sequence.recombination_factor_sequence(),
        np.array([0.39, 1.34, 1.0, 1.0]),
    )


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("chr1\t5\n", "three tab-separated"),
        ("chr1\t5\t3\tAT\nchr1\t5\t1\tAG\n", "strictly increasing"),
        ("chr1\t5\t6\tAT\n", "callable count"),
        ("chr1\t5\t0\tAT\n", "callable count"),
        ("chr1\t5\t3\tAT\nchr2\t12\t5\tAG\n", "multiple chromosomes"),
    ],
)
def test_malformed_multihetsep_is_rejected(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    path = tmp_path / "bad.mhs"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        prepare_psmcplus_sequence(path, bin_size=10)


def test_gapped_and_mismatched_rate_maps_are_rejected(tmp_path: Path) -> None:
    gapped = tmp_path / "gapped.bed"
    gapped.write_text("chr1\t0\t5\t1\nchr1\t6\t60\t1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous"):
        prepare_psmcplus_sequence(
            FIXTURES / "preprocessing_masked.mhs",
            bin_size=10,
            mutation_map_path=gapped,
        )

    mismatched = tmp_path / "mismatched.bed"
    mismatched.write_text("chr2\t0\t60\t1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        prepare_psmcplus_sequence(
            FIXTURES / "preprocessing_masked.mhs",
            bin_size=10,
            mutation_map_path=mismatched,
        )
