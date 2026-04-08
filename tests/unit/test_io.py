"""Tests for I/O modules."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from smckit.io._multihetsep import read_multihetsep
from smckit.io._psmcfa import _CONV_TABLE, read_psmcfa


class TestConvTable:
    def test_homozygous_chars(self):
        for c in b"0ACGTacgt":
            assert _CONV_TABLE[c] == 0

    def test_heterozygous_chars(self):
        for c in b"1KMRSWYkmrswy":
            assert _CONV_TABLE[c] == 1

    def test_missing_chars(self):
        assert _CONV_TABLE[ord("N")] == 2
        assert _CONV_TABLE[ord("n")] == 2
        assert _CONV_TABLE[ord("-")] == 2


class TestReadPsmcfa:
    def test_basic(self, tmp_path):
        fa = tmp_path / "test.psmcfa"
        fa.write_text(">chr1\nTTTTKTTTNNTT\n>chr2\nKKTT\n")
        data = read_psmcfa(fa)
        assert data.uns["n_seqs"] == 2
        r1, r2 = data.uns["records"]
        assert r1["name"] == "chr1"
        assert r1["L"] == 12
        assert r1["L_e"] == 10  # 12 - 2 N's
        assert r1["n_e"] == 1   # one K
        assert r2["n_e"] == 2   # two K's

    def test_empty_file(self, tmp_path):
        fa = tmp_path / "empty.psmcfa"
        fa.write_text("")
        with pytest.raises(ValueError, match="No sequences"):
            read_psmcfa(fa)

    def test_multiline_sequence(self, tmp_path):
        fa = tmp_path / "multi.psmcfa"
        fa.write_text(">seq1\nTTTT\nKKKK\nTTTT\n")
        data = read_psmcfa(fa)
        rec = data.uns["records"][0]
        assert rec["L"] == 12
        assert rec["n_e"] == 4


class TestReadMultihetsep:
    def test_preserves_ambiguous_pair_observations(self, tmp_path):
        path = tmp_path / "test.multihetsep"
        path.write_text("chr1\t10\t10\tAA,AT\n")

        data = read_multihetsep(path)

        obs = data.uns["segments"][0]["obs"][(0, 1)]
        assert obs.dtype == np.float64
        assert obs.tolist() == [1.5]

    def test_skip_ambiguous_treats_ambiguous_as_missing(self, tmp_path):
        path = tmp_path / "test.multihetsep"
        path.write_text("chr1\t10\t10\tAA,AT\n")

        data = read_multihetsep(path, skip_ambiguous=True)

        obs = data.uns["segments"][0]["obs"][(0, 1)]
        assert obs.dtype == np.int8
        assert obs.tolist() == [-1.0]

    def test_missing_in_any_phase_config_is_missing(self, tmp_path):
        path = tmp_path / "test.multihetsep"
        path.write_text("chr1\t10\t10\tAA,?T\n")

        data = read_multihetsep(path)

        obs = data.uns["segments"][0]["obs"][(0, 1)]
        assert obs.dtype == np.int8
        assert obs.tolist() == [0]
