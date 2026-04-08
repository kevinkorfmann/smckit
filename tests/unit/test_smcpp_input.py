"""Tests for SMC++ input parsing."""

from __future__ import annotations

from pathlib import Path

from smckit.io import read_smcpp_input


def test_read_smcpp_input_uses_modal_undistinguished_sample_size(tmp_path: Path) -> None:
    path = tmp_path / "variable_n.smc"
    path.write_text(
        "10\t0\t1\t5\n"
        "1\t1\t0\t3\n"
        "4\t0\t2\t5\n"
    )

    data = read_smcpp_input(path)

    assert data.uns["n_undist"] == 5
    assert data.uns["records"][0]["n_undist"] == 5
    assert data.uns["n_distinguished"] == 2
    assert data.uns["total_sites"] == 15


def test_read_smcpp_input_marks_variable_sample_size_rows_missing(tmp_path: Path) -> None:
    path = tmp_path / "variable_n.smc"
    path.write_text(
        "10\t0\t1\t5\n"
        "1\t1\t0\t3\n"
        "4\t0\t2\t5\n"
    )

    data = read_smcpp_input(path)

    assert data.uns["records"][0]["observations"] == [
        (10, 0, 1),
        (1, -1, -1),
        (4, 0, 2),
    ]


def test_read_smcpp_input_parses_smcpp_header_metadata(tmp_path: Path) -> None:
    path = tmp_path / "with_header.smc"
    path.write_text(
        "# SMC++ {\"version\":\"x\",\"pids\":[\"pop1\"],\"undist\":[[[\"u1\",0],[\"u1\",1],[\"u2\",0]]],\"dist\":[[[\"d1\",0],[\"d1\",1]]]} \n"
        "10\t0\t1\t3\n"
    )

    data = read_smcpp_input(path)

    assert data.uns["n_distinguished"] == 2
    assert data.uns["n_undist"] == 3
    assert data.uns["pids"] == ["pop1"]
    assert data.uns["records"][0]["distinguished_samples"] == [[["d1", 0], ["d1", 1]]]
