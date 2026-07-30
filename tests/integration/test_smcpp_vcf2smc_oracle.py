"""Compare native VCF preparation to the preserved upstream converter."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from smckit.io import read_smcpp_input
from smckit.pp import smcpp_from_vcf

pytestmark = pytest.mark.oracle

ROOT = Path(__file__).resolve().parents[2]
VCF = ROOT / "tests" / "data" / "smcpp_preprocess.vcf"
HELPER = ROOT / "tests" / "helpers" / "run_upstream_smcpp_vcf2smc.py"
VENDOR = ROOT / "vendor" / "smcpp"


@pytest.mark.skipif(importlib.util.find_spec("pysam") is None, reason="pysam unavailable")
def test_native_vcf2smc_matches_preserved_upstream(tmp_path) -> None:
    import pysam

    compressed = tmp_path / "input.vcf.gz"
    pysam.tabix_compress(str(VCF), str(compressed), force=True)
    pysam.tabix_index(str(compressed), preset="vcf", force=True)
    native_path = tmp_path / "native.smc"
    upstream_path = tmp_path / "upstream.smc"

    native = smcpp_from_vcf(
        compressed,
        native_path,
        contig="chr1",
        populations={"pop": ["s1", "s2"]},
        distinguished=[("s1", 0), ("s1", 1)],
        length=20,
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "--vendor-root",
            str(VENDOR),
            "--vcf",
            str(compressed),
            "--output",
            str(upstream_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    upstream = read_smcpp_input(upstream_path)

    assert native.uns["total_sites"] == upstream.uns["total_sites"] == 20
    assert native.uns["n_undist"] == upstream.uns["n_undist"] == 2
    assert native.uns["joint_observations"] == upstream.uns["joint_observations"]
    assert native.uns["smcpp_header"]["pids"] == upstream.uns["smcpp_header"]["pids"]
    assert native.uns["smcpp_header"]["dist"] == upstream.uns["smcpp_header"]["dist"]
    assert native.uns["smcpp_header"]["undist"] == upstream.uns["smcpp_header"]["undist"]
