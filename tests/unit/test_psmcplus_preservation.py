"""Fast integrity checks for the PSMC+ preservation bundle."""

from __future__ import annotations

import json
from pathlib import Path

from smckit._provenance import sha256_file

ROOT = Path(__file__).resolve().parents[2]
MACOS_EVIDENCE = (
    ROOT / "workflow" / "publication" / "evidence" / "psmcplus-macos-arm64" / "sha256-73ea05e5"
)


def _manifest() -> dict[str, str]:
    manifest = json.loads((ROOT / "preservation/upstreams.json").read_text(encoding="utf-8"))
    return manifest["tools"]["psmcplus"]


def test_psmcplus_source_and_oracle_hashes_match_manifest() -> None:
    entry = _manifest()

    assert sha256_file(ROOT / "vendor/PSMCplus/LICENSE.txt") == entry["license_sha256"]
    assert sha256_file(ROOT / "vendor/PSMCplus/PSMCplus.py") == entry["cli_sha256"]
    assert sha256_file(ROOT / "vendor/PSMCplus/README.md") == entry["readme_sha256"]
    assert (
        sha256_file(ROOT / "tests/data/psmcplus/constpop_D4_1iter.final_parameters.txt")
        == entry["oracle_fixture_sha256"]
    )
    assert (
        sha256_file(ROOT / "tests/data/psmcplus/kernel_oracle_v1.npz")
        == entry["kernel_oracle_sha256"]
    )
    assert (
        sha256_file(ROOT / "tests/data/psmcplus/preprocessing_oracle_v1.npz")
        == entry["preprocessing_oracle_sha256"]
    )
    assert (
        sha256_file(ROOT / "tests/data/psmcplus/decode_oracle_v1.npz")
        == entry["decode_oracle_sha256"]
    )
    assert (
        sha256_file(ROOT / "tests/data/psmcplus/rate_map_D4_1iter.final_parameters.txt")
        == entry["rate_map_oracle_sha256"]
    )
    assert (
        sha256_file(ROOT / "tests/data/psmcplus/rate_map_decode_oracle_v1.npz")
        == entry["rate_map_decode_oracle_sha256"]
    )


def test_psmcplus_container_uses_manifest_base_and_hash_lock() -> None:
    entry = _manifest()
    container = ROOT / "preservation/containers/psmcplus"
    dockerfile = (container / "Dockerfile").read_text(encoding="utf-8")
    lock = (container / "requirements.lock").read_text(encoding="utf-8")

    assert entry["container_base"] in dockerfile
    assert "--require-hashes" in dockerfile
    for requirement in [
        "joblib==1.1.0",
        "matplotlib==3.5.2",
        "numba==0.55.2",
        "numpy==1.22.4",
        "pandas==1.4.3",
        "psutil==5.9.1",
        "scipy==1.8.1",
    ]:
        assert requirement in lock


def test_psmcplus_macos_arm64_matrix_passes_frozen_capability_gate() -> None:
    record = json.loads(
        (MACOS_EVIDENCE / "psmcplus-macos-arm64-matrix.json").read_text(encoding="utf-8")
    )

    assert record["source"] == {
        "clean": True,
        "commit": "9f924bdd385fddd1f45cc214d9cbc80b2e761606",
        "status": [],
    }
    assert record["upstream_commit"] == "032168f2ceed3c0e46b7f214f890faf83dff41ae"
    assert record["platform"].startswith("macOS-26.2-arm64")
    assert record["passed"] is True
    assert len(record["cases"]) == 12
    assert all(case["comparison"]["passed"] for case in record["cases"])

    fit = [case for case in record["cases"] if case["mode"] == "fit"]
    decode = [case for case in record["cases"] if case["mode"] == "decode"]
    assert max(case["comparison"]["lambda_relative_error_max"] for case in fit) < 2e-8
    assert max(case["comparison"]["log_likelihood_absolute_error"] for case in fit) < 1e-9
    assert max(case["comparison"]["posterior_absolute_error_max"] for case in decode) < 3e-12
    assert all(case["comparison"]["position_exact"] for case in decode)
    assert all(case["comparison"]["marginal_position_exact"] for case in decode)


def test_psmcplus_macos_arm64_matrix_checksum() -> None:
    lines = (MACOS_EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert lines
    for line in lines:
        expected, filename = line.split(maxsplit=1)
        assert sha256_file(MACOS_EVIDENCE / filename) == expected
