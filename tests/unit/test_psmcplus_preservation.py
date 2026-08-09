"""Fast integrity checks for the PSMC+ preservation bundle."""

from __future__ import annotations

import json
from pathlib import Path

from smckit._provenance import sha256_file

ROOT = Path(__file__).resolve().parents[2]


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
