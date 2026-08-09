"""Fast integrity checks for the diCal2 preservation and repair bundle."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from smckit._provenance import sha256_file

ROOT = Path(__file__).resolve().parents[2]
SOURCE_RELATIVE = Path("src/edu/berkeley/diCal2/csd/SMCSDemoEMObjectiveFunction.java")


def _manifest() -> dict[str, str]:
    manifest = json.loads((ROOT / "preservation/upstreams.json").read_text(encoding="utf-8"))
    return manifest["tools"]["dical2"]


def test_dical2_source_and_repair_hashes_match_manifest() -> None:
    entry = _manifest()
    vendor = ROOT / entry["path"]

    assert sha256_file(vendor / "LICENSE_GPLv3.txt") == entry["license_sha256"]
    assert sha256_file(vendor / "diCal2.jar") == entry["jar_sha256"]
    assert sha256_file(vendor / SOURCE_RELATIVE) == entry["marginal_kl_source_sha256"]
    assert (
        sha256_file(ROOT / "preservation/patches/dical2/marginal-kl-get-likelihood.patch")
        == entry["marginal_kl_repair_patch_sha256"]
    )
    assert (
        sha256_file(vendor / "deprecated_previous_releases/diCal2_2_0_5.tar.gz")
        == entry["source_archive_sha256"]
    )


def test_dical2_marginal_kl_patch_applies_with_zero_fuzz(tmp_path: Path) -> None:
    patch_tool = shutil.which("patch")
    assert patch_tool is not None
    source_root = tmp_path / "source"
    copied_source = source_root / SOURCE_RELATIVE
    copied_source.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "vendor/diCal2" / SOURCE_RELATIVE, copied_source)
    patch_path = ROOT / "preservation/patches/dical2/marginal-kl-get-likelihood.patch"

    subprocess.run(
        [patch_tool, "--batch", "--fuzz=0", "-p1", "-i", str(patch_path)],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    )

    patched = copied_source.read_text(encoding="utf-8")
    assert "return this.logLikelihood;" in patched
    assert "return this.value(this.oldCore, 1d, null);" not in patched
    assert (
        sha256_file(ROOT / "vendor/diCal2" / SOURCE_RELATIVE)
        == _manifest()["marginal_kl_source_sha256"]
    )
