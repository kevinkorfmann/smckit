"""Compare native multihetsep preparation to pinned msmc-tools execution."""

from __future__ import annotations

import gzip
import os
import subprocess
import sys
from pathlib import Path

import pytest

from smckit._provenance import sha256_file
from smckit.pp import multihetsep_from_vcf

PINNED_SHA256 = "caaa87a07e0fe2dc7228f30c9aff759cf86e9f61b8332aabd41398399ea6331b"


def _write_vcf(
    path: Path,
    sample: str,
    rows: list[tuple[int, str, str, str]],
) -> None:
    lines = [
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + sample,
    ]
    for position, reference, alternate, genotype in rows:
        lines.append(f"chr1\t{position}\t.\t{reference}\t{alternate}\t.\tPASS\t.\tGT\t{genotype}")
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _normalized_rows(text: str) -> list[tuple[str, int, int, tuple[str, ...]]]:
    rows = []
    for line in text.splitlines():
        contig, position, callable_count, configurations = line.split("\t")
        rows.append(
            (
                contig,
                int(position),
                int(callable_count),
                tuple(sorted(configurations.split(","))),
            )
        )
    return rows


@pytest.mark.integration
@pytest.mark.oracle
def test_native_matches_pinned_generate_multihetsep(tmp_path: Path) -> None:
    raw_script = os.environ.get("SMCKIT_MSMC_TOOLS_GENERATE")
    if not raw_script:
        pytest.skip("Set SMCKIT_MSMC_TOOLS_GENERATE to the pinned helper source.")
    script = Path(raw_script).expanduser().resolve()
    if not script.is_file():
        pytest.fail(f"Pinned msmc-tools helper does not exist: {script}")
    assert sha256_file(script) == PINNED_SHA256

    child = tmp_path / "child.vcf.gz"
    father = tmp_path / "father.vcf.gz"
    mother = tmp_path / "mother.vcf.gz"
    first_mask = tmp_path / "first.bed"
    second_mask = tmp_path / "second.mask"
    native_output = tmp_path / "native.mhs"
    _write_vcf(child, "child", [(2, "A", "G", "0/1"), (5, "C", "T", "0/1")])
    _write_vcf(father, "father", [(2, "A", "G", "0|0"), (7, "C", "T", "0|1")])
    _write_vcf(mother, "mother", [(2, "A", "G", "1|1"), (5, "C", "T", "0|0")])
    first_mask.write_text("chr1\t0\t4\nchr1\t6\t12\n", encoding="utf-8")
    second_mask.write_text("1\t10\n", encoding="utf-8")

    command = [
        sys.executable,
        str(script),
        f"--mask={first_mask}",
        f"--mask={second_mask}",
        "--trio=0,1,2",
        str(child),
        str(father),
        str(mother),
    ]
    upstream = subprocess.run(command, check=False, capture_output=True, text=True)
    assert upstream.returncode == 0, upstream.stderr

    multihetsep_from_vcf(
        [child, father, mother],
        native_output,
        mask_paths=[first_mask, second_mask],
        trios=[(0, 1, 2)],
    )

    assert _normalized_rows(native_output.read_text()) == _normalized_rows(upstream.stdout)
    assert _normalized_rows(upstream.stdout) == [
        ("chr1", 2, 2, ("AAGG",)),
        ("chr1", 7, 3, ("CTCC",)),
    ]
