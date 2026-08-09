"""Direct native-versus-Java diCal2 VCF reader oracles."""

from __future__ import annotations

import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest

import smckit
from smckit.io._dical2 import DiCal2Config, read_dical2_vcf

ROOT = Path(__file__).resolve().parents[2]
JAR = ROOT / "vendor/diCal2/diCal2.jar"
HARNESS = ROOT / "tests/helpers/DiCal2VcfOracle.java"
JAVA = smckit.upstream.status("dical2")["runtime"]["path"]
JAVAC = None if JAVA is None else str(Path(JAVA).with_name("javac"))

pytestmark = [
    pytest.mark.oracle,
    pytest.mark.skipif(
        JAVA is None or JAVAC is None or not Path(JAVAC).is_file(),
        reason="Java compiler/runtime not available",
    ),
]


def _config(n_haplotypes: int, *, n_alleles: int = 2, include=None) -> DiCal2Config:
    if include is None:
        include = [True] * n_haplotypes
    multiplicities = np.asarray([[int(value)] for value in include], dtype=np.int64)
    return DiCal2Config(
        seq_length=8,
        n_alleles=n_alleles,
        n_populations=1,
        haplotype_populations=[0 if value else -1 for value in include],
        haplotypes_to_include=list(include),
        haplotype_multiplicities=multiplicities,
        sample_sizes=multiplicities.sum(axis=0),
    )


def _write_case(tmp_path: Path, rows) -> tuple[Path, Path]:
    reference = tmp_path / "reference.fa"
    reference.write_text("ACGTACGT\n")
    vcf = tmp_path / "case.vcf"
    sample_count = len(rows[0][-1])
    header = [
        "#CHROM",
        "POS",
        "ID",
        "REF",
        "ALT",
        "QUAL",
        "FILTER",
        "INFO",
        "FORMAT",
        *[f"S{idx + 1}" for idx in range(sample_count)],
    ]
    lines = ["\t".join(header)]
    for pos, ref, alt, genotypes in rows:
        lines.append("\t".join(["1", str(pos), ".", ref, alt, ".", ".", ".", "GT", *genotypes]))
    vcf.write_text("\n".join(lines) + "\n")
    return vcf, reference


@pytest.fixture(scope="module")
def java_oracle(tmp_path_factory):
    build = tmp_path_factory.mktemp("dical2-vcf-java")
    nested = build / "nested"
    nested.mkdir()
    jars = [JAR]
    with zipfile.ZipFile(JAR) as archive:
        for name in archive.namelist():
            if not name.endswith(".jar"):
                continue
            target = nested / Path(name).name
            target.write_bytes(archive.read(name))
            jars.append(target)
    classpath = os.pathsep.join(str(path) for path in jars)
    compiled = subprocess.run(
        [JAVAC, "-cp", classpath, "-d", str(build), str(HARNESS)],
        check=False,
        capture_output=True,
        text=True,
    )
    if compiled.returncode != 0:
        pytest.fail(f"Could not compile direct diCal2 VCF oracle:\n{compiled.stderr}")
    runtime_classpath = os.pathsep.join([str(build), classpath])

    def run(vcf, reference, config, *, accept=False, ignore_duplicates=False):
        include = ",".join("1" if value else "0" for value in config.haplotypes_to_include)
        proc = subprocess.run(
            [
                str(JAVA),
                "-cp",
                runtime_classpath,
                "DiCal2VcfOracle",
                str(vcf),
                str(reference),
                str(config.n_alleles),
                include,
                str(bool(accept)).lower(),
                str(bool(ignore_duplicates)).lower(),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload: dict[str, object] = {"haplotypes": []}
        for line in proc.stdout.splitlines():
            if not line.startswith("SMCKIT_"):
                continue
            fields = line.split("\t")
            key = fields[0].removeprefix("SMCKIT_").lower()
            if key == "hap":
                payload["haplotypes"].append([int(value) for value in fields[2].split(",")])
            elif key in {"seg", "reference"}:
                payload[key] = [int(value) for value in fields[1].split(",") if value]
            else:
                payload[key] = fields[1]
        return payload

    return run


@pytest.mark.parametrize(
    ("rows", "config", "options"),
    [
        (
            [(2, "C", "G", ["0", "1"]), (4, "T", "A", ["1", "0"])],
            _config(2),
            {},
        ),
        ([(2, "C", "G", ["0|.", "1|0"])], _config(4), {}),
        ([(2, "C", "G", ["0/1", "0|1"])], _config(4), {"accept": True}),
        ([(2, "C", "G", ["0|1", "1|0"])], _config(4, n_alleles=4), {}),
        ([(2, "C", ".", ["1", "0"])], _config(2), {}),
        (
            [(2, "C", "G", ["0|1"]), (2, "C", "T", ["1|0"])],
            _config(2),
            {"ignore_duplicates": True},
        ),
        (
            [(2, "C", "G", ["0|1", "x|y"])],
            _config(4, n_alleles=4, include=[True, True, False, False]),
            {},
        ),
    ],
    ids=[
        "haploid",
        "partially-missing-phased",
        "unphased-opt-in",
        "four-allele",
        "missing-alt",
        "duplicate-opt-in",
        "excluded-malformed-sample",
    ],
)
def test_native_vcf_arrays_match_direct_java(java_oracle, tmp_path, rows, config, options):
    vcf, reference = _write_case(tmp_path, rows)
    java = java_oracle(vcf, reference, config, **options)
    assert java["status"] == "OK"

    sequences, positions, reference_length, reference_alleles = read_dical2_vcf(
        vcf,
        reference,
        config,
        accept_unphased_as_missing=options.get("accept", False),
        vcf_ignore_double_entries=options.get("ignore_duplicates", False),
    )
    java_positions = np.asarray(java["seg"], dtype=np.int64)
    java_reference = np.asarray(java["reference"], dtype=np.int8)
    java_reference[java_positions] = -1
    np.testing.assert_array_equal(positions, java_positions)
    np.testing.assert_array_equal(sequences, np.asarray(java["haplotypes"], dtype=np.int8))
    np.testing.assert_array_equal(reference_alleles, java_reference)
    assert reference_length == len(java_reference)


@pytest.mark.parametrize(
    ("rows", "config", "java_message", "native_message"),
    [
        (
            [(2, "C", "G", ["0/1", "0|1"])],
            _config(4),
            "Genotype is not phased",
            "is not phased",
        ),
        (
            [(2, "C", "G", ["0|10", "0|1"])],
            _config(4),
            "Genotype entry",
            "one haploid allele",
        ),
        (
            [(2, "C", "G", ["2|0", "0|1"])],
            _config(4),
            "Invalid allele",
            "Invalid allele",
        ),
        (
            [(2, "C", "G", ["0|1"]), (2, "C", "T", ["1|0"])],
            _config(2),
            "More than one entry",
            "duplicate entry",
        ),
    ],
    ids=["unphased-default", "malformed-width", "invalid-index", "duplicate-default"],
)
def test_native_vcf_failures_match_direct_java(
    java_oracle,
    tmp_path,
    rows,
    config,
    java_message,
    native_message,
):
    vcf, reference = _write_case(tmp_path, rows)
    java = java_oracle(vcf, reference, config)
    assert java["status"] == "ERROR"
    assert java_message in java["error_message"]
    with pytest.raises(ValueError, match=native_message):
        read_dical2_vcf(vcf, reference, config)
