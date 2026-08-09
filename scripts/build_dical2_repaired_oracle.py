#!/usr/bin/env python3
"""Build the GPL diCal2 marginal-KL repair as a separate oracle jar."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/diCal2"
SOURCE_RELATIVE = Path("src/edu/berkeley/diCal2/csd/SMCSDemoEMObjectiveFunction.java")
PINNED_JAR = VENDOR / "diCal2.jar"
PINNED_SOURCE = VENDOR / SOURCE_RELATIVE
PATCH = ROOT / "preservation/patches/dical2/marginal-kl-get-likelihood.patch"
EXPECTED_JAR_SHA256 = "edd72b7eaee65e8b2e12fdcc3200217a81983c50a1bded828395915dc77acfaf"
EXPECTED_SOURCE_SHA256 = "cc1b972e13f71ba8e25fce61e44afe6acdf89f81249d3592bdde768559e21a2b"
EXPECTED_PATCH_SHA256 = "bfed5d6d0141ee4db160d378a80bcbaa5a83d7d97eb137a719cf59a79cc7fc71"
EXPECTED_REPAIRED_JAR_SHA256 = "ef3327b6ee681302e06879abaf4b4bfb88b10042e28033fa7a73b89604306ff0"
FIXED_ZIP_DATE = "2020-01-01T00:00:00Z"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(
            f"Pinned diCal2 input changed: {path} has SHA-256 {observed}, expected {expected}."
        )


def resolve_tool(explicit: str | None, name: str) -> str:
    candidates: list[str | None] = [explicit]
    java_override = os.environ.get("SMCKIT_DICAL2_JAVA")
    if java_override:
        candidates.append(str(Path(java_override).expanduser().resolve().with_name(name)))
    candidates.extend(
        [
            f"/opt/homebrew/opt/openjdk/bin/{name}",
            f"/usr/local/opt/openjdk/bin/{name}",
            shutil.which(name),
        ]
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(Path(candidate).resolve())
    raise RuntimeError(f"Could not locate the JDK {name} executable.")


def run(command: list[str], *, cwd: Path | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def build(output: Path, *, javac: str | None, jar_tool: str | None, force: bool) -> dict:
    output = output.expanduser().resolve()
    require_hash(PINNED_JAR, EXPECTED_JAR_SHA256)
    require_hash(PINNED_SOURCE, EXPECTED_SOURCE_SHA256)
    require_hash(PATCH, EXPECTED_PATCH_SHA256)
    if output == VENDOR or VENDOR in output.parents:
        raise ValueError("The repaired oracle must be built outside immutable vendor/.")
    if output.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing output without --force: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    javac_path = resolve_tool(javac, "javac")
    jar_path = resolve_tool(jar_tool, "jar")

    with tempfile.TemporaryDirectory(prefix="smckit-dical2-repair-") as raw_temp:
        temp = Path(raw_temp)
        source_root = temp / "source"
        patched_source = source_root / SOURCE_RELATIVE
        patched_source.parent.mkdir(parents=True)
        shutil.copy2(PINNED_SOURCE, patched_source)
        run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(PATCH)], cwd=source_root)

        classes = temp / "classes"
        classes.mkdir()
        run(
            [
                javac_path,
                "--release",
                "8",
                "-classpath",
                str(PINNED_JAR),
                "-d",
                str(classes),
                str(patched_source),
            ]
        )
        compiled = sorted(path.relative_to(classes) for path in classes.rglob("*.class"))
        if not compiled:
            raise RuntimeError("The repaired diCal2 compilation produced no class files.")

        staged_output = temp / "repaired.jar"
        shutil.copy2(PINNED_JAR, staged_output)
        command = [
            jar_path,
            "--update",
            "--file",
            str(staged_output),
            f"--date={FIXED_ZIP_DATE}",
        ]
        for class_file in compiled:
            command.extend(["-C", str(classes), class_file.as_posix()])
        run(command)
        shutil.copy2(staged_output, output)

    output_sha256 = sha256(output)
    if output_sha256 != EXPECTED_REPAIRED_JAR_SHA256:
        raise RuntimeError(
            "Repaired diCal2 jar is not byte-reproducible: "
            f"observed {output_sha256}, expected {EXPECTED_REPAIRED_JAR_SHA256}."
        )
    version = subprocess.run(
        [javac_path, "-version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "schema_version": 1,
        "purpose": "diCal2 marginal-KL repaired-source scientific oracle",
        "license": "GPL-3.0-or-later",
        "pinned_jar": str(PINNED_JAR),
        "pinned_jar_sha256": EXPECTED_JAR_SHA256,
        "pinned_source": str(PINNED_SOURCE),
        "pinned_source_sha256": EXPECTED_SOURCE_SHA256,
        "patch": str(PATCH),
        "patch_sha256": sha256(PATCH),
        "repair": "return the already-computed sequence log likelihood from marginal-KL E-steps",
        "javac": javac_path,
        "javac_version": (version.stderr or version.stdout).strip(),
        "output": str(output),
        "output_sha256": output_sha256,
        "compiled_classes": [path.as_posix() for path in compiled],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--javac")
    parser.add_argument("--jar-tool")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    metadata = build(
        args.output.expanduser().resolve(),
        javac=args.javac,
        jar_tool=args.jar_tool,
        force=args.force,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
