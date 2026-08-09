"""Validate and freeze the publication protocol as canonical JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SCENARIOS = {
    "constant",
    "bottleneck",
    "expansion",
    "split_with_migration",
    "structure",
    "selfing_dormancy",
}
REQUIRED_PLATFORMS = {"linux-x86_64", "macos-arm64", "nvidia-gpu"}
REQUIRED_METRICS = {
    "log_integrated_trajectory_error",
    "parameter_error",
    "likelihood_difference",
    "posterior_coverage",
    "runtime_seconds",
    "peak_memory_bytes",
    "installation_success",
}
PHLASH_SCENARIOS = {
    "constant",
    "bottleneck",
    "expansion",
    "selfing_dormancy",
}


def _positive_number(config: dict[str, Any], name: str) -> float:
    value = config.get(name)
    if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive number.")
    return float(value)


def validate_protocol(config: dict[str, Any]) -> None:
    """Reject publication configurations that omit a frozen roadmap requirement."""
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1.")
    for name in ("replicates", "timing_repetitions", "seed"):
        value = config.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if config["replicates"] < 20:
        raise ValueError("Publication protocol requires at least 20 simulation replicates.")
    if config["timing_repetitions"] < 5:
        raise ValueError("Publication protocol requires at least five timing repetitions.")
    for name in ("sequence_length", "recombination_rate", "mutation_rate"):
        _positive_number(config, name)

    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list) or set(scenarios) != REQUIRED_SCENARIOS:
        raise ValueError("scenarios must contain each frozen simulation scenario exactly once.")
    if len(scenarios) != len(set(scenarios)):
        raise ValueError("scenarios must not contain duplicates.")

    platforms = config.get("platforms")
    if not isinstance(platforms, list) or not REQUIRED_PLATFORMS.issubset(platforms):
        raise ValueError("platforms must include Linux x86-64, macOS ARM64, and NVIDIA GPU.")

    metrics = config.get("metrics")
    if not isinstance(metrics, list) or not REQUIRED_METRICS.issubset(metrics):
        raise ValueError("metrics omit one or more frozen publication outcomes.")

    empirical = config.get("empirical")
    if not isinstance(empirical, dict) or not {"human", "nonhuman"}.issubset(empirical):
        raise ValueError("empirical must define human and nonhuman datasets.")

    phlash = config.get("phlash")
    if not isinstance(phlash, dict):
        raise ValueError("phlash must define the frozen PHLASH inference protocol.")
    if set(phlash.get("scenarios", [])) != PHLASH_SCENARIOS:
        raise ValueError("phlash scenarios must contain every applicable scenario exactly once.")
    if len(phlash["scenarios"]) != len(set(phlash["scenarios"])):
        raise ValueError("phlash scenarios must not contain duplicates.")
    for name in (
        "window_size",
        "grid_size",
        "niter",
        "num_particles",
        "num_workers",
        "max_samples",
        "overlap",
        "inference_seed",
    ):
        value = phlash.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"phlash.{name} must be a positive integer.")
    if phlash["num_particles"] < 500:
        raise ValueError("Publication PHLASH inference requires at least 500 particles.")
    if phlash["niter"] < 1000:
        raise ValueError("Publication PHLASH inference requires at least 1000 iterations.")
    if phlash.get("hold_out") is not True:
        raise ValueError("Publication PHLASH inference requires an independent holdout contig.")
    credible_level = phlash.get("credible_level")
    if not isinstance(credible_level, int | float) or not 0 < credible_level < 1:
        raise ValueError("phlash.credible_level must lie strictly between zero and one.")
    evaluation_min = _positive_number(phlash, "evaluation_min_generations")
    evaluation_max = _positive_number(phlash, "evaluation_max_generations")
    if evaluation_min >= evaluation_max:
        raise ValueError("PHLASH evaluation time bounds must be strictly increasing.")


def freeze_protocol(source: Path, target: Path) -> dict[str, Any]:
    """Validate *source* and write a deterministic, content-addressed protocol."""
    target = Path(target)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite immutable protocol record: {target}")
    raw = source.read_bytes()
    loaded = yaml.safe_load(raw)
    if not isinstance(loaded, dict):
        raise ValueError("Publication config must be a YAML mapping.")
    validate_protocol(loaded)
    canonical_config = json.dumps(
        loaded,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    payload = {
        "schema_version": 1,
        "protocol_id": f"sha256:{hashlib.sha256(canonical_config).hexdigest()}",
        "source": {
            "path": source.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "config": loaded,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    freeze_protocol(args.source, args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
