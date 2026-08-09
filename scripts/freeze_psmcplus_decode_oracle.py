#!/usr/bin/env python3
"""Normalize exact PSMC+ decode artifacts into a frozen numerical oracle.

Generate the source artifacts with the pinned raw upstream runner before using
this script. Their hashes are embedded so the normalized NPZ remains auditable.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

UPSTREAM_COMMIT = "032168f2ceed3c0e46b7f214f890faf83dff41ae"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--marginal-recombination", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/data/psmcplus/decode_oracle_v1.npz"),
    )
    return parser


def _header_value(text: str, label: str) -> str:
    prefix = f"# {label}"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].lstrip(" =")
    raise ValueError(f"Missing PSMC+ decode header {label!r}.")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _time_boundaries(text: str) -> np.ndarray:
    for line in text.splitlines():
        if not line.startswith("# ") or "," not in line:
            continue
        try:
            values = np.asarray([float(value) for value in line[2:].split(",")])
        except ValueError:
            continue
        if values.size >= 2:
            return values
    raise ValueError("PSMC+ posterior artifact is missing its time boundaries.")


def main() -> int:
    args = _parser().parse_args()
    posterior_path = args.posterior.resolve()
    marginal_path = args.marginal_recombination.resolve()
    posterior_text = posterior_path.read_text(encoding="utf-8")
    posterior_values = np.loadtxt(posterior_path, dtype=np.float64)
    marginal_values = np.loadtxt(marginal_path, dtype=np.float64)
    if posterior_values.ndim != 2 or posterior_values.shape[0] < 2:
        raise ValueError("PSMC+ posterior artifact must contain positions and state rows.")
    if marginal_values.ndim != 2 or marginal_values.shape[0] != 3:
        raise ValueError("PSMC+ marginal artifact must contain exactly three rows.")
    posterior = posterior_values[1:].T
    marginal = marginal_values[1:].T
    if not np.allclose(posterior.sum(axis=1), 1.0, rtol=1e-12, atol=1e-14):
        raise ValueError("PSMC+ posterior rows are not normalized.")
    if not np.allclose(marginal.sum(axis=1), 1.0, rtol=1e-12, atol=1e-14):
        raise ValueError("PSMC+ marginal probabilities are not normalized.")
    boundaries = _time_boundaries(posterior_text)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        oracle_commit=np.array(UPSTREAM_COMMIT),
        posterior_source_sha256=np.array(_sha256(posterior_path)),
        marginal_source_sha256=np.array(_sha256(marginal_path)),
        theta=np.array(float(_header_value(posterior_text, "theta=4*N_E*mu"))),
        rho=np.array(float(_header_value(posterior_text, "rho=4*N_E*r"))),
        bin_size=np.array(int(_header_value(posterior_text, "bin_size"))),
        log_likelihood=np.array(float(_header_value(posterior_text, "log likelihood is"))),
        boundaries=boundaries,
        positions=posterior_values[0].astype(np.int64),
        posterior=posterior,
        marginal_positions=marginal_values[0].astype(np.int64),
        marginal_recombination=marginal,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
