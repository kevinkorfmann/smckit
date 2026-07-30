"""Read and write SMC++ span-encoded input files."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any

from smckit._core import SmcData


def _open_text(path: Path, mode: str):
    return (
        gzip.open(path, mode, encoding="utf-8")
        if path.suffix == ".gz"
        else path.open(
            mode,
            encoding="utf-8",
        )
    )


def _header_dimensions(header: dict[str, Any] | None) -> tuple[int, list[int], list[int]]:
    if header is None:
        return 1, [2], []
    pids = list(header.get("pids", []))
    undist = list(header.get("undist", []))
    dist = list(header.get("dist", []))
    n_populations = max(len(pids), len(undist), len(dist), 1)
    n_undist = [len(undist[index]) if index < len(undist) else 0 for index in range(n_populations)]
    n_distinguished = [
        len(dist[index]) if index < len(dist) else 0 for index in range(n_populations)
    ]
    return n_populations, n_distinguished, n_undist


def read_smcpp_input(
    path: str | Path,
    window_size: int = 1,
) -> SmcData:
    """Read one- or two-population ``.smc``/``.smc.gz`` data.

    Each row is a run-length encoded observation with one ``(a, b, n)``
    triplet per population. One-population observations retain the historical
    ``(span, a, b)`` representation used by the native HMM. Multi-population
    observations are preserved in ``uns["records_by_population"]`` and
    ``uns["joint_observations"]`` without silently dropping columns.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"SMC++ input does not exist: {source}")
    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    header_meta: dict[str, Any] | None = None
    row_tokens: list[list[int]] = []
    with _open_text(source, "rt") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if line.startswith("# SMC++ "):
                    try:
                        candidate = json.loads(line[8:].strip())
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Malformed SMC++ JSON header at {source}:{line_number}."
                        ) from exc
                    if not isinstance(candidate, dict):
                        raise ValueError("SMC++ header must decode to a JSON object.")
                    header_meta = candidate
                continue
            try:
                values = [int(token) for token in line.split()]
            except ValueError as exc:
                raise ValueError(f"Non-integer SMC++ row at {source}:{line_number}.") from exc
            if len(values) < 4 or (len(values) - 1) % 3:
                raise ValueError(
                    f"SMC++ row at {source}:{line_number} must contain a span "
                    "and one or more (a, b, n) triplets."
                )
            if values[0] <= 0:
                raise ValueError(f"SMC++ span must be positive at {source}:{line_number}.")
            row_tokens.append(values)

    row_populations = {int((len(values) - 1) / 3) for values in row_tokens}
    if len(row_populations) > 1:
        raise ValueError("All SMC++ rows must have the same number of population triplets.")
    row_n_populations = next(iter(row_populations), 1)
    header_n_populations, header_n_distinguished, header_n_undist = _header_dimensions(header_meta)
    if header_meta is not None and row_tokens and row_n_populations != header_n_populations:
        raise ValueError("SMC++ header population count does not match the observation columns.")
    n_populations = row_n_populations if row_tokens else header_n_populations

    modal_n_undist: list[int] = []
    for population in range(n_populations):
        if population < len(header_n_undist) and header_n_undist[population]:
            modal_n_undist.append(header_n_undist[population])
            continue
        counts = Counter(values[1 + 3 * population + 2] for values in row_tokens)
        nonzero = Counter({key: value for key, value in counts.items() if key > 0})
        modal_n_undist.append(nonzero.most_common(1)[0][0] if nonzero else 0)

    observations_by_population: list[list[tuple[int, int, int]]] = [
        [] for _ in range(n_populations)
    ]
    joint_observations: list[tuple[int, tuple[tuple[int, int, int], ...]]] = []
    total_sites = 0
    for values in row_tokens:
        span = values[0]
        total_sites += span
        population_values: list[tuple[int, int, int]] = []
        for population in range(n_populations):
            a, b, n_observed = values[1 + 3 * population : 4 + 3 * population]
            n_expected = modal_n_undist[population]
            if a < -1 or b < 0 or n_observed < 0 or b > n_observed:
                raise ValueError("SMC++ observations must satisfy a >= -1 and 0 <= b <= n.")
            observation = (a, b, n_observed)
            if a == -1 or (n_expected and n_observed != n_expected):
                observations_by_population[population].append((span, -1, -1))
            else:
                observations_by_population[population].append((span, a, b))
            population_values.append(observation)
        joint_observations.append((span, tuple(population_values)))

    pids = list((header_meta or {}).get("pids", []))
    if len(pids) < n_populations:
        pids.extend(f"population_{index + 1}" for index in range(len(pids), n_populations))
    dist = list((header_meta or {}).get("dist", []))
    undist = list((header_meta or {}).get("undist", []))
    n_distinguished = [
        (
            header_n_distinguished[index]
            if index < len(header_n_distinguished) and header_n_distinguished[index]
            else (2 if n_populations == 1 else 0)
        )
        for index in range(n_populations)
    ]

    records_by_population = [
        [
            {
                "name": source.stem,
                "population": pids[population],
                "observations": observations_by_population[population],
                "n_undist": modal_n_undist[population],
                "n_distinguished": n_distinguished[population],
                "total_sites": total_sites,
                "pids": pids,
                "distinguished_samples": (
                    dist
                    if n_populations == 1
                    else (dist[population] if population < len(dist) else [])
                ),
                "undistinguished_samples": (
                    undist
                    if n_populations == 1
                    else (undist[population] if population < len(undist) else [])
                ),
            }
        ]
        for population in range(n_populations)
    ]
    records = records_by_population[0] if n_populations == 1 else []

    return SmcData(
        window_size=window_size,
        uns={
            "records": records,
            "records_by_population": records_by_population,
            "joint_observations": joint_observations,
            "n_populations": n_populations,
            "populations": pids,
            "n_undist": modal_n_undist[0] if n_populations == 1 else None,
            "n_undist_by_population": modal_n_undist,
            "n_distinguished": n_distinguished[0] if n_populations == 1 else None,
            "n_distinguished_by_population": n_distinguished,
            "n_seqs": 1,
            "total_sites": total_sites,
            "pids": pids,
            "source_path": str(source),
            "smcpp_header": header_meta,
        },
    )


def write_smcpp_input(
    data: SmcData,
    path: str | Path,
) -> Path:
    """Write preserved one- or multi-population observations to SMC++ format."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    header = data.uns.get("smcpp_header")
    observations = data.uns.get("joint_observations")
    if observations is None:
        records = data.uns.get("records", [])
        if len(records) != 1:
            raise ValueError("write_smcpp_input requires one record or joint observations.")
        n_undist = int(data.uns["n_undist"])
        observations = [
            (span, ((a, b, n_undist),) if a >= 0 else ((-1, 0, 0),))
            for span, a, b in records[0]["observations"]
        ]

    with _open_text(target, "wt") as handle:
        if header is not None:
            handle.write("# SMC++ " + json.dumps(header, sort_keys=True) + "\n")
        for span, populations in observations:
            flattened = [int(span)]
            for a, b, n_observed in populations:
                flattened.extend([int(a), int(b), int(n_observed)])
            # Upstream SMC++ uses ``pandas.read_csv(sep=" ")`` rather than
            # generic whitespace parsing, so emit literal single spaces.
            handle.write(" ".join(map(str, flattened)) + "\n")
    return target


__all__ = ["read_smcpp_input", "write_smcpp_input"]
