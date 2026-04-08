"""Read SMC++ input format (.smc / .smc.gz).

The SMC++ input format is tab-separated with columns:

    span  distinguished_allele  undistinguished_derived  undistinguished_total

Each row is a run-length encoded observation repeated ``span`` times, not a
"gap before variant" record. In upstream SMC++, monomorphic stretches are
represented directly as rows such as ``span  0  0  n``.

Lines starting with '#' are comments.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

from smckit._core import SmcData


def read_smcpp_input(
    path: str | Path,
    window_size: int = 1,
) -> SmcData:
    """Read an SMC++ input file into SmcData.

    Parameters
    ----------
    path : str or Path
        Path to .smc or .smc.gz file.
    window_size : int
        Window size in base pairs (default 1 = per-base).

    Returns
    -------
    SmcData
        Data container with observations in ``uns["records"]`` and
        sample size in ``uns["n_undist"]``.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open

    raw_rows: list[tuple[int, int, int, int]] = []
    total_sites = 0
    header_meta: dict | None = None

    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if line.startswith("# SMC++ "):
                    try:
                        header_meta = json.loads(line[8:].strip())
                    except json.JSONDecodeError:
                        header_meta = None
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            span = int(parts[0])
            a = int(parts[1])
            b = int(parts[2])
            n_obs = int(parts[3])
            raw_rows.append((span, a, b, n_obs))
            total_sites += span

    if header_meta is not None:
        undist = header_meta.get("undist", [])
        dist = header_meta.get("dist", [])
        pids = header_meta.get("pids", [])
        n_distinguished = len(dist[0]) if dist else 0
        n_undist_header = len(undist[0]) if undist else None
    else:
        undist = []
        dist = []
        pids = []
        n_distinguished = 2
        n_undist_header = None

    if not raw_rows:
        n_undist = n_undist_header or 0
        observations: list[tuple[int, int, int]] = []
    else:
        if n_undist_header is not None:
            n_undist = n_undist_header
        else:
            counts = Counter(n_obs for _, _, _, n_obs in raw_rows)
            n_undist = counts.most_common(1)[0][0]
        observations = []
        for span, a, b, n_obs in raw_rows:
            if n_obs != n_undist:
                # Variable undistinguished sample size is treated as missing for
                # the fixed-size HMM, using the dedicated missing emission row.
                observations.append((span, -1, -1))
            else:
                observations.append((span, a, b))

    record = {
        "name": path.stem,
        "observations": observations,
        "n_undist": n_undist,
        "n_distinguished": n_distinguished,
        "total_sites": total_sites,
        "pids": pids,
        "distinguished_samples": dist,
        "undistinguished_samples": undist,
    }

    data = SmcData(
        window_size=window_size,
        uns={
            "records": [record],
            "n_undist": n_undist,
            "n_distinguished": n_distinguished,
            "n_seqs": 1,
            "total_sites": total_sites,
            "pids": pids,
        },
    )

    return data
