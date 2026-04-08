"""Report tracked SMC++ native-vs-upstream parity metrics.

This script compares:

1. the strict small one-pop control fixture
2. a larger tracked `.smc` one-pop fixture in ``tests/data``

It is a diagnostic script, not a public API.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smckit._core import SmcData
from smckit.io import read_smcpp_input
from smckit.tl import smcpp


def _small_control_data() -> SmcData:
    observations = []
    for i in range(120):
        observations.append((4999, 0, 0))
        observations.append((1, i % 3, (i * 2) % 6))
    return SmcData(
        uns={
            "records": [{"name": "default_onepop", "observations": observations, "n_undist": 5}],
            "n_undist": 5,
        },
    )


def _shared_grid_metrics(native: dict, upstream: dict, *, n_grid: int = 200) -> dict[str, float]:
    native_time = np.asarray(native["time"], dtype=float)
    upstream_time = np.asarray(upstream["time"], dtype=float)
    t_min = max(
        float(np.min(native_time[native_time > 0])),
        float(np.min(upstream_time[upstream_time > 0])),
    )
    t_max = min(float(np.max(native_time)), float(np.max(upstream_time)))
    grid = np.geomspace(t_min, t_max, n_grid)

    def _step_eval(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(times, query, side="right") - 1
        idx = np.clip(idx, 0, len(values) - 1)
        return values[idx]

    native_ne = _step_eval(native_time, np.asarray(native["ne"], dtype=float), grid)
    upstream_ne = _step_eval(upstream_time, np.asarray(upstream["ne"], dtype=float), grid)
    rel = np.abs(native_ne - upstream_ne) / np.maximum(np.abs(upstream_ne), 1e-12)
    return {
        "log_corr": float(np.corrcoef(np.log(native_ne), np.log(upstream_ne))[0, 1]),
        "scale_ratio": float(np.median(native_ne) / np.median(upstream_ne)),
        "median_rel": float(np.median(rel)),
        "median_log10_error": float(np.median(np.abs(np.log10(native_ne / upstream_ne)))),
    }


def _run_pair(data: SmcData) -> tuple[dict, dict]:
    params = {
        "n_intervals": 4,
        "regularization": 10.0,
        "max_iterations": 2,
        "seed": 42,
        "generation_time": 1.0,
    }
    native = smcpp(copy.deepcopy(data), implementation="native", **params).results["smcpp"]
    upstream = smcpp(copy.deepcopy(data), implementation="upstream", **params).results["smcpp"]
    return native, upstream


def main() -> None:
    larger_path = ROOT / "tests" / "data" / "smcpp_onepop_larger.smc"
    cases = {
        "small_control": _small_control_data(),
        "larger_tracked_fixture": read_smcpp_input(larger_path),
    }
    rows = {}
    for name, data in cases.items():
        native, upstream = _run_pair(data)
        rows[name] = _shared_grid_metrics(native, upstream)
        rows[name]["native_intervals"] = int(len(native["ne"]))
        rows[name]["upstream_intervals"] = int(len(upstream["ne"]))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
