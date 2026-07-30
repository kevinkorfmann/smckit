"""Independent synthetic split-family oracle for MSMC-IM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smckit.tl import msmc_im

INPUT = Path(__file__).parents[1] / "data" / "msmc_im_synthetic_split.final.txt"
ARRAY_FIELDS = (
    "left_boundary",
    "right_boundary",
    "N1",
    "N2",
    "N1_raw",
    "N2_raw",
    "m",
    "m_thresholded",
    "M",
)

pytestmark = pytest.mark.oracle


def test_synthetic_split_native_matches_preserved_upstream(tmp_path) -> None:
    options = {
        "pattern": "1*6",
        "mu": 1.25e-8,
        "N1_init": 20_000.0,
        "N2_init": 16_000.0,
        "m_init": 2e-5,
        "beta": (1e-8, 1e-6),
    }
    upstream = msmc_im(
        INPUT,
        implementation="upstream",
        output_prefix=tmp_path / "upstream",
        **options,
    ).results["msmc_im"]
    native = msmc_im(
        INPUT,
        implementation="native",
        output_prefix=tmp_path / "native",
        **options,
    ).results["msmc_im"]

    for field in ARRAY_FIELDS:
        np.testing.assert_allclose(
            np.asarray(native[field]),
            np.asarray(upstream[field]),
            rtol=1e-8,
            atol=1e-12,
        )
    assert native["init_chi_square"] == upstream["init_chi_square"]
    assert native["final_chi_square"] == upstream["final_chi_square"]
    assert native["split_time_quantiles"] == upstream["split_time_quantiles"]
    assert len(native["provenance"]["artifacts"]) == 1
    assert len(upstream["provenance"]["artifacts"]) == 2
    assert all(
        Path(artifact["path"]).is_file()
        for result in (native, upstream)
        for artifact in result["provenance"]["artifacts"]
    )
