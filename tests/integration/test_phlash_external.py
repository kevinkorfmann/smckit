"""Real-package smoke test for the maintained PHLASH integration."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from smckit.tl import phlash

pytestmark = pytest.mark.integration


@pytest.mark.skipif(importlib.util.find_spec("phlash") is None, reason="PHLASH is not installed")
def test_installed_phlash_psmcfa_smoke(tmp_path) -> None:
    source = tmp_path / "synthetic.psmcfa"
    sequence = np.full(5_000, "T", dtype="<U1")
    sequence[::50] = "K"
    source.write_text(
        ">synthetic\n" + "".join(sequence.tolist()) + "\n",
        encoding="utf-8",
    )

    data = phlash(
        [source],
        input_kind="psmcfa",
        hold_out=False,
        random_seed=17,
        grid_size=16,
        niter=1,
        num_particles=4,
        num_workers=1,
        max_samples=1,
        overlap=20,
        progress=False,
    )
    result = data.results["phlash"]

    assert result["upstream"]["version"] == "1.0.6"
    assert result["n_posterior_samples"] == 4
    assert result["posterior_ne"].shape == (4, 16)
    assert np.all(np.isfinite(result["posterior_ne"]))
    assert result["provenance"]["seed"] == 17
