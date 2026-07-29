from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from smckit.tl import _phlash as adapter
from smckit.tl import phlash


class _Eta:
    t = np.array([0.0, 1.0, 10.0])

    def __call__(self, time, *, Ne=False):
        assert Ne
        return np.asarray(time) + 100.0


def test_phlash_normalizes_external_posterior(monkeypatch) -> None:
    models = [SimpleNamespace(theta=0.1, rho=0.2, eta=_Eta()) for _ in range(3)]
    fake = SimpleNamespace(
        __version__="1.0.6",
        psmc=lambda paths, **options: models,
    )
    monkeypatch.setattr(adapter, "_load_phlash", lambda: fake)

    result = phlash(["a.psmcfa"], input_kind="psmcfa", grid_size=5)
    payload = result.results["phlash"]

    assert payload["implementation"] == "upstream"
    assert payload["n_posterior_samples"] == 3
    np.testing.assert_allclose(payload["ne"], payload["credible_interval"]["lower"])


def test_phlash_rejects_native_rewrite_request() -> None:
    with pytest.raises(NotImplementedError, match="external integration"):
        phlash(["a.psmcfa"], implementation="native")
