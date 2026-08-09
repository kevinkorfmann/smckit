from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smckit.tl import _phlash as adapter
from smckit.tl import phlash


class _Eta:
    t = np.array([0.0, 1.0, 10.0])

    def __init__(self, offset: float = 100.0) -> None:
        self.offset = offset

    def __call__(self, time, *, Ne=False):
        assert Ne
        return np.asarray(time) + self.offset


def _models() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(theta=0.1 + index, rho=0.2 + index, eta=_Eta(100.0 + index))
        for index in range(3)
    ]


def test_phlash_normalizes_external_posterior(monkeypatch, tmp_path) -> None:
    source = tmp_path / "input.psmcfa"
    source.write_text(">chr1\nTTTT\n", encoding="utf-8")
    calls = []
    fake = SimpleNamespace(
        __version__="1.0.6",
        psmc=lambda paths, **options: calls.append((paths, options)) or _models(),
    )
    monkeypatch.setattr(adapter, "_load_phlash", lambda: fake)

    result = phlash(
        [source],
        input_kind="psmcfa",
        grid_size=5,
        credible_level=0.8,
        random_seed=None,
    )
    payload = result.results["phlash"]

    assert payload["implementation"] == "upstream"
    assert payload["n_posterior_samples"] == 3
    assert payload["credible_interval"]["level"] == 0.8
    assert payload["theta"] == pytest.approx(1.1)
    assert payload["rho"] == pytest.approx(1.2)
    assert payload["provenance"]["input_sha256"]
    assert calls[0][1]["window_size"] == 100


def test_phlash_constructs_vcf_contigs_and_test_data(monkeypatch, tmp_path) -> None:
    train = tmp_path / "train.vcf.gz"
    test = tmp_path / "test.vcf.gz"
    train.write_bytes(b"vcf")
    test.write_bytes(b"vcf")
    contig_calls: list[tuple[str, list[str], str | None]] = []
    fit_calls = []

    def make_contig(path, *, samples, region):
        contig_calls.append((path, samples, region))
        return f"contig:{Path(path).name}"

    def fit(contigs, *, test_data, **options):
        fit_calls.append((contigs, test_data, options))
        return _models()

    fake = SimpleNamespace(
        __version__="1.0.6",
        contig=make_contig,
        fit=fit,
    )
    monkeypatch.setattr(adapter, "_load_phlash", lambda: fake)

    result = phlash(
        [train],
        samples=["sample-a", "sample-b"],
        region="chr1:1-1000",
        test_input=test,
        random_seed=None,
    )

    assert result.uns["phlash_input_kind"] == "vcf"
    assert len(contig_calls) == 2
    assert fit_calls[0][0] == ["contig:train.vcf.gz"]
    assert fit_calls[0][1] == "contig:test.vcf.gz"
    assert fit_calls[0][2]["window_size"] == 100


def test_phlash_holds_out_first_constructed_contig(monkeypatch) -> None:
    fit_calls = []

    def fit(contigs, *, test_data, **options):
        fit_calls.append((contigs, test_data, options))
        return _models()

    fake = SimpleNamespace(__version__="1.0.6", fit=fit)
    monkeypatch.setattr(adapter, "_load_phlash", lambda: fake)
    contigs = [SimpleNamespace(name=name) for name in ("first", "second", "third")]

    phlash(
        contigs,
        input_kind="contig",
        random_seed=None,
        window_size=250,
    )

    assert fit_calls == [([contigs[1], contigs[2]], contigs[0], {"window_size": 250})]


def test_phlash_writes_hashed_json_and_npz(monkeypatch, tmp_path) -> None:
    source = tmp_path / "input.psmcfa"
    source.write_text(">chr1\nTTTT\n", encoding="utf-8")
    fake = SimpleNamespace(
        __version__="1.0.6",
        psmc=lambda paths, **options: _models(),
    )
    monkeypatch.setattr(adapter, "_load_phlash", lambda: fake)

    result = phlash(
        [source],
        random_seed=None,
        output_prefix=tmp_path / "analysis",
    )
    payload = result.results["phlash"]
    json_path = tmp_path / "analysis.phlash.json"
    npz_path = tmp_path / "analysis.phlash.posterior.npz"

    assert json_path.is_file()
    assert npz_path.is_file()
    on_disk = json.loads(json_path.read_text(encoding="utf-8"))
    assert on_disk["provenance"]["artifacts"][0]["kind"] == "posterior_archive"
    assert {artifact["kind"] for artifact in payload["provenance"]["artifacts"]} == {
        "normalized_result",
        "posterior_archive",
    }
    with np.load(npz_path) as archive:
        assert archive["posterior_ne"].shape == (3, 200)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"input_kind": "nonsense"}, "input_kind"),
        ({"grid_size": 1}, "grid_size"),
        ({"credible_level": 1.0}, "credible_level"),
        ({"random_seed": -1}, "random_seed"),
    ],
)
def test_phlash_validates_options(monkeypatch, tmp_path, kwargs, message) -> None:
    source = tmp_path / "input.psmcfa"
    source.write_text(">chr1\nTTTT\n", encoding="utf-8")
    fake = SimpleNamespace(
        __version__="1.0.6",
        psmc=lambda paths, **options: _models(),
    )
    monkeypatch.setattr(adapter, "_load_phlash", lambda: fake)

    with pytest.raises(ValueError, match=message):
        phlash([source], **kwargs)


def test_phlash_rejects_mixed_path_formats(monkeypatch, tmp_path) -> None:
    psmcfa = tmp_path / "input.psmcfa"
    trees = tmp_path / "input.trees"
    psmcfa.touch()
    trees.touch()
    monkeypatch.setattr(
        adapter,
        "_load_phlash",
        lambda: SimpleNamespace(__version__="1.0.6"),
    )

    with pytest.raises(ValueError, match="mixed path formats"):
        phlash([psmcfa, trees])


def test_phlash_rejects_missing_input(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        adapter,
        "_load_phlash",
        lambda: SimpleNamespace(__version__="1.0.6"),
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        phlash([tmp_path / "missing.psmcfa"])


def test_phlash_rejects_native_rewrite_request() -> None:
    with pytest.raises(NotImplementedError, match="external integration"):
        phlash(["a.psmcfa"], implementation="native")
