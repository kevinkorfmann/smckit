"""Tests for stable SMC++ model/result serialization."""

from __future__ import annotations

import copy
import importlib
import json

import numpy as np
import pytest

from smckit._core import SmcData
from smckit.io import read_smcpp_model, smcpp_model_payload, write_smcpp_model
from smckit.tl import smcpp


def _small_data() -> SmcData:
    return SmcData(
        uns={
            "records": [
                {
                    "name": "chr1",
                    "observations": [
                        (2_000, 0, 0),
                        (1, 1, 2),
                        (2_000, 0, 0),
                        (1, 0, 1),
                        (2_000, 0, 0),
                    ],
                }
            ],
            "n_undist": 4,
            "n_distinguished": 2,
            "n_populations": 1,
        }
    )


def _two_population_data() -> SmcData:
    return SmcData(
        uns={
            "n_populations": 2,
            "populations": ["pop-a", "pop-b"],
            "pids": ["pop-a", "pop-b"],
            "joint_observations": [
                (100, ((0, 0, 4), (0, 0, 4))),
                (1, ((1, 1, 4), (0, 1, 4))),
                (100, ((0, 0, 4), (0, 0, 4))),
            ],
            "smcpp_header": {
                "pids": ["pop-a", "pop-b"],
                "dist": [[], []],
                "undist": [
                    [["pop-a", index] for index in range(4)],
                    [["pop-b", index] for index in range(4)],
                ],
            },
        }
    )


def test_smcpp_output_prefix_writes_reloadable_artifacts(tmp_path) -> None:
    data = smcpp(
        _small_data(),
        n_intervals=3,
        max_iterations=1,
        seed=23,
        implementation="native",
        output_prefix=tmp_path / "analysis",
    )
    model_path = tmp_path / "analysis.smcpp.model.json"
    result_path = tmp_path / "analysis.smcpp.json"

    assert model_path.is_file()
    assert result_path.is_file()
    model = read_smcpp_model(model_path)
    assert model["schema_version"] == 1
    assert model["model"]["spline_class"] == "Piecewise"
    assert model["hidden_states"] == {"ALL": [0.0, float("inf")]}
    assert len(model["model"]["knots"]) == len(data.results["smcpp"]["time"])
    assert {
        artifact["kind"]
        for artifact in data.results["smcpp"]["provenance"]["artifacts"]
    } == {"model", "normalized_result"}
    on_disk = json.loads(result_path.read_text(encoding="utf-8"))
    assert on_disk["provenance"]["artifacts"][0]["kind"] == "model"


def test_serialized_model_can_initialize_native_fit(tmp_path) -> None:
    baseline = smcpp(
        _small_data(),
        n_intervals=3,
        max_iterations=1,
        seed=11,
        implementation="native",
    )
    model_path = write_smcpp_model(baseline, tmp_path / "model.json")
    initialized = smcpp(
        copy.deepcopy(_small_data()),
        n_intervals=3,
        max_iterations=1,
        seed=99,
        implementation="native",
        initial_model=model_path,
    )

    result = initialized.results["smcpp"]
    assert result["initial_model_used"] is True
    assert np.all(np.isfinite(result["ne"]))
    assert np.all(result["ne"] > 0)


def test_model_payload_accepts_result_mapping() -> None:
    result = smcpp(
        _small_data(),
        n_intervals=3,
        max_iterations=1,
        seed=5,
        implementation="native",
    ).results["smcpp"]
    payload = smcpp_model_payload(result, population="pop-a")
    assert payload["model"]["pid"] == "pop-a"
    assert payload["theta"] == result["theta"]


def test_model_reader_rejects_invalid_knots(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "model": {
                    "class": "SMCModel",
                    "knots": [2.0, 1.0],
                    "N0": 10_000,
                    "spline_class": "Piecewise",
                    "y": [0.0, 0.0],
                    "pid": "ALL",
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="increase"):
        read_smcpp_model(path)


def test_native_split_failure_is_explicit() -> None:
    data = _small_data()
    data.uns["n_populations"] = 2
    with pytest.raises(NotImplementedError, match="split inference"):
        smcpp(data, implementation="native")


def test_upstream_split_routes_two_population_data(monkeypatch) -> None:
    module = importlib.import_module("smckit.tl._smcpp")
    data = _two_population_data()
    calls = {}

    monkeypatch.setattr(module, "choose_implementation", lambda *args, **kwargs: "upstream")

    def fake_split(current, **kwargs):
        calls.update(kwargs)
        current.results["smcpp"] = {"analysis": "split", "implementation": "upstream"}
        return current

    monkeypatch.setattr(module, "_smcpp_upstream_split", fake_split)
    result = smcpp(
        data,
        implementation="upstream",
        split_models=({"model": "one"}, {"model": "two"}),
    )

    assert result.results["smcpp"]["analysis"] == "split"
    assert calls["marginal_models"] == ({"model": "one"}, {"model": "two"})


def test_upstream_split_requires_two_marginal_models(monkeypatch) -> None:
    module = importlib.import_module("smckit.tl._smcpp")
    monkeypatch.setattr(module, "choose_implementation", lambda *args, **kwargs: "upstream")

    with pytest.raises(ValueError, match="split_models"):
        smcpp(_two_population_data(), implementation="upstream")
