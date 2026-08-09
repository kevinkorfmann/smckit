"""Typed PSMC+ option, normalization, artifact, and provenance tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from smckit._core import SmcData
from smckit._provenance import sha256_file
from smckit.tl import PSMCPlusOptions, psmcplus
from smckit.tl import _psmcplus as module
from smckit.upstream._run import UpstreamRunResult

ROOT = Path(__file__).resolve().parents[2]
FIT_FIXTURE = ROOT / "tests/data/psmcplus/constpop_D4_1iter.final_parameters.txt"


def _path_backed_data(tmp_path: Path, *, count: int = 1) -> SmcData:
    paths = []
    for index in range(count):
        path = tmp_path / f"input-{index}.multihetsep"
        path.write_text("chr1\t1\t1\tAT\n", encoding="utf-8")
        paths.append(str(path))
    return SmcData(uns={"source_paths": paths})


def _artifact(path: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(output_dir)),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _result(
    output_dir: Path,
    artifacts: list[dict[str, Any]],
    *,
    returncode: int = 0,
    stderr: str = "",
) -> UpstreamRunResult:
    return UpstreamRunResult(
        tool="psmcplus",
        command=["python", "PSMCplus.py"],
        cwd=str(output_dir),
        returncode=returncode,
        stdout="typed upstream test",
        stderr=stderr,
        runtime_seconds=0.25,
        artifacts=artifacts,
        compatibility_patches=["Restore numpy.math; vendored source unchanged."],
    )


def _ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "method_upstream_available", lambda method: True)


def test_all_original_inference_controls_are_forwarded(tmp_path: Path) -> None:
    options = PSMCPlusOptions(
        mode="decode",
        number_time_windows=8,
        spread_1=0.2,
        spread_2=40,
        bin_size=50,
        scaled_recombination_rate=0.0004,
        scaled_mutation_rate=0.001,
        rho_fixed=True,
        mutation_recombination_ratio=2,
        lambda_lower_bound=0.2,
        lambda_upper_bound=20,
        recombination_map_downsamples=17,
        iterations=3,
        likelihood_threshold=0.2,
        lambda_initial=[1, 2],
        lambda_segments="1*4,1*4",
        parameter_tolerance=1e-5,
        objective_tolerance=2e-5,
        nonexponential_recombination=True,
        midpoint_transitions=True,
        midpoint_emissions=True,
        final_time_factor=3,
        optimization_method="Nelder-Mead",
        save_iteration_files=True,
        decode_downsample=7,
        cores=2,
    )
    args = module._upstream_args(
        options,
        inputs=[tmp_path / "a.mhs"],
        mutation_maps=[tmp_path / "a.M.bed"],
        recombination_maps=[tmp_path / "a.R.bed"],
        output_path=tmp_path / "posterior.txt",
        marginal_recombination_path=tmp_path / "recomb.txt",
    )

    for flag in [
        "-in",
        "-in_M",
        "-in_R",
        "-o",
        "-D",
        "-spread_1",
        "-spread_2",
        "-b",
        "-rho",
        "-theta",
        "-rho_fixed",
        "-mu_over_rho_ratio",
        "-lambda_lwr",
        "-lambda_upr",
        "-number_downsamples_R",
        "-its",
        "-thresh",
        "-lambda_A_fg",
        "-lambda_A_segments",
        "-xtol",
        "-ftol",
        "-recombnoexp",
        "-midpoint_transitions",
        "-midpoint_emissions",
        "-final_T_factor",
        "-optimisation_method",
        "-save_iteration_files",
        "-decode",
        "-decode_downsample",
        "-o_R",
        "-c",
    ]:
        assert flag in args
    assert args[args.index("-lambda_A_fg") + 1] == "1.0,2.0"
    assert args[args.index("-midpoint_transitions") + 1] == "True"


def test_typed_fit_normalizes_scales_artifacts_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _path_backed_data(tmp_path)
    captured: dict[str, Any] = {}

    def fake_run(tool, args, *, output_dir, timeout, env):
        captured.update(tool=tool, args=args, timeout=timeout, env=env)
        output_dir = Path(output_dir)
        prefix = Path(args[args.index("-o") + 1])
        final = Path(f"{prefix}final_parameters.txt")
        shutil.copy2(FIT_FIXTURE, final)
        return _result(output_dir, [_artifact(final, output_dir)])

    _ready(monkeypatch)
    monkeypatch.setattr(module.upstream, "run", fake_run)
    output_prefix = tmp_path / "persisted/result_"
    result_data = psmcplus(
        data,
        options=PSMCPlusOptions(number_time_windows=4, iterations=1, cores=1),
        mutation_rate=1e-8,
        generation_time=25,
        output_prefix=output_prefix,
        implementation="auto",
        timeout=30,
    )

    result = result_data.results["psmcplus"]
    expected = np.loadtxt(FIT_FIXTURE)
    assert result_data is data
    assert result["implementation_requested"] == "auto"
    assert result["implementation"] == "upstream"
    assert result["mode"] == "fit"
    assert result["theta"] == pytest.approx(0.001140751755406813)
    assert result["rho"] == pytest.approx(0.0007821311791495348)
    assert result["log_likelihood"] == pytest.approx(-3556.760710596285)
    np.testing.assert_allclose(result["time"], expected[:, 0] / 1e-8 * 25)
    np.testing.assert_allclose(result["ne"], 1.0 / expected[:, 2] / 1e-8)
    np.testing.assert_allclose(
        result["lambda"],
        expected[:, 2] * result["theta"] / 4.0,
    )
    assert result["time_units"] == "years"
    assert result["ne_units"] == "individuals"
    assert data.log_likelihood("psmcplus") == pytest.approx(result["log_likelihood"])
    assert data.effective_population_size("psmcplus").shape == (4,)

    persisted = Path(f"{output_prefix}final_parameters.txt")
    assert persisted.read_bytes() == FIT_FIXTURE.read_bytes()
    assert result["artifacts"][0]["persisted"] is True
    assert result["artifacts"][0]["path"] == str(persisted)
    assert result["provenance"]["artifacts"] == result["artifacts"]
    assert result["provenance"]["runtime_seconds"] == pytest.approx(0.25)
    assert "numpy.math" in result["provenance"]["warnings"][0]
    json.dumps(result["provenance"])
    assert captured["tool"] == "psmcplus"
    assert captured["timeout"] == 30
    assert set(captured["env"].values()) == {"1"}


def test_typed_decode_normalizes_posterior_and_marginal_recombination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _path_backed_data(tmp_path)

    def fake_run(tool, args, *, output_dir, timeout, env):
        output_dir = Path(output_dir)
        posterior = Path(args[args.index("-o") + 1])
        posterior.write_text(
            "# theta=4*N_E*mu = 0.001\n"
            "# rho=4*N_E*r = 0.0005\n"
            "# bin_size = 100\n"
            "# first row is position\n"
            "# log likelihood is -12.5\n"
            "# 0.0,1.0,2.0\n"
            "0 100 200\n"
            "0.8 0.2 0.5\n"
            "0.2 0.8 0.5\n",
            encoding="utf-8",
        )
        marginal = Path(args[args.index("-o_R") + 1])
        marginal.write_text(
            "# marginal recombination\n0 100\n0.1 0.2\n0.9 0.8\n",
            encoding="utf-8",
        )
        return _result(
            output_dir,
            [_artifact(marginal, output_dir), _artifact(posterior, output_dir)],
        )

    _ready(monkeypatch)
    monkeypatch.setattr(module.upstream, "run", fake_run)
    posterior_path = tmp_path / "outputs/posterior.txt"
    marginal_path = tmp_path / "outputs/recombination.txt"
    psmcplus(
        data,
        options=PSMCPlusOptions(mode="decode", number_time_windows=2, cores=1),
        mutation_rate=1e-8,
        generation_time=20,
        output_prefix=posterior_path,
        marginal_recombination_path=marginal_path,
        implementation="upstream",
    )

    result = data.results["psmcplus"]
    np.testing.assert_array_equal(result["position"], [0, 100, 200])
    np.testing.assert_allclose(
        result["posterior"],
        [[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]],
    )
    np.testing.assert_allclose(result["time_boundaries"], [0, 1e6, 2e6])
    np.testing.assert_allclose(result["time"], [5e5, 1.5e6])
    np.testing.assert_allclose(result["posterior_mean_time"], [7e5, 1.3e6, 1e6])
    marginal = result["marginal_recombination"]
    np.testing.assert_allclose(marginal["recombination_probability"], [0.1, 0.2])
    assert posterior_path.is_file()
    assert marginal_path.is_file()
    assert {artifact["kind"] for artifact in result["artifacts"]} == {
        "marginal_recombination",
        "posterior_decoding",
    }


@pytest.mark.parametrize(
    "options",
    [
        PSMCPlusOptions(number_time_windows=1),
        PSMCPlusOptions(bin_size=0),
        PSMCPlusOptions(scaled_mutation_rate=-1),
        PSMCPlusOptions(lambda_lower_bound=2, lambda_upper_bound=1),
        PSMCPlusOptions(final_time_factor=0),
        PSMCPlusOptions(cores=0),
    ],
)
def test_invalid_options_fail_before_execution(options: PSMCPlusOptions) -> None:
    with pytest.raises(ValueError):
        options.validate()


def test_native_mode_is_explicitly_unavailable() -> None:
    with pytest.raises(NotImplementedError, match=r"native PSMC\+"):
        psmcplus(SmcData(), implementation="native")


def test_path_backing_and_map_cardinality_are_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ready(monkeypatch)
    with pytest.raises(ValueError, match="path-backed"):
        psmcplus(SmcData(), implementation="upstream")

    data = _path_backed_data(tmp_path, count=2)
    map_path = tmp_path / "map.bed"
    map_path.write_text("chr1\t0\t1\t1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one mutation map"):
        psmcplus(
            data,
            mutation_map_paths=[map_path],
            implementation="upstream",
        )
    with pytest.raises(ValueError, match="exactly one"):
        psmcplus(
            data,
            options=PSMCPlusOptions(mode="decode"),
            implementation="upstream",
        )


def test_nonzero_original_exit_is_not_silently_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _path_backed_data(tmp_path)

    def fake_run(tool, args, *, output_dir, timeout, env):
        return _result(Path(output_dir), [], returncode=2, stderr="upstream failed")

    _ready(monkeypatch)
    monkeypatch.setattr(module.upstream, "run", fake_run)
    with pytest.raises(RuntimeError, match="status 2: upstream failed"):
        psmcplus(data, implementation="upstream")
