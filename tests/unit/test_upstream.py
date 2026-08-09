"""Tests for upstream registry readiness surfaces."""

from __future__ import annotations

import sys

import pytest

import smckit
from smckit.tl._implementation import method_upstream_available, standard_upstream_metadata
from smckit.upstream import _run


def test_upstream_status_reports_known_tools() -> None:
    status = smckit.upstream.status()
    for tool in [
        "psmc",
        "psmcplus",
        "msmc2",
        "msmc_im",
        "smcpp",
        "esmc2",
        "asmc",
        "dical2",
    ]:
        assert tool in status
        assert "ready" in status[tool]
        assert "missing" in status[tool]
        assert "install_help" in status[tool]


def test_smcpp_status_reports_vendored_source_tree() -> None:
    status = smckit.upstream.status("smcpp")
    assert status["public_upstream"] is True
    assert status["vendor_path"] is not None
    assert status["source_present"] is True


def test_standard_upstream_metadata_includes_registry_fields() -> None:
    metadata = standard_upstream_metadata("esmc2", effective_args={"n_states": 6})
    assert metadata["tool"] == "esmc2"
    assert "runtime" in metadata
    assert metadata["effective_args"]["n_states"] == 6


def test_public_registry_entries_report_boolean_readiness() -> None:
    assert isinstance(method_upstream_available("psmc"), bool)
    assert smckit.upstream.status("dical2")["public_upstream"] is True
    assert smckit.upstream.status("dical2")["version"] == "2.0.5"


def test_install_help_is_available_for_known_tools() -> None:
    help_text = smckit.upstream.install_help("psmc")
    assert isinstance(help_text, str)
    assert help_text
    assert "smckit[psmc]" in help_text


def test_raw_runner_is_shell_free_and_captures_artifacts(tmp_path, monkeypatch) -> None:
    script = tmp_path / "runner.py"
    script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "Path('artifact.txt').write_text(Path(sys.argv[1]).read_text())\n"
        "print('captured')\n",
        encoding="utf-8",
    )
    source = tmp_path / "input.txt"
    source.write_text("payload", encoding="utf-8")
    output = tmp_path / "output"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_run, "command_prefix", lambda tool: [sys.executable, str(script)])

    result = _run.run("psmc", ["input.txt", "; touch injected"], output_dir=output)

    assert result.returncode == 0
    assert result.stdout.strip() == "captured"
    assert not (output / "injected").exists()
    assert result.artifacts[0]["path"] == "artifact.txt"
    assert result.compatibility_patches == []


def test_raw_runner_returns_124_on_timeout(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        _run,
        "command_prefix",
        lambda tool: [sys.executable, "-c", "import time; time.sleep(2)"],
    )

    result = _run.run("psmc", [], output_dir=tmp_path / "output", timeout=0.01)

    assert result.returncode == 124
    assert "exceeded timeout" in result.stderr


def test_psmc_raw_runner_rejects_unknown_entrypoint() -> None:
    with pytest.raises(ValueError, match="Unknown PSMC entry point"):
        _run.command_prefix("psmc", "not-a-real-helper")


def test_psmcplus_raw_runner_exposes_both_original_entrypoints(monkeypatch) -> None:
    status = smckit.upstream.status("psmcplus")
    status["runtime"] = {**status["runtime"], "path": sys.executable}

    class ReadyPSMCPlus:
        @staticmethod
        def status():
            return status

    monkeypatch.setattr(_run, "get_tool", lambda tool: ReadyPSMCPlus())
    inference = _run.command_prefix("psmcplus")
    simulation = _run.command_prefix("psmcplus", "simulate_HMM.py")

    assert inference[-1].endswith("vendor/PSMCplus/PSMCplus.py")
    assert simulation[-1].endswith("vendor/PSMCplus/simulate_HMM.py")
    assert inference[-2].endswith("_psmcplus_runner.py")


def test_psmcplus_raw_runner_rejects_unknown_entrypoint() -> None:
    with pytest.raises(ValueError, match=r"Unknown PSMC\+ entry point"):
        _run.command_prefix("psmcplus", "not-a-real-helper")


def test_psmcplus_compatibility_policy_is_explicit() -> None:
    patches = _run._compatibility_patches("psmcplus")

    assert len(patches) == 1
    assert "numpy.math" in patches[0]
    assert "vendored source unchanged" in patches[0]


def test_raw_esmc2_runner_exposes_bootstrapped_r_library(
    tmp_path,
    monkeypatch,
) -> None:
    script = tmp_path / "runner.py"
    script.write_text(
        "import os\nprint(os.environ.get('R_LIBS_USER', ''))\n",
        encoding="utf-8",
    )
    r_library = tmp_path / "r-library"
    r_library.mkdir()

    class Tool:
        cache_path = r_library

    monkeypatch.setattr(_run, "get_tool", lambda tool: Tool())
    monkeypatch.setattr(
        _run,
        "command_prefix",
        lambda tool: [sys.executable, str(script)],
    )
    result = _run.run("esmc2", [], output_dir=tmp_path / "output")

    assert result.stdout.strip() == str(r_library)
