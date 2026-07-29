from __future__ import annotations

import json

from smckit.cli import main


def test_methods_command_reports_capabilities(capsys) -> None:
    assert main(["methods", "psmc"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["method"] == "psmc"
    assert payload["native"]["default_eligible"] is True


def test_status_command_reports_upstream_registry(capsys) -> None:
    assert main(["status", "dical2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tool"] == "dical2"
    assert "runtime_ready" in payload


def test_upstream_command_uses_shell_free_runner(monkeypatch, tmp_path, capsys) -> None:
    class Result:
        returncode = 7

        @staticmethod
        def to_dict():
            return {"returncode": 7, "command": ["fake"]}

    recorded = {}

    def fake_run(tool, args, *, output_dir, timeout):
        recorded.update(
            tool=tool,
            args=args,
            output_dir=output_dir,
            timeout=timeout,
        )
        return Result()

    monkeypatch.setattr("smckit.cli.smckit.upstream.run", fake_run)
    code = main(
        [
            "upstream",
            "psmc",
            "--output-dir",
            str(tmp_path),
            "--",
            "-N",
            "10",
        ]
    )
    assert code == 7
    assert recorded["tool"] == "psmc"
    assert recorded["args"] == ["-N", "10"]
    assert json.loads(capsys.readouterr().out)["returncode"] == 7


def test_upstream_command_selects_original_helper_entrypoint(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    class Result:
        returncode = 0

        @staticmethod
        def to_dict():
            return {"returncode": 0, "command": ["fq2psmcfa"]}

    recorded = {}

    def fake_run(tool, args, *, output_dir, timeout, entrypoint):
        recorded.update(
            tool=tool,
            args=args,
            output_dir=output_dir,
            timeout=timeout,
            entrypoint=entrypoint,
        )
        return Result()

    monkeypatch.setattr("smckit.cli.smckit.upstream.run", fake_run)
    code = main(
        [
            "upstream",
            "psmc",
            "--output-dir",
            str(tmp_path),
            "--entrypoint",
            "fq2psmcfa",
            "--",
            "-q20",
            "sample.fq.gz",
        ]
    )

    assert code == 0
    assert recorded["entrypoint"] == "fq2psmcfa"
    assert recorded["args"] == ["-q20", "sample.fq.gz"]
    assert json.loads(capsys.readouterr().out)["returncode"] == 0
