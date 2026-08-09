"""Tests for the content-addressed publication protocol."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "workflow" / "publication" / "scripts" / "freeze_protocol.py"
CONFIG = ROOT / "workflow" / "publication" / "config.yaml"
SPEC = importlib.util.spec_from_file_location("smckit_publication_protocol", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_protocol_is_valid_and_deterministic(tmp_path) -> None:
    first = MODULE.freeze_protocol(CONFIG, tmp_path / "first.json")
    second = MODULE.freeze_protocol(CONFIG, tmp_path / "second.json")
    assert first == second
    assert first["protocol_id"].startswith("sha256:")
    assert first["config"]["replicates"] == 20


def test_protocol_refuses_to_overwrite_immutable_record(tmp_path) -> None:
    output = tmp_path / "protocol.json"
    output.write_text("original\n")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        MODULE.freeze_protocol(CONFIG, output)

    assert output.read_text() == "original\n"


def test_protocol_rejects_missing_scenario(tmp_path) -> None:
    config = yaml.safe_load(CONFIG.read_text())
    config["scenarios"].remove("structure")
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="simulation scenario"):
        MODULE.freeze_protocol(invalid, tmp_path / "protocol.json")


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("replicates", 19, "at least 20"),
        ("timing_repetitions", 4, "at least five"),
        ("sequence_length", 0, "positive"),
    ],
)
def test_protocol_rejects_weakened_acceptance_gates(key, value, message) -> None:
    config = yaml.safe_load(CONFIG.read_text())
    config[key] = value
    with pytest.raises(ValueError, match=message):
        MODULE.validate_protocol(config)
