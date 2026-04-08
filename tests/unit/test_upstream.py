"""Tests for upstream registry readiness surfaces."""

from __future__ import annotations

import smckit
from smckit.tl._implementation import method_upstream_available, standard_upstream_metadata


def test_upstream_status_reports_known_tools() -> None:
    status = smckit.upstream.status()
    for tool in ["psmc", "msmc2", "msmc_im", "smcpp", "esmc2", "asmc", "dical2"]:
        assert tool in status
        assert "ready" in status[tool]
        assert "missing" in status[tool]


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
