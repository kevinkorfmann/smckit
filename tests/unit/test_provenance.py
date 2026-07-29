from __future__ import annotations

import json

from smckit._provenance import build_provenance, sha256_file


def test_build_provenance_is_json_serializable(tmp_path) -> None:
    input_path = tmp_path / "input.txt"
    input_path.write_text("smc\n", encoding="utf-8")
    provenance = build_provenance(
        method="psmc",
        implementation_requested="auto",
        implementation_used="native",
        arguments={"n_iterations": 2},
        inputs=[input_path],
        seed=1,
    )
    assert provenance["schema_version"] == "1.0"
    assert provenance["input_sha256"][str(input_path)] == sha256_file(input_path)
    json.dumps(provenance)
