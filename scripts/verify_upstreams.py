"""Verify immutable submodule pins in the preservation manifest."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "preservation" / "upstreams.json"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures: list[str] = []
    for name, entry in manifest["tools"].items():
        expected = entry.get("commit")
        if expected is None or "path" not in entry:
            continue
        path = ROOT / entry["path"]
        if not path.exists():
            failures.append(f"{name}: missing {path.relative_to(ROOT)}")
            continue
        actual = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual != expected:
            failures.append(f"{name}: expected {expected}, found {actual}")
    if failures:
        raise SystemExit("\n".join(failures))
    print("All immutable upstream pins match preservation/upstreams.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
