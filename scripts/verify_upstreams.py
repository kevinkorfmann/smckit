"""Verify immutable upstream checkouts recorded in the preservation manifest."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "preservation" / "upstreams.json"


def _is_gitlink(path: Path) -> bool | None:
    """Return gitlink status, or ``None`` when *path* is absent from the index."""
    relative = path.relative_to(ROOT)
    records = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--stage", "--", str(relative)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if not records:
        return None
    return any(
        record.startswith("160000 ") and record.split("\t", 1)[-1] == str(relative)
        for record in records
    )


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures: list[str] = []
    tracked_snapshots: list[str] = []
    for name, entry in manifest["tools"].items():
        expected = entry.get("commit")
        if expected is None or "path" not in entry:
            continue
        path = ROOT / entry["path"]
        if not path.exists():
            failures.append(f"{name}: missing {path.relative_to(ROOT)}")
            continue
        is_gitlink = _is_gitlink(path)
        if is_gitlink is None:
            failures.append(f"{name}: {path.relative_to(ROOT)} is not tracked by git")
            continue
        if not is_gitlink:
            tracked_snapshots.append(name)
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
    print("All immutable upstream gitlink pins match preservation/upstreams.json")
    if tracked_snapshots:
        names = ", ".join(sorted(tracked_snapshots))
        print(
            "Tracked upstream snapshots use the enclosing smckit commit and recorded "
            f"artifact checksums for integrity: {names}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
