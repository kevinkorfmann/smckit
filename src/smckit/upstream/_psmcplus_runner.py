"""Execute pinned PSMC+ entry points with narrow runtime compatibility fixes."""

from __future__ import annotations

import math
import runpy
import sys
from pathlib import Path
from typing import NoReturn


def main(argv: list[str] | None = None) -> NoReturn:
    """Run one vendored PSMC+ script without modifying the upstream source."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        raise SystemExit("usage: _psmcplus_runner.py ENTRYPOINT [ORIGINAL ARGUMENTS ...]")

    script = Path(arguments.pop(0)).resolve()
    if script.name not in {"PSMCplus.py", "simulate_HMM.py"}:
        raise SystemExit(f"unsupported PSMC+ entry point: {script.name}")
    if not script.is_file():
        raise SystemExit(f"PSMC+ entry point does not exist: {script}")

    import numpy as np

    if not hasattr(np, "math"):
        np.math = math  # type: ignore[attr-defined]

    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *arguments]
    runpy.run_path(str(script), run_name="__main__")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
