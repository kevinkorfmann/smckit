"""Run only vendored SMC++ vcf2smc without importing its compiled inference module."""

from __future__ import annotations

import argparse
import sys
import types
import warnings
from pathlib import Path

import numpy as np


class _DistributionNotFound(Exception):
    pass


def _prepare_imports(vendor_root: Path) -> None:
    pkg_resources = types.ModuleType("pkg_resources")

    def missing_distribution(name):
        raise _DistributionNotFound(name)

    pkg_resources.get_distribution = missing_distribution
    pkg_resources.DistributionNotFound = _DistributionNotFound
    sys.modules["pkg_resources"] = pkg_resources
    if not hasattr(np, "VisibleDeprecationWarning"):
        np.VisibleDeprecationWarning = DeprecationWarning

    sys.path.insert(0, str(vendor_root))
    import smcpp

    compiled_stub = types.ModuleType("smcpp._smcpp")
    smcpp._smcpp = compiled_stub
    sys.modules["smcpp._smcpp"] = compiled_stub
    commands_package = types.ModuleType("smcpp.commands")
    commands_package.__path__ = [str(vendor_root / "smcpp" / "commands")]
    sys.modules["smcpp.commands"] = commands_package


def main() -> int:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-root", required=True)
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    vendor_root = Path(args.vendor_root).resolve()
    _prepare_imports(vendor_root)

    from smcpp.commands import vcf2smc
    from smcpp.log import init_logging

    init_logging()
    vcf2smc.logger.warn = vcf2smc.logger.warning
    command_parser = argparse.ArgumentParser()
    command = vcf2smc.Vcf2Smc(command_parser)
    namespace = command_parser.parse_args(
        [
            "-d",
            "s1",
            "s1",
            "--length",
            "20",
            args.vcf,
            args.output,
            "chr1",
            "pop:s1,s2",
        ]
    )
    command.main(namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
