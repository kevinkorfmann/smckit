"""I/O: read/write PSMC, MSMC, SMC++ native formats, VCF, tree sequences."""

from smckit.io._psmcfa import read_psmcfa
from smckit.io._psmc_output import read_psmc_output, write_psmc_output

__all__ = ["read_psmcfa", "read_psmc_output", "write_psmc_output"]
