"""I/O: read/write PSMC, MSMC, ASMC, SMC++, diCal2 native formats, VCF, tree sequences."""

from smckit.io._asmc import (
    read_asmc,
    read_decoding_quantities,
    read_hap,
    read_map,
    read_samples,
)
from smckit.io._dical2 import (
    read_dical2,
    read_dical2_config,
    read_dical2_demo,
    read_dical2_param,
    read_dical2_rates,
    read_dical2_sequences,
)
from smckit.io._msmc_im import read_msmc_im_output, write_msmc_im_output
from smckit.io._multihetsep import (
    read_msmc_combined_output,
    read_msmc_output,
    read_multihetsep,
)
from smckit.io._psmcfa import read_psmcfa
from smckit.io._psmc_output import read_psmc_output, write_psmc_output
from smckit.io._smcpp_input import read_smcpp_input

__all__ = [
    "read_asmc",
    "read_decoding_quantities",
    "read_dical2",
    "read_dical2_config",
    "read_dical2_demo",
    "read_dical2_param",
    "read_dical2_rates",
    "read_dical2_sequences",
    "read_hap",
    "read_map",
    "read_msmc_combined_output",
    "read_msmc_im_output",
    "read_msmc_output",
    "read_multihetsep",
    "read_psmcfa",
    "read_psmc_output",
    "read_samples",
    "read_smcpp_input",
    "write_msmc_im_output",
    "write_psmc_output",
]
