"""Preprocessing: sequence data to SMC input formats."""

from smckit.pp._psmc import psmcfa_from_consensus
from smckit.pp._smcpp import smcpp_from_vcf

__all__ = ["psmcfa_from_consensus", "smcpp_from_vcf"]
