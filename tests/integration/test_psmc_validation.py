"""Validate smckit PSMC against original C implementation on real 1000G data.

Uses NA12878 chr22 from the 1000 Genomes high-coverage dataset.
The reference .psmc file was generated with:
    psmc -N 10 -t 15 -r 5 -p "4+5*3+4" NA12878_chr22.psmcfa
"""

from pathlib import Path

import numpy as np
import pytest

from smckit.io import read_psmcfa, read_psmc_output
from smckit.tl._psmc import psmc

DATA_DIR = Path(__file__).parent.parent / "data"
PSMCFA = DATA_DIR / "NA12878_chr22.psmcfa"
PSMC_REF = DATA_DIR / "NA12878_chr22.psmc"


@pytest.mark.skipif(not PSMCFA.exists(), reason="Validation data not available")
@pytest.mark.slow
def test_psmc_matches_original():
    """smckit PSMC lambda values should correlate > 0.99 with original C PSMC."""
    orig_rounds = read_psmc_output(PSMC_REF)
    final_orig = orig_rounds[-1]

    data = read_psmcfa(PSMCFA)
    data = psmc(
        data,
        pattern="4+5*3+4",
        n_iterations=10,
        max_t=15.0,
        tr_ratio=5.0,
        seed=42,
    )

    res = data.results["psmc"]

    # Lambda correlation should be very high
    corr = np.corrcoef(final_orig["lambda"], res["lambda"])[0, 1]
    assert corr > 0.99, f"Lambda correlation {corr:.4f} < 0.99"

    # Theta should be within 5% of original
    rel_diff_theta = abs(res["theta"] - final_orig["theta"]) / final_orig["theta"]
    assert rel_diff_theta < 0.05, f"Theta relative diff {rel_diff_theta:.4f} > 0.05"

    # Rho should be within 5% of original
    rel_diff_rho = abs(res["rho"] - final_orig["rho"]) / final_orig["rho"]
    assert rel_diff_rho < 0.05, f"Rho relative diff {rel_diff_rho:.4f} > 0.05"
