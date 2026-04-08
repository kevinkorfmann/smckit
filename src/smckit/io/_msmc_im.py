"""Read/write MSMC-IM estimates files."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_msmc_im_output(path: str | Path) -> dict[str, np.ndarray]:
    """Read MSMC-IM ``.estimates.txt`` output file.

    The estimates file is tab-separated with columns:

    - left_time_boundary (float): time in generations
    - im_N1 (float): effective population size of pop 1
    - im_N2 (float): effective population size of pop 2
    - m (float): symmetric migration rate
    - M (float): cumulative migration probability

    Parameters
    ----------
    path : str or Path
        Path to MSMC-IM ``.estimates.txt`` file.

    Returns
    -------
    dict
        Dictionary with keys ``"left_boundary"``, ``"N1"``, ``"N2"``,
        ``"m"``, and ``"M"``, each containing a NumPy array.
    """
    path = Path(path)

    left_boundaries: list[float] = []
    n1_list: list[float] = []
    n2_list: list[float] = []
    m_list: list[float] = []
    M_list: list[float] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("left"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            left_boundaries.append(float(parts[0]))
            n1_list.append(float(parts[1]))
            n2_list.append(float(parts[2]))
            m_list.append(float(parts[3]))
            M_list.append(float(parts[4]))

    if not left_boundaries:
        raise ValueError(f"No data found in {path}")

    return {
        "left_boundary": np.array(left_boundaries, dtype=np.float64),
        "N1": np.array(n1_list, dtype=np.float64),
        "N2": np.array(n2_list, dtype=np.float64),
        "m": np.array(m_list, dtype=np.float64),
        "M": np.array(M_list, dtype=np.float64),
    }


def write_msmc_im_output(
    path: str | Path,
    left_boundary: np.ndarray,
    N1: np.ndarray,
    N2: np.ndarray,
    m: np.ndarray,
    M: np.ndarray,
) -> None:
    """Write MSMC-IM estimates to a tab-separated file.

    Parameters
    ----------
    path : str or Path
        Output file path.
    left_boundary : np.ndarray
        Left time boundaries in generations.
    N1 : np.ndarray
        Effective population size of population 1.
    N2 : np.ndarray
        Effective population size of population 2.
    m : np.ndarray
        Symmetric migration rate per generation.
    M : np.ndarray
        Cumulative migration probability.
    """
    path = Path(path)

    with open(path, "w") as f:
        f.write("left_time_boundary\tim_N1\tim_N2\tm\tM\n")
        for i in range(len(left_boundary)):
            f.write(
                f"{left_boundary[i]}\t{N1[i]}\t{N2[i]}\t{m[i]}\t{M[i]}\n"
            )
