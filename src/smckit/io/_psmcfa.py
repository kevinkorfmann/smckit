"""Read PSMCFA format files."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from smckit._core import SmcData

# Mapping from ASCII characters to PSMC codes:
# 0 = homozygous (T, A, C, G, 0)
# 1 = heterozygous (K, M, R, S, W, Y, 1)
# 2 = missing (N, everything else)
_CONV_TABLE = np.full(256, 2, dtype=np.int8)
for _c in b"0ACGTacgt":
    _CONV_TABLE[_c] = 0
for _c in b"1KMRSWYkmrswy":
    _CONV_TABLE[_c] = 1


def read_psmcfa(path: str | Path) -> SmcData:
    """Read a PSMCFA file into a SmcData object.

    Parameters
    ----------
    path : str or Path
        Path to ``.psmcfa`` or ``.psmcfa.gz`` file.

    Returns
    -------
    SmcData
        Data container with sequences loaded. Each FASTA record becomes
        a separate entry in ``uns["sequences"]`` (list of named arrays).
        The concatenated sequence (excluding missing) is in ``sequences``.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" or str(path).endswith(".psmcfa.gz") else open

    records: list[dict] = []
    current_name: str | None = None
    current_seq: list[bytes] = []

    def _flush():
        if current_name is not None and current_seq:
            raw = b"".join(current_seq)
            codes = _CONV_TABLE[np.frombuffer(raw, dtype=np.uint8)]
            callable_mask = codes < 2
            records.append({
                "name": current_name,
                "codes": codes,
                "L": len(codes),
                "L_e": int(callable_mask.sum()),
                "n_e": int((codes == 1).sum()),
            })

    with opener(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(b">"):
                _flush()
                current_name = line[1:].decode().split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    _flush()

    if not records:
        raise ValueError(f"No sequences found in {path}")

    # Build SmcData
    sum_L = sum(r["L_e"] for r in records)
    sum_n = sum(r["n_e"] for r in records)

    data = SmcData()
    data.uns["records"] = records
    data.uns["sum_L"] = sum_L
    data.uns["sum_n"] = sum_n
    data.uns["n_seqs"] = len(records)

    return data
