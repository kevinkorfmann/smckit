"""PSMC preprocessing compatible with the original ``fq2psmcfa`` utility."""

from __future__ import annotations

import gzip
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np

from smckit._core import SmcData
from smckit.io._psmcfa import write_psmcfa

_NT16 = {
    "A": 1,
    "B": 14,
    "C": 2,
    "D": 13,
    "G": 4,
    "H": 11,
    "K": 12,
    "M": 3,
    "N": 15,
    "R": 5,
    "S": 6,
    "T": 8,
    "V": 7,
    "W": 9,
    "X": 0,
    "Y": 10,
}
_BIT_COUNTS = np.array([4, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4])
_MUTATION_FILTERS = {None, "transversions", "transitions", "cpg", "exclude_cpg"}
_PSEUDOAUTOSOMAL_HG18 = ((0, 2_709_520), (154_584_236, 154_913_754))


def _open_text(path: Path):
    return (
        gzip.open(path, "rt", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open("rt", encoding="utf-8")
    )


def _sequence_records(path: Path) -> Iterator[tuple[str, str, str | None]]:
    with _open_text(path) as handle:
        first = handle.readline()
        if not first:
            return
        if first.startswith(">"):
            name = first[1:].strip().split()[0]
            chunks: list[str] = []
            for line in handle:
                if line.startswith(">"):
                    yield name, "".join(chunks), None
                    name = line[1:].strip().split()[0]
                    chunks = []
                else:
                    chunks.append(line.strip())
            yield name, "".join(chunks), None
            return
        if not first.startswith("@"):
            raise ValueError(f"{path} is not FASTA or FASTQ.")

        header: str | None = first
        while header is not None:
            name = header[1:].strip().split()[0]
            sequence_chunks: list[str] = []
            for line in handle:
                if line.startswith("+"):
                    break
                sequence_chunks.append(line.strip())
            else:
                raise ValueError(f"Truncated FASTQ sequence for {name!r}.")
            sequence = "".join(sequence_chunks)
            quality_chunks: list[str] = []
            quality_length = 0
            while quality_length < len(sequence):
                line = handle.readline()
                if not line:
                    raise ValueError(f"Truncated FASTQ quality string for {name!r}.")
                value = line.rstrip("\r\n")
                quality_chunks.append(value)
                quality_length += len(value)
            quality = "".join(quality_chunks)
            if len(quality) != len(sequence):
                raise ValueError(f"FASTQ sequence and quality lengths differ for {name!r}.")
            yield name, sequence, quality
            header = handle.readline() or None
            if header is not None and not header.startswith("@"):
                raise ValueError(f"Expected a FASTQ header after {name!r}.")


def _mutation_mask(codes: np.ndarray, mutation_filter: str | None) -> np.ndarray:
    masked = np.zeros(len(codes), dtype=bool)
    if mutation_filter is None:
        return masked
    if mutation_filter == "transversions":
        return np.isin(codes, [5, 10])
    if mutation_filter == "transitions":
        masked |= np.isin(codes, [3, 6, 9, 12])
        for index in range(1, len(codes)):
            if codes[index] in (4, 5) and codes[index - 1] in (2, 10):
                masked[index - 1 : index + 1] = True
        return masked
    if mutation_filter == "cpg":
        masked |= np.isin(codes, [3, 6, 9, 12])
        for index in range(1, len(codes)):
            previous, current = codes[index - 1], codes[index]
            if previous == 10 and current not in (4, 5):
                masked[index - 1] = True
            elif current == 5 and previous not in (2, 10):
                masked[index] = True
        return masked
    for index in range(1, len(codes)):
        if codes[index] in (4, 5) and codes[index - 1] in (2, 10):
            masked[index - 1 : index + 1] = True
    return masked


def _apply_intervals(mask: np.ndarray, intervals: Sequence[tuple[int, int]]) -> None:
    for start, end in intervals:
        if start < 0 or end < start:
            raise ValueError("Mask intervals must be 0-based half-open coordinates.")
        mask[start : min(end, len(mask))] = True


def psmcfa_from_consensus(
    path: str | Path,
    *,
    output_path: str | Path | None = None,
    min_quality: int = 10,
    min_good_bases: int = 10_000,
    block_size: int = 100,
    max_missing_fraction: float = 0.9,
    minimum_good_fraction: float = 0.2,
    mask_pseudoautosomal: bool = False,
    mutation_filter: str | None = None,
    masks: Mapping[str, Sequence[tuple[int, int]]] | None = None,
) -> SmcData:
    """Convert a diploid consensus FASTA/FASTQ to PSMCFA windows.

    The filtering and window calls reproduce the original ``fq2psmcfa``
    behavior. Custom masks use 0-based half-open base coordinates.
    """
    path = Path(path)
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if min_quality < 0 or min_good_bases < 0:
        raise ValueError("Quality and good-base thresholds must be non-negative.")
    if not 0 <= max_missing_fraction <= 1:
        raise ValueError("max_missing_fraction must be between zero and one.")
    if not 0 <= minimum_good_fraction <= 1:
        raise ValueError("minimum_good_fraction must be between zero and one.")
    if mutation_filter not in _MUTATION_FILTERS:
        choices = ", ".join(sorted(value for value in _MUTATION_FILTERS if value is not None))
        raise ValueError(f"mutation_filter must be one of: None, {choices}.")

    records: list[dict] = []
    skipped: list[dict] = []
    for name, sequence, quality in _sequence_records(path):
        if not sequence:
            skipped.append({"name": name, "reason": "empty"})
            continue
        characters = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        uppercase = np.frombuffer(sequence.upper().encode("ascii"), dtype=np.uint8)
        codes = np.array([_NT16.get(chr(value), 15) for value in uppercase], dtype=np.int8)
        missing = (characters >= ord("a")) & (characters <= ord("z"))
        missing |= codes == 15
        if quality is not None:
            qualities = (
                np.frombuffer(quality.encode("ascii"), dtype=np.uint8).astype(np.int16) - 33
            )
            missing |= qualities < min_quality
        if mask_pseudoautosomal and name in {"X", "chrX"}:
            _apply_intervals(missing, _PSEUDOAUTOSOMAL_HG18)
        if masks is not None:
            _apply_intervals(missing, masks.get(name, ()))
        missing |= _mutation_mask(codes, mutation_filter)

        good_bases = int((~missing).sum())
        if good_bases < min_good_bases or good_bases / len(sequence) < minimum_good_fraction:
            skipped.append(
                {
                    "name": name,
                    "reason": "insufficient_callable_bases",
                    "good_bases": good_bases,
                    "length": len(sequence),
                }
            )
            continue

        windows: list[int] = []
        for start in range(0, len(sequence), block_size):
            stop = min(start + block_size, len(sequence))
            missing_count = int(missing[start:stop].sum())
            if missing_count / block_size > max_missing_fraction:
                windows.append(2)
                continue
            heterozygous = np.any(_BIT_COUNTS[codes[start:stop]] == 2)
            heterozygous &= np.any((~missing)[start:stop] & (_BIT_COUNTS[codes[start:stop]] == 2))
            windows.append(1 if heterozygous else 0)
        window_codes = np.asarray(windows, dtype=np.int8)
        records.append(
            {
                "name": name,
                "codes": window_codes,
                "L": len(window_codes),
                "L_e": int((window_codes < 2).sum()),
                "n_e": int((window_codes == 1).sum()),
                "source_bases": len(sequence),
                "callable_bases": good_bases,
            }
        )

    if not records:
        raise ValueError(f"No consensus records in {path} passed the callable-base filters.")
    data = SmcData(window_size=block_size)
    data.uns.update(
        {
            "records": records,
            "sum_L": sum(record["L_e"] for record in records),
            "sum_n": sum(record["n_e"] for record in records),
            "n_seqs": len(records),
            "source_path": str(path),
            "source_format": "consensus",
            "skipped_records": skipped,
            "preprocessing": {
                "min_quality": min_quality,
                "min_good_bases": min_good_bases,
                "block_size": block_size,
                "max_missing_fraction": max_missing_fraction,
                "minimum_good_fraction": minimum_good_fraction,
                "mask_pseudoautosomal": mask_pseudoautosomal,
                "mutation_filter": mutation_filter,
            },
        }
    )
    if output_path is not None:
        written = write_psmcfa(data, output_path)
        data.uns["psmcfa_path"] = str(written)
    return data


__all__ = ["psmcfa_from_consensus"]
