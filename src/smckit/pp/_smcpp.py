"""Native VCF-to-SMC++ preparation with explicit masks and provenance."""

from __future__ import annotations

import bisect
import gzip
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from smckit._core import SmcData
from smckit._provenance import package_version, sha256_file
from smckit.io._smcpp_input import read_smcpp_input

_CONTIG_RE = re.compile(r"##contig=<(.+)>")
_GT_SPLIT_RE = re.compile(r"[|/]")


def _open_text(path: Path, mode: str):
    return gzip.open(path, mode, encoding="utf-8") if path.suffix == ".gz" else path.open(
        mode,
        encoding="utf-8",
    )


def _parse_contig_metadata(line: str) -> tuple[str | None, int | None]:
    match = _CONTIG_RE.fullmatch(line)
    if match is None:
        return None, None
    fields: dict[str, str] = {}
    for token in match.group(1).split(","):
        key, separator, value = token.partition("=")
        if separator:
            fields[key.strip()] = value.strip()
    contig = fields.get("ID")
    length = int(fields["length"]) if fields.get("length", "").isdigit() else None
    return contig, length


def _read_vcf_header(path: Path) -> tuple[list[str], dict[str, int | None]]:
    samples: list[str] | None = None
    contigs: dict[str, int | None] = {}
    with _open_text(path, "rt") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("##contig="):
                contig, length = _parse_contig_metadata(line)
                if contig is not None:
                    contigs[contig] = length
            elif line.startswith("#CHROM"):
                fields = line.split("\t")
                if len(fields) < 10:
                    raise ValueError("VCF must contain at least one diploid sample.")
                samples = fields[9:]
                break
    if samples is None:
        raise ValueError("VCF is missing the #CHROM header.")
    if len(samples) != len(set(samples)):
        raise ValueError("VCF sample names must be unique.")
    return samples, contigs


def _normalize_populations(
    populations: Mapping[str, Sequence[str]],
    *,
    available_samples: Sequence[str],
    ignore_missing_samples: bool,
) -> list[tuple[str, list[str]]]:
    if not 1 <= len(populations) <= 2:
        raise ValueError("SMC++ preparation supports one or two populations.")
    available = set(available_samples)
    normalized: list[tuple[str, list[str]]] = []
    assigned: set[str] = set()
    for population, raw_samples in populations.items():
        population_name = str(population).strip()
        if not population_name:
            raise ValueError("Population identifiers must be non-empty.")
        samples = [str(sample) for sample in raw_samples]
        if not samples:
            raise ValueError(f"Population {population_name!r} has no samples.")
        if len(samples) != len(set(samples)):
            raise ValueError(f"Population {population_name!r} contains duplicate samples.")
        overlap = assigned.intersection(samples)
        if overlap:
            raise ValueError(
                "Populations must be disjoint; duplicated samples: "
                + ", ".join(sorted(overlap))
            )
        missing = [sample for sample in samples if sample not in available]
        if missing and not ignore_missing_samples:
            raise ValueError(
                f"Samples absent from VCF for population {population_name!r}: "
                + ", ".join(missing)
            )
        samples = [sample for sample in samples if sample in available]
        if not samples:
            raise ValueError(
                f"Population {population_name!r} has no samples after missing-sample filtering."
            )
        assigned.update(samples)
        normalized.append((population_name, samples))
    return normalized


def _normalize_distinguished(
    populations: Sequence[tuple[str, Sequence[str]]],
    distinguished: Sequence[tuple[str, int]] | None,
) -> tuple[list[list[tuple[str, int]]], list[list[tuple[str, int]]]]:
    if distinguished is None:
        first_sample = populations[0][1][0]
        distinguished = [(first_sample, 0), (first_sample, 1)]
    normalized = [(str(sample), int(haplotype)) for sample, haplotype in distinguished]
    if len(normalized) != 2 or len(set(normalized)) != 2:
        raise ValueError("Exactly two distinct distinguished haplotypes are required.")
    if any(haplotype not in {0, 1} for _, haplotype in normalized):
        raise ValueError("Distinguished haplotype indices must be 0 or 1.")

    population_sets = [set(samples) for _, samples in populations]
    dist: list[list[tuple[str, int]]] = [[] for _ in populations]
    for item in normalized:
        matches = [index for index, samples in enumerate(population_sets) if item[0] in samples]
        if len(matches) != 1:
            raise ValueError(f"Distinguished sample {item[0]!r} is not in one population.")
        dist[matches[0]].append(item)
    undist = [
        [
            (sample, haplotype)
            for sample in samples
            for haplotype in (0, 1)
            if (sample, haplotype) not in dist[index]
        ]
        for index, (_, samples) in enumerate(populations)
    ]
    if sum(map(len, undist)) == 0:
        raise ValueError("At least one undistinguished haplotype is required.")
    return dist, undist


def _parse_gt(value: str) -> tuple[int | None, int | None]:
    alleles = _GT_SPLIT_RE.split(value)
    if len(alleles) != 2:
        raise ValueError(f"Expected a diploid GT value, got {value!r}.")
    parsed: list[int | None] = []
    for allele in alleles:
        if allele == ".":
            parsed.append(None)
        elif allele in {"0", "1"}:
            parsed.append(int(allele))
        else:
            raise ValueError(f"Only biallelic diploid genotypes are supported, got {value!r}.")
    return parsed[0], parsed[1]


def _read_mask(path: Path, contig: str, length: int) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    with _open_text(path, "rt") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"Malformed BED mask at {path}:{line_number}.")
            if fields[0] != contig:
                continue
            try:
                start, end = int(fields[1]), int(fields[2])
            except ValueError as exc:
                raise ValueError(f"Non-integer BED coordinates at {path}:{line_number}.") from exc
            if start < 0 or end <= start:
                raise ValueError("BED masks must use valid 0-based half-open intervals.")
            start = min(start, length)
            end = min(end, length)
            if start < end:
                intervals.append((start + 1, end + 1))
    intervals.sort()
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _masked(position: int, intervals: Sequence[tuple[int, int]], starts: Sequence[int]) -> bool:
    index = bisect.bisect_right(starts, position) - 1
    return index >= 0 and position < intervals[index][1]


def _append_span(
    rows: list[tuple[int, tuple[tuple[int, int, int], ...]]],
    span: int,
    observation: tuple[tuple[int, int, int], ...],
) -> None:
    if span <= 0:
        return
    if rows and rows[-1][1] == observation:
        rows[-1] = (rows[-1][0] + span, observation)
    else:
        rows.append((span, observation))


def _vcf_events(
    path: Path,
    *,
    contig: str,
    sample_indices: Mapping[str, int],
    dist: Sequence[Sequence[tuple[str, int]]],
    undist: Sequence[Sequence[tuple[str, int]]],
) -> dict[int, tuple[tuple[int, int, int], ...]]:
    events: dict[int, tuple[tuple[int, int, int], ...]] = {}
    with _open_text(path, "rt") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line or raw_line.startswith("#"):
                continue
            fields = raw_line.rstrip("\n").split("\t")
            if len(fields) < 10 or fields[0] != contig:
                continue
            try:
                position = int(fields[1])
            except ValueError as exc:
                raise ValueError(f"Invalid VCF position at {path}:{line_number}.") from exc
            if position in events:
                continue
            reference, alternatives = fields[3], fields[4].split(",")
            if len(reference) != 1 or len(alternatives) != 1 or len(alternatives[0]) != 1:
                continue
            format_fields = fields[8].split(":")
            if "GT" not in format_fields:
                raise ValueError(f"VCF record lacks GT at {path}:{line_number}.")
            gt_index = format_fields.index("GT")
            genotypes: dict[str, tuple[int | None, int | None]] = {}
            for sample, index in sample_indices.items():
                sample_fields = fields[9 + index].split(":")
                if gt_index >= len(sample_fields):
                    raise ValueError(
                        f"VCF sample {sample!r} lacks GT at {path}:{line_number}."
                    )
                genotypes[sample] = _parse_gt(sample_fields[gt_index])

            population_values: list[tuple[int, int, int]] = []
            all_derived = True
            for population in range(len(dist)):
                distinguished_alleles = [
                    genotypes[sample][haplotype] for sample, haplotype in dist[population]
                ]
                a = (
                    -1
                    if any(allele is None for allele in distinguished_alleles)
                    else sum(int(allele) for allele in distinguished_alleles)
                )
                undistinguished_alleles = [
                    genotypes[sample][haplotype]
                    for sample, haplotype in undist[population]
                    if genotypes[sample][haplotype] is not None
                ]
                b = sum(int(allele) for allele in undistinguished_alleles)
                n_observed = len(undistinguished_alleles)
                population_values.append((a, b, n_observed))
                all_derived &= (
                    a == len(dist[population])
                    and n_observed == len(undist[population])
                    and b == n_observed
                )
            if all_derived:
                population_values = [
                    (0, 0, n_observed) for _, _, n_observed in population_values
                ]
            events[position] = tuple(population_values)
    return events


def smcpp_from_vcf(
    vcf_path: str | Path,
    output_path: str | Path,
    *,
    contig: str,
    populations: Mapping[str, Sequence[str]],
    distinguished: Sequence[tuple[str, int]] | None = None,
    length: int | None = None,
    mask_path: str | Path | None = None,
    missing_cutoff: int | None = None,
    ignore_missing_samples: bool = False,
    drop_first_last: bool = False,
) -> SmcData:
    """Convert plain or gzip-compressed VCF data to native SMC++ input.

    BED masks use standard 0-based half-open coordinates. ``missing_cutoff``
    marks unobserved non-variant segments longer than the supplied number of
    bases as missing, matching the upstream converter's conservative option.
    """
    source = Path(vcf_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"VCF input does not exist: {source}")
    if missing_cutoff is not None and missing_cutoff < 0:
        raise ValueError("missing_cutoff must be non-negative.")
    if mask_path is not None and missing_cutoff is not None:
        raise ValueError("mask_path and missing_cutoff are mutually exclusive.")

    header_samples, header_contigs = _read_vcf_header(source)
    if contig not in header_contigs and length is None:
        raise ValueError(
            f"Contig {contig!r} is absent from VCF metadata; supply an explicit length."
        )
    contig_length = int(length or header_contigs.get(contig) or 0)
    if contig_length <= 0:
        raise ValueError("A positive contig length is required.")
    normalized_populations = _normalize_populations(
        populations,
        available_samples=header_samples,
        ignore_missing_samples=ignore_missing_samples,
    )
    dist, undist = _normalize_distinguished(normalized_populations, distinguished)
    selected_samples = {
        sample for _, samples in normalized_populations for sample in samples
    }
    sample_indices = {
        sample: header_samples.index(sample) for sample in sorted(selected_samples)
    }
    events = _vcf_events(
        source,
        contig=contig,
        sample_indices=sample_indices,
        dist=dist,
        undist=undist,
    )

    mask_source = Path(mask_path).expanduser().resolve() if mask_path is not None else None
    if mask_source is not None and not mask_source.is_file():
        raise FileNotFoundError(f"BED mask does not exist: {mask_source}")
    masks = _read_mask(mask_source, contig, contig_length) if mask_source else []
    mask_starts = [start for start, _ in masks]
    events = {
        position: observation
        for position, observation in events.items()
        if 1 <= position <= contig_length and not _masked(position, masks, mask_starts)
    }

    boundaries = {1, contig_length + 1}
    for position in events:
        boundaries.update((position, position + 1))
    for start, end in masks:
        boundaries.update((start, end))
    ordered = sorted(boundary for boundary in boundaries if 1 <= boundary <= contig_length + 1)
    nonseg = tuple((0, 0, len(population)) for population in undist)
    missing = tuple((-1, 0, 0) for _ in undist)
    rows: list[tuple[int, tuple[tuple[int, int, int], ...]]] = []
    for start, end in zip(ordered[:-1], ordered[1:], strict=True):
        span = end - start
        if start in events:
            observation = events[start]
        elif _masked(start, masks, mask_starts):
            observation = missing
        elif missing_cutoff is not None and span > missing_cutoff:
            observation = missing
        else:
            observation = nonseg
        _append_span(rows, span, observation)
    if drop_first_last:
        rows = rows[1:-1] if len(rows) > 2 else []
    if not rows:
        raise ValueError("VCF conversion produced no SMC++ observations.")

    header = {
        "version": f"smckit-{package_version()}",
        "pids": [population for population, _ in normalized_populations],
        "undist": undist,
        "dist": dist,
        "contig": contig,
        "length": contig_length,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(target, "wt") as handle:
        handle.write("# SMC++ " + json.dumps(header, sort_keys=True) + "\n")
        for span, population_values in rows:
            values = [span]
            for a, b, n_observed in population_values:
                values.extend([a, b, n_observed])
            handle.write("\t".join(map(str, values)) + "\n")

    data = read_smcpp_input(target)
    data.uns["preprocessing"] = {
        "source_vcf": str(source),
        "source_sha256": sha256_file(source),
        "output_path": str(target),
        "output_sha256": sha256_file(target),
        "contig": contig,
        "length": contig_length,
        "mask_path": str(mask_source) if mask_source else None,
        "mask_sha256": sha256_file(mask_source) if mask_source else None,
        "missing_cutoff": missing_cutoff,
        "ignore_missing_samples": ignore_missing_samples,
        "drop_first_last": drop_first_last,
    }
    return data


__all__ = ["smcpp_from_vcf"]
