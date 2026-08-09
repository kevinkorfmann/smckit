"""Native, callability-aware VCF to multihetsep conversion.

The implementation is independent of ``msmc-tools``.  Its observable contract
is validated against the pinned ``generate_multihetsep.py`` helper used by
PSMC+ and MSMC2, while adding strict input validation and provenance.
"""

from __future__ import annotations

import bisect
import gzip
import heapq
import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from smckit._core import SmcData
from smckit._provenance import package_version, sha256_file
from smckit.io._multihetsep import read_multihetsep

_GT_SPLIT = re.compile(r"([/|])")
_BASES = frozenset("ACGT")
_MSMC_TOOLS_REPOSITORY = "https://github.com/stschiff/msmc-tools.git"
_MSMC_TOOLS_COMMIT = "4d1f05f39f7b4f8c205e602c180b44a7c68a7bba"
_GENERATE_MULTIHETSEP_SHA256 = "caaa87a07e0fe2dc7228f30c9aff759cf86e9f61b8332aabd41398399ea6331b"


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


@dataclass(frozen=True)
class _VcfRecord:
    contig: str
    position: int
    reference: str
    alleles: tuple[str, ...]
    genotype: tuple[int, int]
    phased: bool


def _read_vcf_samples(path: Path) -> tuple[str, ...]:
    with _open_text(path) as handle:
        for raw_line in handle:
            if raw_line.startswith("#CHROM"):
                fields = raw_line.rstrip("\n").split("\t")
                if len(fields) < 10:
                    raise ValueError(f"{path}: VCF contains no sample columns.")
                samples = tuple(fields[9:])
                if len(samples) != len(set(samples)):
                    raise ValueError(f"{path}: VCF sample names must be unique.")
                return samples
    raise ValueError(f"{path}: VCF is missing the #CHROM header.")


def _resolve_sample(path: Path, requested: str | None) -> tuple[str, int]:
    samples = _read_vcf_samples(path)
    if requested is None:
        if len(samples) != 1:
            raise ValueError(f"{path}: VCF contains {len(samples)} samples; select exactly one.")
        return samples[0], 0
    if requested not in samples:
        raise ValueError(f"{path}: selected sample {requested!r} is absent from the VCF.")
    return requested, samples.index(requested)


def _parse_genotype(
    value: str,
    *,
    path: Path,
    line_number: int,
    as_phased: bool,
) -> tuple[tuple[int, int], bool]:
    tokens = _GT_SPLIT.split(value)
    if len(tokens) == 1:
        allele_tokens = (tokens[0], tokens[0])
        phased = as_phased
    elif len(tokens) == 3 and tokens[1] in {"/", "|"}:
        allele_tokens = (tokens[0], tokens[2])
        phased = as_phased or tokens[1] == "|"
    else:
        raise ValueError(f"{path}:{line_number}: expected a haploid or diploid GT, got {value!r}.")
    if any(token == "." for token in allele_tokens):
        raise ValueError(
            f"{path}:{line_number}: missing GT alleles are not valid in a called-site VCF."
        )
    try:
        genotype = tuple(int(token) for token in allele_tokens)
    except ValueError as error:
        raise ValueError(f"{path}:{line_number}: invalid GT value {value!r}.") from error
    if any(allele < 0 for allele in genotype):
        raise ValueError(f"{path}:{line_number}: GT allele indices must be non-negative.")
    return (genotype[0], genotype[1]), phased


def _iter_vcf_records(
    path: Path,
    *,
    sample_index: int,
    as_phased: bool,
):
    previous_contig: str | None = None
    previous_position = 0
    with _open_text(path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line or raw_line.startswith("#"):
                continue
            fields = raw_line.rstrip("\n").split("\t")
            if len(fields) <= 9 + sample_index:
                raise ValueError(f"{path}:{line_number}: malformed VCF sample row.")
            contig = fields[0]
            try:
                position = int(fields[1])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid VCF position.") from error
            if position < 1:
                raise ValueError(f"{path}:{line_number}: VCF positions must be positive.")
            if previous_contig is None:
                previous_contig = contig
            elif contig != previous_contig:
                raise ValueError(f"{path}:{line_number}: one output may contain only one contig.")
            if position <= previous_position:
                raise ValueError(
                    f"{path}:{line_number}: VCF positions must be strictly increasing."
                )
            previous_position = position

            reference = fields[3].upper()
            alternatives = (
                ()
                if fields[4] == "."
                else tuple(allele.upper() for allele in fields[4].split(","))
            )
            alleles = (reference, *alternatives)
            if any(len(allele) != 1 or allele not in _BASES for allele in alleles):
                raise ValueError(
                    f"{path}:{line_number}: multihetsep conversion requires A/C/G/T SNVs."
                )
            format_fields = fields[8].split(":")
            if "GT" not in format_fields:
                raise ValueError(f"{path}:{line_number}: VCF record has no GT field.")
            gt_index = format_fields.index("GT")
            sample_fields = fields[9 + sample_index].split(":")
            if gt_index >= len(sample_fields):
                raise ValueError(f"{path}:{line_number}: selected sample has no GT value.")
            genotype, phased = _parse_genotype(
                sample_fields[gt_index],
                path=path,
                line_number=line_number,
                as_phased=as_phased,
            )
            if any(allele >= len(alleles) for allele in genotype):
                raise ValueError(f"{path}:{line_number}: GT references an unavailable allele.")
            yield _VcfRecord(
                contig=contig,
                position=position,
                reference=reference,
                alleles=alleles,
                genotype=genotype,
                phased=phased,
            )


def _merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _read_mask(path: Path, contig: str) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    with _open_text(path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            try:
                if len(fields) == 2:
                    start, end = int(fields[0]), int(fields[1])
                elif len(fields) >= 3:
                    if fields[0] != contig:
                        continue
                    start, end = int(fields[1]) + 1, int(fields[2])
                else:
                    raise ValueError
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: malformed mask interval.") from error
            if start < 1 or end < start:
                raise ValueError(
                    f"{path}:{line_number}: mask intervals must be positive and non-empty."
                )
            intervals.append((start, end))
    return _merge_intervals(intervals)


def _intersect_intervals(
    left: Sequence[tuple[int, int]], right: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if start <= end:
            result.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def _subtract_intervals(
    source: Sequence[tuple[int, int]], excluded: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    exclusions = _merge_intervals(excluded)
    exclusion_index = 0
    for start, end in source:
        cursor = start
        while exclusion_index < len(exclusions) and exclusions[exclusion_index][1] < cursor:
            exclusion_index += 1
        index = exclusion_index
        while index < len(exclusions) and exclusions[index][0] <= end:
            excluded_start, excluded_end = exclusions[index]
            if excluded_start > cursor:
                result.append((cursor, min(end, excluded_start - 1)))
            cursor = max(cursor, excluded_end + 1)
            if cursor > end:
                break
            index += 1
        if cursor <= end:
            result.append((cursor, end))
    return result


class _Callability:
    def __init__(self, intervals: Sequence[tuple[int, int]]):
        self.intervals = tuple(intervals)
        self.starts = tuple(start for start, _ in intervals)
        self.ends = tuple(end for _, end in intervals)
        cumulative: list[int] = []
        total = 0
        for start, end in intervals:
            cumulative.append(total)
            total += end - start + 1
        self.cumulative_before = tuple(cumulative)
        self.total = total

    def covered_through(self, position: int) -> int:
        index = bisect.bisect_right(self.starts, position) - 1
        if index < 0:
            return 0
        start, end = self.intervals[index]
        return self.cumulative_before[index] + max(0, min(position, end) - start + 1)

    def count(self, start: int, end: int) -> int:
        if end < start:
            return 0
        return self.covered_through(end) - self.covered_through(start - 1)

    def contains(self, position: int) -> bool:
        index = bisect.bisect_right(self.starts, position) - 1
        return index >= 0 and position <= self.ends[index]


def _build_callability(
    *,
    contig: str,
    masks: Sequence[Path],
    negative_masks: Sequence[Path],
    assume_all_sites_callable: bool,
    contig_length: int | None,
) -> _Callability:
    if contig_length is not None and contig_length < 1:
        raise ValueError("contig_length must be positive.")
    if not masks and not assume_all_sites_callable:
        raise ValueError(
            "At least one positive callability mask is required; variant-only VCFs "
            "cannot identify callable invariant sites. Set assume_all_sites_callable=True "
            "only for validated synthetic or all-sites inputs."
        )
    if masks:
        callable_intervals = _read_mask(masks[0], contig)
        for path in masks[1:]:
            callable_intervals = _intersect_intervals(callable_intervals, _read_mask(path, contig))
    else:
        callable_intervals = [(1, contig_length or (2**63 - 1))]
    if contig_length is not None:
        callable_intervals = _intersect_intervals(callable_intervals, [(1, contig_length)])
    excluded = [interval for path in negative_masks for interval in _read_mask(path, contig)]
    if excluded:
        callable_intervals = _subtract_intervals(callable_intervals, excluded)
    if not callable_intervals:
        raise ValueError(f"No callable positions remain for contig {contig!r}.")
    return _Callability(callable_intervals)


def _ordered_alleles(
    records: Sequence[_VcfRecord | None],
    *,
    reference: str,
    trios: Sequence[tuple[int, int, int]],
    max_phase_configurations: int,
) -> tuple[str, ...]:
    configurations: set[tuple[str, ...]] = {()}
    for record in records:
        if record is None:
            alternatives = ((reference, reference),)
        else:
            first = record.alleles[record.genotype[0]]
            second = record.alleles[record.genotype[1]]
            alternatives = ((first, second),)
            if not record.phased and first != second:
                alternatives = ((first, second), (second, first))
        configurations = {prefix + suffix for prefix in configurations for suffix in alternatives}
        if len(configurations) > max_phase_configurations:
            raise ValueError(
                "Unphased genotypes exceed max_phase_configurations; phase the data or "
                "raise the explicit safety limit."
            )

    child_haplotypes: set[int] = set()
    for child, father, mother in trios:
        filtered = {
            config
            for config in configurations
            if {config[2 * child], config[2 * child + 1]}
            == {config[2 * father], config[2 * mother]}
        }
        if filtered:
            configurations = filtered
        child_haplotypes.update((2 * child, 2 * child + 1))
    retained = [index for index in range(2 * len(records)) if index not in child_haplotypes]
    stripped = {tuple(config[index] for index in retained) for config in configurations}
    return tuple("".join(config) for config in sorted(stripped))


def _is_segregating(configurations: Sequence[str]) -> bool:
    return any(
        any(allele != configuration[0] for allele in configuration[1:])
        for configuration in configurations
    )


def _normalize_paths(paths: str | Path | Sequence[str | Path], *, label: str) -> tuple[Path, ...]:
    raw_paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    normalized = tuple(Path(path).expanduser().resolve() for path in raw_paths)
    if not normalized:
        raise ValueError(f"At least one {label} is required.")
    for path in normalized:
        if not path.is_file():
            raise FileNotFoundError(f"{label.capitalize()} does not exist: {path}")
    return normalized


def _normalize_optional_paths(
    paths: Sequence[str | Path] | None, *, label: str
) -> tuple[Path, ...]:
    if not paths:
        return ()
    return _normalize_paths(paths, label=label)


def _normalize_samples(
    samples: str | Sequence[str | None] | None, number_vcfs: int
) -> tuple[str | None, ...]:
    if samples is None:
        return (None,) * number_vcfs
    if isinstance(samples, str):
        if number_vcfs != 1:
            raise ValueError("A single sample selector is valid only with one VCF.")
        return (samples,)
    normalized = tuple(samples)
    if len(normalized) != number_vcfs:
        raise ValueError("samples must contain one selector per VCF.")
    return normalized


def _validate_trios(
    trios: Sequence[tuple[int, int, int]] | None, number_individuals: int
) -> tuple[tuple[int, int, int], ...]:
    normalized = tuple(tuple(int(value) for value in trio) for trio in (trios or ()))
    children: set[int] = set()
    for trio in normalized:
        if len(trio) != 3 or len(set(trio)) != 3:
            raise ValueError("Each trio must contain distinct child, father, and mother indices.")
        if any(index < 0 or index >= number_individuals for index in trio):
            raise ValueError("Trio indices must refer to supplied VCF individuals.")
        if trio[0] in children:
            raise ValueError("Each child may occur in at most one trio.")
        children.add(trio[0])
    if number_individuals - len(children) < 1:
        raise ValueError("Trio removal must leave at least one individual.")
    return normalized


def multihetsep_from_vcf(
    vcf_paths: str | Path | Sequence[str | Path],
    output_path: str | Path,
    *,
    mask_paths: Sequence[str | Path] | None = None,
    negative_mask_paths: Sequence[str | Path] | None = None,
    samples: str | Sequence[str | None] | None = None,
    trios: Sequence[tuple[int, int, int]] | None = None,
    chromosome: str | None = None,
    as_phased: bool = False,
    assume_all_sites_callable: bool = False,
    contig_length: int | None = None,
    max_phase_configurations: int = 1_000_000,
    pair_indices: Sequence[tuple[int, int]] | None = None,
) -> SmcData:
    """Convert called single-sample VCFs and masks to multihetsep.

    Positive masks are intersected and negative masks are subtracted. Standard
    three-column BED files use 0-based half-open coordinates; two-column masks
    use the one-based inclusive convention accepted by ``msmc-tools``.

    A positive callability mask is mandatory by default. A variant-only VCF
    cannot distinguish confidently homozygous reference sites from sites that
    were not callable, so bypassing this requirement must be explicit.
    """
    if max_phase_configurations < 1:
        raise ValueError("max_phase_configurations must be positive.")
    sources = _normalize_paths(vcf_paths, label="VCF input")
    masks = _normalize_optional_paths(mask_paths, label="positive mask")
    negative_masks = _normalize_optional_paths(negative_mask_paths, label="negative mask")
    selectors = _normalize_samples(samples, len(sources))
    normalized_trios = _validate_trios(trios, len(sources))
    selected = [_resolve_sample(path, selector) for path, selector in zip(sources, selectors)]
    iterators = [
        iter(_iter_vcf_records(path, sample_index=index, as_phased=as_phased))
        for path, (_, index) in zip(sources, selected)
    ]
    heap: list[tuple[int, int, _VcfRecord]] = []
    for index, iterator in enumerate(iterators):
        try:
            record = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(heap, (record.position, index, record))
    if not heap:
        raise ValueError("VCF inputs contain no variant records.")

    source_contig = heap[0][2].contig
    output_contig = chromosome or source_contig
    callability = _build_callability(
        contig=source_contig,
        masks=masks,
        negative_masks=negative_masks,
        assume_all_sites_callable=assume_all_sites_callable,
        contig_length=contig_length,
    )
    target = Path(output_path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing multihetsep output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    variant_positions = 0
    emitted_sites = 0
    previous_emitted_position = 0
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wt",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            while heap:
                position = heap[0][0]
                present: list[_VcfRecord | None] = [None] * len(sources)
                while heap and heap[0][0] == position:
                    _, index, record = heapq.heappop(heap)
                    if record.contig != source_contig:
                        raise ValueError("All VCF inputs must contain the same single contig.")
                    present[index] = record
                    try:
                        following = next(iterators[index])
                    except StopIteration:
                        continue
                    heapq.heappush(heap, (following.position, index, following))
                variant_positions += 1
                if contig_length is not None and position > contig_length:
                    raise ValueError(
                        f"VCF position {source_contig}:{position} exceeds contig_length."
                    )
                references = {record.reference for record in present if record is not None}
                if len(references) != 1:
                    raise ValueError(
                        "VCF inputs disagree on the reference allele at "
                        f"{source_contig}:{position}."
                    )
                reference = next(iter(references))
                configurations = _ordered_alleles(
                    present,
                    reference=reference,
                    trios=normalized_trios,
                    max_phase_configurations=max_phase_configurations,
                )
                if not callability.contains(position) or not _is_segregating(configurations):
                    continue
                callable_count = callability.count(previous_emitted_position + 1, position)
                if callable_count < 1:
                    raise ValueError(
                        "A callable segregating site must contribute at least one callable base."
                    )
                allele_field = ",".join(configurations)
                handle.write(f"{output_contig}\t{position}\t{callable_count}\t{allele_field}\n")
                previous_emitted_position = position
                emitted_sites += 1
        if emitted_sites == 0:
            raise ValueError("VCF conversion produced no callable segregating sites.")
        os.replace(temporary_name, target)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

    data = read_multihetsep(target, pair_indices=pair_indices)
    data.uns["preprocessing"] = {
        "schema_version": "1.0",
        "tool": "smckit.pp.multihetsep_from_vcf",
        "package_version": package_version(),
        "source_vcfs": [str(path) for path in sources],
        "source_vcf_sha256": {str(path): sha256_file(path) for path in sources},
        "samples": [sample for sample, _ in selected],
        "positive_masks": [str(path) for path in masks],
        "positive_mask_sha256": {str(path): sha256_file(path) for path in masks},
        "negative_masks": [str(path) for path in negative_masks],
        "negative_mask_sha256": {str(path): sha256_file(path) for path in negative_masks},
        "source_contig": source_contig,
        "output_contig": output_contig,
        "contig_length": contig_length,
        "as_phased": bool(as_phased),
        "assume_all_sites_callable": bool(assume_all_sites_callable),
        "trios": [list(trio) for trio in normalized_trios],
        "max_phase_configurations": max_phase_configurations,
        "variant_positions_read": variant_positions,
        "segregating_sites_emitted": emitted_sites,
        "callable_bases_through_last_site": callability.covered_through(previous_emitted_position),
        "output_path": str(target),
        "output_sha256": sha256_file(target),
        "compatibility_oracle": {
            "repository": _MSMC_TOOLS_REPOSITORY,
            "commit": _MSMC_TOOLS_COMMIT,
            "entrypoint": "generate_multihetsep.py",
            "entrypoint_sha256": _GENERATE_MULTIHETSEP_SHA256,
            "redistributed": False,
        },
    }
    return data


__all__ = ["multihetsep_from_vcf"]
