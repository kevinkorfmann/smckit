"""diCal2: Demographic Inference using Composite Approximate Likelihoods (v2).

Reimplementation of diCal2 (Steinrücken, Kamm & Song, PNAS 116(34):17115–17120,
2019). diCal2 infers parametric demographic histories of multiple populations
from phased whole-genome sequence data, supporting:

* Piecewise-constant population sizes per epoch and ancient deme.
* Continuous migration between demes within an epoch.
* Population splits / mergers across epoch boundaries.
* Pulse migration events at epoch boundaries.

The core idea is the *Conditional Sampling Distribution* (CSD): an HMM over
the genome where the hidden state encodes the time interval and ancient deme
in which an *additional* lineage coalesces with a *trunk* of reference
haplotypes. The product of these CSDs over leave-one-out / pairwise / PAC
configurations gives a composite likelihood that is maximized via EM.

This module follows the smckit conventions: a single ``dical2`` entry point
that takes an ``SmcData`` populated by ``smckit.io.read_dical2`` and writes
results into ``data.results['dical2']``.

References
----------
Steinrücken M., Kamm J. and Song Y.S. (2019).
*Inference of complex population histories using whole-genome sequences from
multiple populations.* PNAS 116(34):17115–17120.
"""

from __future__ import annotations

import logging
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import eig
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.special import logsumexp

from smckit._core import SmcData
from smckit.io._dical2 import DiCal2Config, DiCal2Demo, DiCal2Epoch
from smckit.tl._implementation import (
    annotate_result,
    choose_implementation,
    method_upstream_available,
    normalize_implementation,
    require_upstream_available,
    standard_upstream_metadata,
)

logger = logging.getLogger(__name__)

DICAL2_T_INF = 1000.0
LOG_ZERO = -1e300
EPS = 1e-12


def _dical2_java_help() -> str:
    return (
        "Java runtime is required for upstream diCal2.\n"
        "Install one of the following and retry:\n"
        "  brew install --cask temurin\n"
        "  or: brew install openjdk\n"
        "Then verify with:\n"
        "  java -version"
    )


@dataclass(frozen=True)
class DiCal2ResolvedOptions:
    """Resolved diCal2 execution semantics shared by native and upstream paths."""

    composite_mode: str
    loci_per_hmm_step: int
    seed: int | None
    start_point: np.ndarray | None
    meta_start_file: str | None
    meta_num_iterations: int
    meta_keep_best: int
    meta_num_points: int | None
    bounds: str | list[tuple[float, float]] | None
    number_iterations_em: int
    number_iterations_mstep: int | None
    relative_error_e: float | None
    relative_error_m: float | None
    coordinatewise_mstep: bool
    coordinate_order: tuple[int, ...] | None
    nm_fraction: float | None
    use_param_rel_err: bool
    use_param_rel_err_m: bool
    interval_type: str | None
    interval_params: str | None
    legacy_n_intervals: int
    legacy_max_t: float
    legacy_alpha: float
    meta_stretch_proportion: float = 0.25
    meta_disperse_factor: float = 1.0
    meta_sd_percentage_if_zero: float = 0.2


def _lookup_option(
    options: dict | None,
    *names: str,
    default: object | None = None,
) -> object | None:
    if not options:
        return default
    for name in names:
        if name in options:
            return options[name]
    return default


def _normalize_interval_type(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"loguniform", "log_uniform", "log-uniform"}:
        return "loguniform"
    if lowered == "old":
        return "old"
    if lowered in {"customfixed", "custom_fixed", "custom-fixed"}:
        return "customfixed"
    raise ValueError(
        "Unsupported diCal2 interval_type. Expected one of: old, logUniform, customFixed."
    )


def _coerce_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _coerce_coordinate_order(value: object | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
        return tuple(int(item) for item in items)
    if isinstance(value, np.ndarray):
        return tuple(int(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    raise TypeError("coordinate_order must be a comma-separated string or sequence of ints.")


def _resolve_dical2_options(
    *,
    n_intervals: int,
    max_t: float,
    alpha: float,
    n_em_iterations: int,
    composite_mode: str,
    loci_per_hmm_step: int,
    start_point: np.ndarray | list[float] | None,
    meta_start_file: str | None,
    meta_num_iterations: int,
    meta_keep_best: int,
    meta_num_points: int | None,
    bounds: str | list[tuple[float, float]] | None,
    seed: int | None,
    method_options: dict | None,
) -> DiCal2ResolvedOptions:
    interval_type = _normalize_interval_type(
        _lookup_option(method_options, "interval_type", "intervalType")
    )
    interval_params = _lookup_option(
        method_options,
        "interval_params",
        "intervalParams",
    )
    if interval_params is not None:
        interval_params = str(interval_params)
    if interval_type is not None and interval_params is None:
        if interval_type == "old":
            interval_params = str(int(n_intervals))
        elif interval_type == "loguniform":
            interval_params = f"{int(n_intervals)},{float(alpha)},{float(max_t)}"
    resolved_start_point = _lookup_option(
        method_options,
        "start_point",
        "startPoint",
        default=start_point,
    )
    if resolved_start_point is not None:
        resolved_start_point = np.asarray(resolved_start_point, dtype=np.float64)
    resolved_meta_start = _lookup_option(
        method_options,
        "meta_start_file",
        "metaStartFile",
        default=meta_start_file,
    )
    if resolved_start_point is not None and resolved_meta_start is not None:
        raise ValueError("Specify either start_point or meta_start_file for diCal2, not both.")
    coordinatewise = _lookup_option(
        method_options,
        "coordinatewise_mstep",
        "coordinateWiseMStep",
    )
    if coordinatewise is None:
        disable_coordinatewise = _lookup_option(
            method_options,
            "disable_coordinatewise_mstep",
            "disableCoordinateWiseMStep",
        )
        coordinatewise = not _coerce_bool(disable_coordinatewise, default=False)
    else:
        coordinatewise = _coerce_bool(coordinatewise, default=True)
    return DiCal2ResolvedOptions(
        composite_mode=str(
            _lookup_option(
                method_options,
                "composite_mode",
                "compositeLikelihood",
                default=composite_mode,
            )
        ).lower(),
        loci_per_hmm_step=int(
            _lookup_option(
                method_options,
                "loci_per_hmm_step",
                "lociPerHmmStep",
                default=loci_per_hmm_step,
            )
        ),
        seed=(
            None
            if _lookup_option(method_options, "seed", default=seed) is None
            else int(_lookup_option(method_options, "seed", default=seed))
        ),
        start_point=resolved_start_point,
        meta_start_file=(
            None if resolved_meta_start is None else str(resolved_meta_start)
        ),
        meta_num_iterations=int(
            _lookup_option(
                method_options,
                "meta_num_iterations",
                "metaNumIterations",
                default=meta_num_iterations,
            )
        ),
        meta_keep_best=int(
            _lookup_option(
                method_options,
                "meta_keep_best",
                "metaKeepBest",
                default=meta_keep_best,
            )
        ),
        meta_num_points=(
            None
            if _lookup_option(
                method_options,
                "meta_num_points",
                "metaNumPoints",
                default=meta_num_points,
            )
            is None
            else int(
                _lookup_option(
                    method_options,
                    "meta_num_points",
                    "metaNumPoints",
                    default=meta_num_points,
                )
            )
        ),
        bounds=_lookup_option(method_options, "bounds", default=bounds),
        number_iterations_em=int(
            _lookup_option(
                method_options,
                "number_iterations_em",
                "numberIterationsEM",
                default=n_em_iterations,
            )
        ),
        number_iterations_mstep=(
            None
            if _lookup_option(
                method_options,
                "number_iterations_mstep",
                "numberIterationsMstep",
            )
            is None
            else int(
                _lookup_option(
                    method_options,
                    "number_iterations_mstep",
                    "numberIterationsMstep",
                )
            )
        ),
        relative_error_e=(
            None
            if _lookup_option(
                method_options,
                "relative_error_e",
                "relativeErrorE",
            )
            is None
            else float(
                _lookup_option(
                    method_options,
                    "relative_error_e",
                    "relativeErrorE",
                )
            )
        ),
        relative_error_m=(
            None
            if _lookup_option(
                method_options,
                "relative_error_m",
                "relativeErrorM",
            )
            is None
            else float(
                _lookup_option(
                    method_options,
                    "relative_error_m",
                    "relativeErrorM",
                )
            )
        ),
        coordinatewise_mstep=bool(coordinatewise),
        coordinate_order=_coerce_coordinate_order(
            _lookup_option(
                method_options,
                "coordinate_order",
                "coordinateOrder",
            )
        ),
        nm_fraction=(
            float(_lookup_option(method_options, "nm_fraction", "nmFraction", default=0.2))
        ),
        use_param_rel_err=_coerce_bool(
            _lookup_option(method_options, "use_param_rel_err", "useParamRelErr"),
            default=False,
        ),
        use_param_rel_err_m=_coerce_bool(
            _lookup_option(method_options, "use_param_rel_err_m", "useParamRelErrM"),
            default=False,
        ),
        interval_type=interval_type,
        interval_params=interval_params,
        legacy_n_intervals=int(n_intervals),
        legacy_max_t=float(max_t),
        legacy_alpha=float(alpha),
        meta_stretch_proportion=float(
            _lookup_option(
                method_options,
                "meta_stretch_proportion",
                "metaStretchProportion",
                default=0.25,
            )
        ),
        meta_disperse_factor=float(
            _lookup_option(
                method_options,
                "meta_disperse_factor",
                "metaDisperseFactor",
                default=1.0,
            )
        ),
        meta_sd_percentage_if_zero=float(
            _lookup_option(
                method_options,
                "meta_sd_percentage_if_zero",
                "metaSDPercentageIfZero",
                default=0.2,
            )
        ),
    )


def _resolved_options_metadata(resolved: DiCal2ResolvedOptions) -> dict[str, object]:
    return {
        "composite_mode": resolved.composite_mode,
        "loci_per_hmm_step": resolved.loci_per_hmm_step,
        "seed": resolved.seed,
        "start_point": (
            None if resolved.start_point is None else resolved.start_point.tolist()
        ),
        "meta_start_file": resolved.meta_start_file,
        "meta_num_iterations": resolved.meta_num_iterations,
        "meta_keep_best": resolved.meta_keep_best,
        "meta_num_points": resolved.meta_num_points,
        "bounds": resolved.bounds,
        "number_iterations_em": resolved.number_iterations_em,
        "number_iterations_mstep": resolved.number_iterations_mstep,
        "relative_error_e": resolved.relative_error_e,
        "relative_error_m": resolved.relative_error_m,
        "coordinatewise_mstep": resolved.coordinatewise_mstep,
        "coordinate_order": (
            None if resolved.coordinate_order is None else list(resolved.coordinate_order)
        ),
        "nm_fraction": resolved.nm_fraction,
        "use_param_rel_err": resolved.use_param_rel_err,
        "use_param_rel_err_m": resolved.use_param_rel_err_m,
        "interval_type": resolved.interval_type,
        "interval_params": resolved.interval_params,
        "legacy_n_intervals": resolved.legacy_n_intervals,
        "legacy_max_t": resolved.legacy_max_t,
        "legacy_alpha": resolved.legacy_alpha,
    }


def _meta_logify(fraction: float, minimum: float, maximum: float) -> float:
    if abs(maximum - 1.0) < EPS:
        maximum = 1.0 + EPS if maximum < 1.0 else 1.0 - EPS
    zero_to = np.log(minimum) / np.log(maximum)
    slope = 1.0 - zero_to
    value = maximum ** (slope * fraction + zero_to)
    return float(min(max(value, minimum), maximum))


def _interval_boundaries_from_intervals(
    intervals: list[tuple[float, float]],
) -> np.ndarray:
    boundaries = [float(intervals[0][0])]
    for _, end in intervals:
        boundaries.append(DICAL2_T_INF if not np.isfinite(end) else float(end))
    return np.array(boundaries, dtype=np.float64)


def _balanced_partition(n_intervals: int) -> list[tuple[float, float]]:
    if n_intervals <= 0:
        raise ValueError("n_intervals must be positive")
    if n_intervals == 1:
        return [(0.0, np.inf)]
    weight_per_interval = 1.0 / n_intervals
    intervals: list[tuple[float, float]] = []
    end_point = np.inf
    start_point = -np.log(weight_per_interval)
    intervals.append((float(start_point), float(end_point)))
    for i in range(2, n_intervals):
        end_point = start_point
        start_point = -np.log(i * weight_per_interval)
        intervals.append((float(start_point), float(end_point)))
    intervals.append((0.0, float(start_point)))
    intervals.reverse()
    return intervals


def _exponential_partition(n_intervals: int, exp_rate: float) -> list[tuple[float, float]]:
    return [
        (start / exp_rate, end / exp_rate)
        for start, end in _balanced_partition(n_intervals)
    ]


def _old_interval_boundaries(demo: DiCal2Demo, config: DiCal2Config, interval_params: str) -> np.ndarray:
    num_additional_intervals = int(interval_params)
    first_epoch = next((epoch for epoch in demo.epochs if epoch.pop_sizes is not None), None)
    if first_epoch is None or first_epoch.pop_sizes is None:
        raise ValueError("old interval factory requires a demographic epoch with population sizes.")
    pop_size_by_present: dict[int, float] = {}
    for ancient_idx, group in enumerate(first_epoch.partition):
        for present_idx in group:
            pop_size_by_present[present_idx] = float(first_epoch.pop_sizes[ancient_idx])
    exp_rate = 0.0
    for present_idx in range(config.n_populations):
        sample_size = int(config.sample_sizes[present_idx])
        if sample_size == 0:
            if (
                first_epoch.migration_matrix is None
                or present_idx >= first_epoch.migration_matrix.shape[0]
                or first_epoch.migration_matrix[present_idx, present_idx] == 0.0
            ):
                continue
            exp_rate -= 1.0 / (2.0 * first_epoch.migration_matrix[present_idx, present_idx])
            continue
        pop_size = pop_size_by_present.get(present_idx)
        if pop_size is None:
            continue
        exp_rate += pop_size / sample_size
    if exp_rate <= 0.0:
        raise ValueError("Could not construct old diCal2 intervals from the current demo/config.")
    exp_rate = config.n_populations / exp_rate
    return _interval_boundaries_from_intervals(
        _exponential_partition(num_additional_intervals + 1, exp_rate)
    )


def _loguniform_interval_boundaries(interval_params: str) -> np.ndarray:
    tokens = [chunk.strip() for chunk in interval_params.split(",") if chunk.strip()]
    if len(tokens) % 2 != 1:
        raise ValueError(
            "logUniform interval_params must provide n interval counts followed by n+1 boundaries."
        )
    midpoint = len(tokens) // 2
    interval_numbers = [int(token) for token in tokens[:midpoint]]
    interval_borders = [float(token) for token in tokens[midpoint:]]
    intervals: list[tuple[float, float]] = [(0.0, interval_borders[0])]
    for range_intervals, range_lower, range_upper in zip(
        interval_numbers,
        interval_borders[:-1],
        interval_borders[1:],
    ):
        prev_last = range_lower
        for idx in range(1, range_intervals + 1):
            this_last = (
                range_upper
                if idx == range_intervals
                else _meta_logify(idx / float(range_intervals), range_lower, range_upper)
            )
            intervals.append((float(prev_last), float(this_last)))
            prev_last = this_last
    intervals.append((interval_borders[-1], np.inf))
    return _interval_boundaries_from_intervals(intervals)


def _resolve_interval_boundaries(
    demo: DiCal2Demo | None,
    config: DiCal2Config,
    resolved: DiCal2ResolvedOptions,
) -> np.ndarray:
    if resolved.interval_type is None:
        interval_boundaries = compute_time_intervals(
            resolved.legacy_n_intervals,
            max_t=resolved.legacy_max_t,
            alpha=resolved.legacy_alpha,
        )
        return np.append(interval_boundaries, DICAL2_T_INF)
    if resolved.interval_params is None:
        raise ValueError("diCal2 interval_type requires interval_params.")
    if resolved.interval_type == "loguniform":
        return _loguniform_interval_boundaries(resolved.interval_params)
    if resolved.interval_type == "customfixed":
        points = np.array(
            [float(chunk.strip()) for chunk in resolved.interval_params.split(",") if chunk.strip()],
            dtype=np.float64,
        )
        return np.concatenate(([0.0], points, [DICAL2_T_INF]))
    if resolved.interval_type == "old":
        if demo is None:
            raise ValueError("old diCal2 interval_type requires a demographic model.")
        return _old_interval_boundaries(demo, config, resolved.interval_params)
    raise AssertionError(f"Unhandled diCal2 interval_type: {resolved.interval_type}")


# ===========================================================================
# Time interval discretization
# ===========================================================================


def compute_time_intervals(
    n_intervals: int,
    max_t: float = 15.0,
    alpha: float = 0.1,
) -> np.ndarray:
    """Compute exponentially spaced time boundaries.

    Same form as PSMC: ``t_k = alpha * (exp(beta * k) - 1)``.

    Parameters
    ----------
    n_intervals : int
        Number of refined time intervals.
    max_t : float
        Maximum coalescent time (units of 2N₀).
    alpha : float
        Resolution near t=0.

    Returns
    -------
    t : (n_intervals + 1,) array of boundaries.
    """
    K = n_intervals
    t = np.empty(K + 1, dtype=np.float64)
    beta = np.log(1.0 + max_t / alpha) / K
    for k in range(K):
        t[k] = alpha * (np.exp(beta * k) - 1.0)
    t[K] = max_t
    return t


def refine_demography(
    demo: DiCal2Demo,
    interval_boundaries: np.ndarray,
) -> "RefinedDemography":
    """Refine a demographic model by inserting extra interval boundaries.

    The HMM operates on a *refined* set of intervals: the union of the
    user-provided interval grid and the demographic epoch boundaries.

    Parameters
    ----------
    demo : DiCal2Demo
        Demographic model.
    interval_boundaries : (K+1,) array
        Time interval boundaries (the rightmost should be ``DICAL2_T_INF``).

    Returns
    -------
    RefinedDemography
    """
    boundaries = set()
    for b in interval_boundaries:
        boundaries.add(float(b))
    for b in demo.epoch_boundaries:
        if np.isfinite(b):
            boundaries.add(float(b))
    boundaries.add(DICAL2_T_INF)
    sorted_b = sorted(boundaries)

    refined = np.array(sorted_b, dtype=np.float64)
    n_refined = len(refined) - 1

    # For each refined interval, find which original epoch it belongs to.
    epoch_map = np.zeros(n_refined, dtype=np.int64)
    for i in range(n_refined):
        mid = 0.5 * (refined[i] + refined[i + 1])
        for e in range(len(demo.epochs)):
            if demo.epochs[e].start <= mid < demo.epochs[e].end:
                epoch_map[i] = e
                break
        else:
            epoch_map[i] = len(demo.epochs) - 1

    return RefinedDemography(
        demo=demo,
        refined_boundaries=refined,
        epoch_map=epoch_map,
    )


@dataclass
class RefinedDemography:
    """Demographic model refined to the HMM time grid."""

    demo: DiCal2Demo
    refined_boundaries: np.ndarray  # (n_refined+1,)
    epoch_map: np.ndarray  # (n_refined,) — index into demo.epochs

    @property
    def n_refined(self) -> int:
        return len(self.refined_boundaries) - 1

    def interval_duration(self, i: int) -> float:
        return float(self.refined_boundaries[i + 1] - self.refined_boundaries[i])

    def is_pulse(self, i: int) -> bool:
        return abs(self.interval_duration(i)) < EPS


# ===========================================================================
# Trunk process — absorption rates
# ===========================================================================


@dataclass
class SimpleTrunk:
    """Simple trunk process: absorption rate proportional to trunk size.

    For each epoch and ancient deme, the absorption rate is

        alpha[d] = (number of trunk haplotypes assigned to d) / N[d]

    where ``N[d]`` is the population size of ancient deme ``d``.
    """

    config: DiCal2Config
    additional_hap_idx: int  # the haplotype currently being conditioned on
    trunk_hap_indices: list[int] | None = None

    def _iter_trunk_haps(self):
        if self.trunk_hap_indices is None:
            for h in range(len(self.config.haplotype_populations)):
                if h != self.additional_hap_idx:
                    yield h
        else:
            for h in self.trunk_hap_indices:
                if h != self.additional_hap_idx:
                    yield h

    def trunk_sizes(self, partition: list[list[int]]) -> np.ndarray:
        """Number of trunk haplotypes per ancient deme."""
        n_anc = len(partition)
        sizes = np.zeros(n_anc, dtype=np.float64)
        for a, group in enumerate(partition):
            for present_d in group:
                # Count trunk haplotypes in this present deme
                count = sum(
                    self.config.haplotype_multiplicities[h, present_d]
                    for h in self._iter_trunk_haps()
                    if self.config.haplotype_populations[h] == present_d
                )
                sizes[a] += count
        return sizes

    def absorption_rates(
        self,
        partition: list[list[int]],
        pop_sizes: np.ndarray,
    ) -> np.ndarray:
        """Absorption rate per ancient deme."""
        sizes = self.trunk_sizes(partition)
        rates = np.zeros_like(sizes)
        for a in range(len(sizes)):
            if pop_sizes[a] > 0 and sizes[a] > 0:
                rates[a] = sizes[a] / pop_sizes[a]
        return rates

    def fraction_ancient_to_present(
        self,
        partition: list[list[int]],
        present_deme: int,
        ancient_deme: int,
    ) -> float:
        """Fraction of trunk in ancient deme that comes from present deme."""
        if present_deme not in partition[ancient_deme]:
            return 0.0
        ancient_size = sum(
            sum(
                self.config.haplotype_multiplicities[h, d]
                for h in self._iter_trunk_haps()
                if self.config.haplotype_populations[h] == d
            )
            for d in partition[ancient_deme]
        )
        my_size = sum(
            self.config.haplotype_multiplicities[h, present_deme]
            for h in self._iter_trunk_haps()
            if self.config.haplotype_populations[h] == present_deme
        )
        if ancient_size == 0:
            return 0.0
        return my_size / ancient_size


# ===========================================================================
# Extended migration matrix and eigendecomposition
# ===========================================================================


def build_extended_matrix(
    migration_matrix: np.ndarray | None,
    absorption_rates: np.ndarray,
) -> np.ndarray:
    """Build the extended migration+absorption rate matrix.

    Z = | M - diag(α)   diag(α) |
        |       0           0   |

    Parameters
    ----------
    migration_matrix : (k, k) array or None
        Continuous migration rate matrix (rows sum to 0). None ⇒ zero matrix.
    absorption_rates : (k,) array
        Absorption rate per deme.

    Returns
    -------
    Z : (2k, 2k) array — proper rate matrix.
    """
    k = len(absorption_rates)
    Z = np.zeros((2 * k, 2 * k), dtype=np.float64)
    if migration_matrix is not None:
        Z[:k, :k] = migration_matrix
    # Subtract absorption from diagonal of non-absorbing block
    for i in range(k):
        Z[i, i] -= absorption_rates[i]
        Z[i, k + i] = absorption_rates[i]
    return Z


def _epoch_growth_rates(epoch: DiCal2Epoch) -> np.ndarray:
    if epoch.growth_rates is None:
        return np.zeros(len(epoch.partition), dtype=np.float64)
    return np.asarray(epoch.growth_rates, dtype=np.float64)


def _refined_has_growth(refined: RefinedDemography) -> bool:
    for epoch in refined.demo.epochs:
        growth_rates = _epoch_growth_rates(epoch)
        if np.any(np.abs(growth_rates) > EPS):
            return True
    return False


def _refined_has_pulse(refined: RefinedDemography) -> bool:
    if any(refined.is_pulse(i) for i in range(refined.n_refined)):
        return True
    for epoch in refined.demo.epochs:
        if epoch.pulse_migration is not None:
            return True
    return False


def _absorption_rates_at_time(
    base_absorption_rates: np.ndarray,
    epoch: DiCal2Epoch,
    time_point: float,
) -> np.ndarray:
    growth_rates = _epoch_growth_rates(epoch)
    if np.isinf(epoch.end):
        return base_absorption_rates
    return base_absorption_rates * np.exp(-growth_rates * (epoch.end - time_point))


def _extended_rate_matrix_at_time(
    epoch: DiCal2Epoch,
    base_absorption_rates: np.ndarray,
    time_point: float,
    *,
    extra_loss: float = 0.0,
) -> np.ndarray:
    absorption_rates = _absorption_rates_at_time(base_absorption_rates, epoch, time_point)
    n_anc = len(absorption_rates)
    rate_matrix = np.zeros((2 * n_anc, 2 * n_anc), dtype=np.float64)
    if epoch.migration_matrix is not None:
        rate_matrix[:n_anc, :n_anc] = np.asarray(epoch.migration_matrix, dtype=np.float64)
    for ancient_deme in range(n_anc):
        rate_matrix[ancient_deme, ancient_deme] -= (
            float(absorption_rates[ancient_deme]) + float(extra_loss)
        )
        rate_matrix[ancient_deme, n_anc + ancient_deme] = float(absorption_rates[ancient_deme])
    return rate_matrix


def _integrate_transition_matrix(
    epoch: DiCal2Epoch,
    base_absorption_rates: np.ndarray,
    start_time: float,
    end_time: float,
    *,
    extra_loss: float = 0.0,
    dense_output: bool = False,
):
    n_anc = len(base_absorption_rates)
    if n_anc == 0:
        return np.zeros((0, 0), dtype=np.float64), None
    if abs(end_time - start_time) < EPS:
        identity = np.eye(2 * n_anc, dtype=np.float64)
        return identity, None

    def deriv(time_point: float, flat_state: np.ndarray) -> np.ndarray:
        state = flat_state.reshape((2 * n_anc, 2 * n_anc))
        rate_matrix = _extended_rate_matrix_at_time(
            epoch,
            base_absorption_rates,
            time_point,
            extra_loss=extra_loss,
        )
        return (state @ rate_matrix).reshape(-1)

    initial = np.eye(2 * n_anc, dtype=np.float64).reshape(-1)
    solution = solve_ivp(
        deriv,
        (float(start_time), float(end_time)),
        initial,
        method="DOP853",
        dense_output=dense_output,
        rtol=1e-9,
        atol=1e-11,
    )
    if not solution.success:
        raise RuntimeError(f"ODE integration failed: {solution.message}")
    transition = solution.y[:, -1].reshape((2 * n_anc, 2 * n_anc))
    transition = np.clip(np.real(transition), 0.0, np.inf)
    return transition, solution.sol if dense_output else None


def matrix_exp_eig(
    Z: np.ndarray,
    t: float,
) -> np.ndarray:
    """Matrix exponential exp(t*Z) via eigendecomposition.

    Returns the real part. Falls back to ``scipy.linalg.expm`` style for
    numerical stability with very small or zero matrices.
    """
    if Z.shape[0] == 0:
        return np.zeros_like(Z)
    if t == 0:
        return np.eye(Z.shape[0], dtype=np.float64)

    eigvals, eigvecs = eig(Z)
    try:
        eigvecs_inv = np.linalg.inv(eigvecs)
    except np.linalg.LinAlgError:
        # Pseudo-inverse fallback
        eigvecs_inv = np.linalg.pinv(eigvecs)

    # Result = V @ diag(exp(t*lambda)) @ V^-1
    exp_lambda = np.exp(t * eigvals)
    result = (eigvecs * exp_lambda) @ eigvecs_inv
    return np.real(result)


@dataclass
class EigDecomp:
    """Eigendecomposition of a rate matrix.

    Stores eigenvalues, eigenvectors, and the precomputed VW outer products
    used for the H-integral expansions.
    """

    eigvals: np.ndarray  # (n,) complex
    eigvecs: np.ndarray  # (n, n) complex
    eigvecs_inv: np.ndarray  # (n, n) complex
    VW: np.ndarray  # (n, n, n) complex — VW[k] = v_k ⊗ w_k^T

    @classmethod
    def from_matrix(cls, Z: np.ndarray) -> "EigDecomp":
        if Z.shape[0] == 0:
            return cls(
                eigvals=np.zeros(0, dtype=complex),
                eigvecs=np.zeros((0, 0), dtype=complex),
                eigvecs_inv=np.zeros((0, 0), dtype=complex),
                VW=np.zeros((0, 0, 0), dtype=complex),
            )
        eigvals, eigvecs = eig(Z)
        try:
            eigvecs_inv = np.linalg.inv(eigvecs)
        except np.linalg.LinAlgError:
            eigvecs_inv = np.linalg.pinv(eigvecs)

        n = Z.shape[0]
        VW = np.empty((n, n, n), dtype=complex)
        for k in range(n):
            VW[k] = np.outer(eigvecs[:, k], eigvecs_inv[k, :])

        return cls(
            eigvals=eigvals,
            eigvecs=eigvecs,
            eigvecs_inv=eigvecs_inv,
            VW=VW,
        )


def h_integral(
    a: float,
    b: float,
    u: complex,
    lam: complex,
) -> complex:
    """The diCal2 H integral: ∫_a^b exp(u + lam*t) dt.

    Handles edge cases:

    * a ≈ b → 0
    * |lam| ≈ 0 → exp(u) * (b - a)  (or +inf if b is infinite)
    * b = +inf with Re(lam) < 0 → -exp(u + lam*a) / lam
    * Otherwise: (exp(u + lam*b) - exp(u + lam*a)) / lam
    """
    if abs(b - a) < EPS:
        return 0.0 + 0.0j
    if abs(lam) < EPS:
        if np.isinf(b):
            return float("inf") + 0.0j
        return np.exp(u) * (b - a)
    if np.isinf(b):
        # Assume Re(lam) < 0 for convergence
        if np.real(lam) >= 0:
            return float("inf") + 0.0j
        return -np.exp(u + lam * a) / lam
    return (np.exp(u + lam * b) - np.exp(u + lam * a)) / lam


# ===========================================================================
# Core HMM matrix computation
# ===========================================================================


@dataclass
class CoreMatrices:
    """Precomputed HMM matrices for one CSD configuration."""

    n_states: int  # number of demoStates (refined_interval, ancient_deme)
    state_interval: np.ndarray  # (n_states,) → refined interval index
    state_ancient: np.ndarray  # (n_states,) → ancient deme index
    log_initial: np.ndarray  # (n_states,) initial log probabilities
    log_no_reco: np.ndarray  # (n_states,) log P(no recombination)
    log_reco: np.ndarray  # (n_states, n_states) log P(reco src→dst)
    log_emission: np.ndarray  # (n_states, n_alleles, n_alleles) [trunk][obs]


class EigenCore:
    """Compute HMM transition and emission matrices via eigendecomposition.

    Implements the diCal2 EigenCore: builds extended migration matrices for
    each refined interval, computes the f / P / Q transition probabilities,
    and assembles the conditional log-transition and log-emission matrices
    used by the forward-backward algorithm.
    """

    def __init__(
        self,
        refined: RefinedDemography,
        trunk: SimpleTrunk,
        observed_present_deme: int,
        mutation_matrix: np.ndarray,
        theta: float,
        rho: float,
    ):
        self.refined = refined
        self.trunk = trunk
        self.observed_present_deme = observed_present_deme
        self.mutation_matrix = mutation_matrix
        self.theta = theta
        self.rho = rho

        self.n_alleles = mutation_matrix.shape[0]
        self._eig_mut = EigDecomp.from_matrix(mutation_matrix)
        self._mutation_generator = self.theta * (
            self.mutation_matrix - np.eye(self.n_alleles, dtype=np.float64)
        )

        self._build_per_interval_data()
        self._build_states()
        self._compute_p_matrices()
        self._compute_q_matrices()
        self._compute_initial_log_probs()
        self._compute_log_no_reco()
        self._compute_log_reco()
        self._compute_log_emission()

    # ----- per-interval rate matrices -----
    @staticmethod
    def _epoch_pop_sizes_at_time(epoch: DiCal2Epoch, t: float) -> np.ndarray:
        if epoch.pop_sizes is None:
            return np.ones(len(epoch.partition), dtype=np.float64)
        pop_sizes = epoch.pop_sizes.astype(np.float64, copy=True)
        if epoch.growth_rates is None or np.isinf(epoch.end):
            return pop_sizes
        return pop_sizes * np.exp(epoch.growth_rates * (epoch.end - t))

    def _build_per_interval_data(self) -> None:
        n_ref = self.refined.n_refined
        self.alpha_per_interval: list[np.ndarray] = []
        self.M_per_interval: list[np.ndarray | None] = []
        self.partition_per_interval: list[list[list[int]]] = []
        self.popsize_per_interval: list[np.ndarray] = []
        self.Z_per_interval: list[np.ndarray] = []
        self.eig_per_interval: list[EigDecomp] = []
        self.f_per_interval: list[np.ndarray] = []  # exp(dt * Z)
        self.n_anc_per_interval: list[int] = []

        for i in range(n_ref):
            e_idx = self.refined.epoch_map[i]
            epoch = self.refined.demo.epochs[e_idx]
            partition = epoch.partition
            n_anc = len(partition)
            dt = self.refined.interval_duration(i)
            t0 = self.refined.refined_boundaries[i]
            t1 = self.refined.refined_boundaries[i + 1]
            t_mid = t0 + 0.5 * dt if np.isfinite(t1) else t0
            pop_sizes = self._epoch_pop_sizes_at_time(epoch, t_mid)
            alpha = self.trunk.absorption_rates(partition, pop_sizes)
            Z = build_extended_matrix(epoch.migration_matrix, alpha)

            f = matrix_exp_eig(Z, dt)
            eig_decomp = EigDecomp.from_matrix(Z)

            self.alpha_per_interval.append(alpha)
            self.M_per_interval.append(epoch.migration_matrix)
            self.partition_per_interval.append(partition)
            self.popsize_per_interval.append(pop_sizes)
            self.Z_per_interval.append(Z)
            self.eig_per_interval.append(eig_decomp)
            self.f_per_interval.append(f)
            self.n_anc_per_interval.append(n_anc)

    # ----- enumerate hidden states (refined_interval, ancient_deme) -----
    def _build_states(self) -> None:
        n_ref = self.refined.n_refined
        state_interval = []
        state_ancient = []
        for i in range(n_ref):
            for a in range(self.n_anc_per_interval[i]):
                state_interval.append(i)
                state_ancient.append(a)
        self.state_interval = np.array(state_interval, dtype=np.int64)
        self.state_ancient = np.array(state_ancient, dtype=np.int64)
        self.n_states = len(state_interval)

        # Map (interval, ancient) → state index
        self.state_index_map: dict[tuple[int, int], int] = {}
        for s in range(self.n_states):
            self.state_index_map[
                (int(self.state_interval[s]), int(self.state_ancient[s]))
            ] = s

    # ----- multi-interval non-absorption probabilities P -----
    def _compute_p_matrices(self) -> None:
        """P[i][j] = (n_anc[i] x n_anc[j]) non-absorbed transition probabilities."""
        n_ref = self.refined.n_refined
        self.P: list[list[np.ndarray]] = [[None] * n_ref for _ in range(n_ref)]
        for i in range(n_ref):
            self.P[i][i] = np.eye(self.n_anc_per_interval[i], dtype=np.float64)
            for j in range(i + 1, n_ref):
                k_i = self.n_anc_per_interval[i]
                k_jm1 = self.n_anc_per_interval[j - 1]
                k_j = self.n_anc_per_interval[j]
                f_jm1 = self.f_per_interval[j - 1]  # (2*k_jm1, 2*k_jm1)
                # Non-absorbing block of f_jm1: (k_jm1, k_jm1)
                f_non = f_jm1[:k_jm1, :k_jm1]
                # P[i][j-1]: (k_i, k_jm1)
                # We need to transition from k_jm1 demes to k_j demes
                # using the partition mapping (which present demes belong to which ancient).
                # Build a mapping matrix: when going from epoch e1 to e2 (e2 > e1),
                # multiple demes can merge. The mapping is: child[a_jm1] → parent[a_j]
                # if all present demes of a_jm1 are in a_j.
                merge_map = np.zeros((k_jm1, k_j), dtype=np.float64)
                e1 = self.refined.epoch_map[j - 1]
                e2 = self.refined.epoch_map[j]
                if e1 == e2:
                    merge_map[: min(k_jm1, k_j), : min(k_jm1, k_j)] = np.eye(
                        min(k_jm1, k_j)
                    )
                else:
                    part1 = self.partition_per_interval[j - 1]
                    part2 = self.partition_per_interval[j]
                    for a1 in range(k_jm1):
                        s1 = set(part1[a1])
                        for a2 in range(k_j):
                            s2 = set(part2[a2])
                            if s1.issubset(s2):
                                merge_map[a1, a2] = 1.0
                                break
                self.P[i][j] = self.P[i][j - 1] @ f_non @ merge_map

    # ----- absorption probabilities Q[i][j][s][a] -----
    def _compute_q_matrices(self) -> None:
        n_ref = self.refined.n_refined
        self.Q: list[list[np.ndarray]] = [[None] * n_ref for _ in range(n_ref)]
        for i in range(n_ref):
            for j in range(i, n_ref):
                k_j = self.n_anc_per_interval[j]
                f_j = self.f_per_interval[j]
                # Absorption part of f_j: top-right block (k_j, k_j)
                f_abs = f_j[:k_j, k_j : 2 * k_j]
                # P[i][j] @ f_abs
                self.Q[i][j] = self.P[i][j] @ f_abs

    # ----- initial state probabilities -----
    def _compute_initial_log_probs(self) -> None:
        """logP(absorption in (interval, ancient_deme) | observed deme)."""
        n_ref = self.refined.n_refined
        log_initial = np.full(self.n_states, LOG_ZERO, dtype=np.float64)

        # The observed (additional) lineage starts in observed_present_deme.
        # In the first interval, it belongs to whichever ancient deme contains
        # observed_present_deme.
        partition_0 = self.partition_per_interval[0]
        start_anc = -1
        for a, group in enumerate(partition_0):
            if self.observed_present_deme in group:
                start_anc = a
                break
        if start_anc < 0:
            # Observed deme is not in the partition — shouldn't happen
            start_anc = 0
        self.start_anc = start_anc

        for i in range(n_ref):
            Q_i = self.Q[0][i]  # (n_anc[0], n_anc[i])
            for a in range(self.n_anc_per_interval[i]):
                p = Q_i[start_anc, a]
                if p > 0:
                    s = self.state_index_map[(i, a)]
                    log_initial[s] = np.log(max(p, np.exp(LOG_ZERO)))

        # Normalize so they sum to 1
        m = np.max(log_initial)
        if np.isfinite(m):
            log_total = m + np.log(np.sum(np.exp(log_initial - m)))
            log_initial -= log_total
        self.log_initial = log_initial

    def _is_non_absorbing(self, interval: int, ancient_deme: int) -> bool:
        return self.alpha_per_interval[interval][ancient_deme] <= EPS

    def _compute_exact_ancient_no_reco_joint_log(
        self,
        recombination_rate: float,
    ) -> list[np.ndarray]:
        h_per_interval: list[np.ndarray | None] = []
        for interval, eig_decomp in enumerate(self.eig_per_interval):
            if self.refined.is_pulse(interval):
                h_per_interval.append(None)
                continue
            interval_start = self.refined.refined_boundaries[interval]
            interval_end = self.refined.refined_boundaries[interval + 1]
            curr_h = np.zeros(len(eig_decomp.eigvals), dtype=np.complex128)
            for eig_idx, eigval in enumerate(eig_decomp.eigvals):
                curr_h[eig_idx] = h_integral(
                    interval_start,
                    interval_end,
                    -interval_start * eigval,
                    eigval - recombination_rate,
                )
            h_per_interval.append(curr_h)

        joint_logs: list[np.ndarray] = []
        for epoch in range(self.refined.n_refined):
            n_anc = self.n_anc_per_interval[epoch]
            epoch_joint = np.full(n_anc, LOG_ZERO, dtype=np.float64)
            if self.refined.is_pulse(epoch):
                joint_logs.append(epoch_joint)
                continue
            eig_decomp = self.eig_per_interval[epoch]
            h_list = h_per_interval[epoch]
            assert h_list is not None
            for ancient_deme in range(n_anc):
                if self._is_non_absorbing(epoch, ancient_deme):
                    continue
                absorb_idx = n_anc + ancient_deme
                total = 0.0 + 0.0j
                for eig_idx, eigval in enumerate(eig_decomp.eigvals):
                    lambda_h = eigval * h_list[eig_idx]
                    for start_deme in range(n_anc):
                        total += (
                            self.P[0][epoch][self.start_anc, start_deme]
                            * lambda_h
                            * eig_decomp.VW[eig_idx][start_deme, absorb_idx]
                        )
                epoch_joint[ancient_deme] = np.log(max(float(np.real(total)), 1e-300))
            joint_logs.append(epoch_joint)
        return joint_logs

    def _r_term(
        self,
        start_ancient_deme: int,
        interval: int,
        first_end_ancient_deme: int,
        second_end_ancient_deme: int,
        recombination_rate: float,
        h_per_interval: list[np.ndarray | None],
        cache: dict[tuple[int, int, int, int], float],
    ) -> float:
        if self.refined.is_pulse(interval):
            return 0.0
        key = (
            start_ancient_deme,
            interval,
            first_end_ancient_deme,
            second_end_ancient_deme,
        )
        if key in cache:
            return cache[key]

        eig_decomp = self.eig_per_interval[interval]
        h_tensor = h_per_interval[interval]
        assert h_tensor is not None
        n_anc = self.n_anc_per_interval[interval]

        total = 0.0 + 0.0j
        for reco_ancient_deme in range(n_anc):
            for k in range(len(eig_decomp.eigvals)):
                for m in range(len(eig_decomp.eigvals)):
                    for n in range(len(eig_decomp.eigvals)):
                        total += (
                            eig_decomp.VW[k][start_ancient_deme, reco_ancient_deme]
                            * eig_decomp.VW[m][reco_ancient_deme, first_end_ancient_deme]
                            * eig_decomp.VW[n][reco_ancient_deme, second_end_ancient_deme]
                            * h_tensor[k, m, n]
                        )
        value = recombination_rate * float(np.real(total))
        cache[key] = max(value, 0.0)
        return cache[key]

    def _w_term(
        self,
        start_ancient_deme: int,
        first_absorb_interval: int,
        first_absorb_ancient_deme: int,
        second_absorb_interval: int,
        second_absorb_ancient_deme: int,
        recombination_rate: float,
        h_per_interval: list[np.ndarray | None],
        cache: dict[tuple[int, int, int, int], float],
    ) -> float:
        if second_absorb_interval < first_absorb_interval:
            return self._w_term(
                start_ancient_deme,
                second_absorb_interval,
                second_absorb_ancient_deme,
                first_absorb_interval,
                first_absorb_ancient_deme,
                recombination_rate,
                h_per_interval,
                cache,
            )

        n_anc = self.n_anc_per_interval[first_absorb_interval]
        if first_absorb_interval == second_absorb_interval:
            return self._r_term(
                start_ancient_deme,
                first_absorb_interval,
                n_anc + first_absorb_ancient_deme,
                n_anc + second_absorb_ancient_deme,
                recombination_rate,
                h_per_interval,
                cache,
            )

        result = 0.0
        next_interval = first_absorb_interval + 1
        next_n_anc = self.n_anc_per_interval[next_interval]
        for next_start_ancient_deme in range(next_n_anc):
            tmp = 0.0
            for member_ancient_deme in range(n_anc):
                if set(self.partition_per_interval[first_absorb_interval][member_ancient_deme]).issubset(
                    set(self.partition_per_interval[next_interval][next_start_ancient_deme])
                ):
                    tmp += self._r_term(
                        start_ancient_deme,
                        first_absorb_interval,
                        n_anc + first_absorb_ancient_deme,
                        member_ancient_deme,
                        recombination_rate,
                        h_per_interval,
                        cache,
                    )
            tmp *= self.Q[next_interval][second_absorb_interval][
                next_start_ancient_deme, second_absorb_ancient_deme
            ]
            result += tmp
        return max(result, 0.0)

    def _compute_exact_ancient_reco_joint_log(
        self,
        recombination_rate: float,
    ) -> list[list[np.ndarray]]:
        h_per_interval: list[np.ndarray | None] = []
        for interval, eig_decomp in enumerate(self.eig_per_interval):
            if self.refined.is_pulse(interval):
                h_per_interval.append(None)
                continue
            interval_start = self.refined.refined_boundaries[interval]
            interval_end = self.refined.refined_boundaries[interval + 1]
            n_eig = len(eig_decomp.eigvals)
            curr_h = np.zeros((n_eig, n_eig, n_eig), dtype=np.complex128)
            for k, lambda_k in enumerate(eig_decomp.eigvals):
                for m, lambda_m in enumerate(eig_decomp.eigvals):
                    for n, lambda_n in enumerate(eig_decomp.eigvals):
                        curr_h[k, m, n] = h_integral(
                            interval_start,
                            interval_end,
                            interval_end * (lambda_m + lambda_n) - interval_start * lambda_k,
                            lambda_k - lambda_m - lambda_n - recombination_rate,
                        )
            h_per_interval.append(curr_h)

        r_cache: dict[tuple[int, int, int, int], float] = {}
        joint_logs: list[list[np.ndarray]] = []
        for prev_absorb_epoch in range(self.refined.n_refined):
            prev_epoch_rows: list[np.ndarray] = []
            joint_logs.append(prev_epoch_rows)
            for next_absorb_epoch in range(self.refined.n_refined):
                curr = np.full(
                    (
                        self.n_anc_per_interval[prev_absorb_epoch],
                        self.n_anc_per_interval[next_absorb_epoch],
                    ),
                    LOG_ZERO,
                    dtype=np.float64,
                )
                prev_epoch_rows.append(curr)
                for prev_absorb_deme in range(curr.shape[0]):
                    for next_absorb_deme in range(curr.shape[1]):
                        if self._is_non_absorbing(prev_absorb_epoch, prev_absorb_deme) or self._is_non_absorbing(
                            next_absorb_epoch, next_absorb_deme
                        ):
                            continue
                        min_absorb_epoch = min(prev_absorb_epoch, next_absorb_epoch)
                        pre_z = 0.0
                        for reco_interval in range(min_absorb_epoch):
                            summand = 0.0
                            curr_n_anc = self.n_anc_per_interval[reco_interval]
                            next_n_anc = self.n_anc_per_interval[reco_interval + 1]
                            for reco_interval_start_deme in range(curr_n_anc):
                                for prev_next_start_deme in range(next_n_anc):
                                    prev_members = [
                                        member
                                        for member in range(curr_n_anc)
                                        if set(self.partition_per_interval[reco_interval][member]).issubset(
                                            set(self.partition_per_interval[reco_interval + 1][prev_next_start_deme])
                                        )
                                    ]
                                    for next_next_start_deme in range(next_n_anc):
                                        next_members = [
                                            member
                                            for member in range(curr_n_anc)
                                            if set(self.partition_per_interval[reco_interval][member]).issubset(
                                                set(self.partition_per_interval[reco_interval + 1][next_next_start_deme])
                                            )
                                        ]
                                        for prev_member_deme in prev_members:
                                            for next_member_deme in next_members:
                                                tmp = self.P[0][reco_interval][
                                                    self.start_anc,
                                                    reco_interval_start_deme,
                                                ]
                                                tmp *= self._r_term(
                                                    reco_interval_start_deme,
                                                    reco_interval,
                                                    prev_member_deme,
                                                    next_member_deme,
                                                    recombination_rate,
                                                    h_per_interval,
                                                    r_cache,
                                                )
                                                tmp *= self.Q[reco_interval + 1][prev_absorb_epoch][
                                                    prev_next_start_deme,
                                                    prev_absorb_deme,
                                                ]
                                                tmp *= self.Q[reco_interval + 1][next_absorb_epoch][
                                                    next_next_start_deme,
                                                    next_absorb_deme,
                                                ]
                                                summand += tmp
                            pre_z += summand

                        summand = 0.0
                        for reco_interval_start_deme in range(
                            self.n_anc_per_interval[min_absorb_epoch]
                        ):
                            tmp = self.P[0][min_absorb_epoch][
                                self.start_anc,
                                reco_interval_start_deme,
                            ]
                            tmp *= self._w_term(
                                reco_interval_start_deme,
                                prev_absorb_epoch,
                                prev_absorb_deme,
                                next_absorb_epoch,
                                next_absorb_deme,
                                recombination_rate,
                                h_per_interval,
                                r_cache,
                            )
                            summand += tmp
                        pre_z += summand
                        curr[prev_absorb_deme, next_absorb_deme] = np.log(max(pre_z, 1e-300))
        return joint_logs

    def _compute_exact_ancient_emission_log_probs(self) -> list[np.ndarray]:
        h_per_interval: list[np.ndarray | None] = []
        for interval, eig_migration in enumerate(self.eig_per_interval):
            if self.refined.is_pulse(interval):
                h_per_interval.append(None)
                continue
            interval_start = self.refined.refined_boundaries[interval]
            interval_end = self.refined.refined_boundaries[interval + 1]
            curr_h = np.zeros(
                (len(eig_migration.eigvals), len(self._eig_mut.eigvals)),
                dtype=np.complex128,
            )
            for k, lambda_k in enumerate(eig_migration.eigvals):
                for j, lambda_j in enumerate(self._eig_mut.eigvals):
                    curr_h[k, j] = h_integral(
                        interval_start,
                        interval_end,
                        -interval_start * lambda_k,
                        self.theta * (lambda_j - 1.0) + lambda_k,
                    )
            h_per_interval.append(curr_h)

        log_joint: list[np.ndarray] = []
        for epoch in range(self.refined.n_refined):
            n_anc = self.n_anc_per_interval[epoch]
            curr = np.full(
                (n_anc, self.n_alleles, self.n_alleles),
                LOG_ZERO,
                dtype=np.float64,
            )
            log_joint.append(curr)
            if self.refined.is_pulse(epoch):
                continue
            eig_migration = self.eig_per_interval[epoch]
            curr_h = h_per_interval[epoch]
            assert curr_h is not None
            for ancient_deme in range(n_anc):
                if self._is_non_absorbing(epoch, ancient_deme):
                    continue
                for emission_type in range(self.n_alleles):
                    for trunk_type in range(self.n_alleles):
                        total = 0.0 + 0.0j
                        for absorb_start_deme in range(n_anc):
                            pre_sum = 0.0 + 0.0j
                            for k, lambda_k in enumerate(eig_migration.eigvals):
                                for j in range(len(self._eig_mut.eigvals)):
                                    pre_sum += (
                                        lambda_k
                                        * curr_h[k, j]
                                        * self._eig_mut.VW[j][trunk_type, emission_type]
                                        * eig_migration.VW[k][
                                            absorb_start_deme, n_anc + ancient_deme
                                        ]
                                    )
                            total += self.P[0][epoch][self.start_anc, absorb_start_deme] * pre_sum
                        curr[ancient_deme, trunk_type, emission_type] = np.log(
                            max(float(np.real(total)), 1e-300)
                        )
        return log_joint

    # ----- log P(no recombination) per state -----
    def _compute_log_no_reco(self) -> None:
        """Per-state probability of no recombination across one site step.

        For a state s = (i, a) in interval i, ancient deme a:

            P(no reco | s) = exp(-rho * t_a) integrated over the absorption
            density within interval i.

        We approximate this with the *interval midpoint* coalescence time,
        which gives the leading-order behaviour and matches diCal2 to high
        precision for finely refined grids.
        """
        if all(
            ep.growth_rates is None or np.allclose(ep.growth_rates, 0.0)
            for ep in self.refined.demo.epochs
        ):
            log_no_reco = np.full(self.n_states, LOG_ZERO, dtype=np.float64)
            ancient_joint = self._compute_exact_ancient_no_reco_joint_log(self.rho)
            for s in range(self.n_states):
                i = int(self.state_interval[s])
                a = int(self.state_ancient[s])
                if ancient_joint[i][a] <= LOG_ZERO / 2 or self.log_initial[s] <= LOG_ZERO / 2:
                    continue
                log_no_reco[s] = ancient_joint[i][a] - self.log_initial[s]
            self.log_no_reco = log_no_reco
            return

        log_no_reco = np.zeros(self.n_states, dtype=np.float64)
        for s in range(self.n_states):
            i = int(self.state_interval[s])
            t0 = self.refined.refined_boundaries[i]
            t1 = self.refined.refined_boundaries[i + 1]
            if np.isinf(t1):
                # Use t0 + 1/alpha as the expected coalescence time
                a = int(self.state_ancient[s])
                alpha = self.alpha_per_interval[i][a]
                t_mid = t0 + (1.0 / max(alpha, EPS))
            else:
                t_mid = 0.5 * (t0 + t1)
            log_no_reco[s] = -self.rho * t_mid
        self.log_no_reco = log_no_reco

    # ----- log P(reco src→dst) -----
    def _compute_log_reco(self) -> None:
        """Recombination transition matrix.

        After a recombination event, the lineage detaches and re-coalesces
        with the trunk. We approximate the destination distribution as the
        marginal initial distribution (independent of the source state),
        which is the leading-order term in diCal2's R/W expansion and is
        widely used as the SMC' approximation.
        """
        if all(
            ep.growth_rates is None or np.allclose(ep.growth_rates, 0.0)
            for ep in self.refined.demo.epochs
        ):
            log_reco = np.full((self.n_states, self.n_states), LOG_ZERO, dtype=np.float64)
            ancient_joint = self._compute_exact_ancient_reco_joint_log(self.rho)
            for src in range(self.n_states):
                src_i = int(self.state_interval[src])
                src_a = int(self.state_ancient[src])
                if self.log_initial[src] <= LOG_ZERO / 2:
                    continue
                for dst in range(self.n_states):
                    dst_i = int(self.state_interval[dst])
                    dst_a = int(self.state_ancient[dst])
                    joint = ancient_joint[src_i][dst_i][src_a, dst_a]
                    if joint <= LOG_ZERO / 2:
                        continue
                    log_reco[src, dst] = joint - self.log_initial[src]
            self.log_reco = log_reco
            return

        # P(reco) = 1 - exp(log_no_reco)
        # The destination distribution is approximated as the initial
        # marginal: p_dst = exp(log_initial[dst]).
        log_reco = np.zeros((self.n_states, self.n_states), dtype=np.float64)
        for src in range(self.n_states):
            log_p_reco = np.log1p(-min(np.exp(self.log_no_reco[src]), 1.0 - EPS))
            for dst in range(self.n_states):
                log_reco[src, dst] = log_p_reco + self.log_initial[dst]
        self.log_reco = log_reco

    # ----- emission matrices -----
    def _compute_log_emission(self) -> None:
        """Emission probabilities P(observed allele | trunk allele, state).

        Computed via the matrix exponential of the mutation rate matrix
        scaled by the expected coalescence time of each state.
        """
        if all(
            ep.growth_rates is None or np.allclose(ep.growth_rates, 0.0)
            for ep in self.refined.demo.epochs
        ):
            log_em = np.full(
                (self.n_states, self.n_alleles, self.n_alleles),
                LOG_ZERO,
                dtype=np.float64,
            )
            joint = self._compute_exact_ancient_emission_log_probs()
            for s in range(self.n_states):
                i = int(self.state_interval[s])
                a = int(self.state_ancient[s])
                if self.log_initial[s] <= LOG_ZERO / 2:
                    continue
                log_em[s] = joint[i][a] - self.log_initial[s]
                for trunk_type in range(self.n_alleles):
                    row = log_em[s, trunk_type]
                    row -= logsumexp(row)
                    log_em[s, trunk_type] = row
            self.log_emission = log_em
            return

        n_alleles = self.n_alleles
        log_em = np.full(
            (self.n_states, n_alleles, n_alleles),
            LOG_ZERO,
            dtype=np.float64,
        )

        for s in range(self.n_states):
            i = int(self.state_interval[s])
            t0 = self.refined.refined_boundaries[i]
            t1 = self.refined.refined_boundaries[i + 1]
            if np.isinf(t1):
                a = int(self.state_ancient[s])
                alpha = self.alpha_per_interval[i][a]
                t_mid = t0 + (1.0 / max(alpha, EPS))
            else:
                t_mid = 0.5 * (t0 + t1)

            # Two parents at time t_mid → 2*t_mid total mutation time
            P_mut = matrix_exp_eig(self._mutation_generator, 2.0 * t_mid)
            # Clip to be positive then take log
            P_mut = np.clip(P_mut, 1e-300, 1.0)
            # Renormalize rows
            row_sum = P_mut.sum(axis=1, keepdims=True)
            P_mut = P_mut / np.maximum(row_sum, 1e-300)
            log_em[s] = np.log(P_mut)

        self.log_emission = log_em

    # ----- assemble core matrices -----
    def core_matrices(self) -> CoreMatrices:
        return CoreMatrices(
            n_states=self.n_states,
            state_interval=self.state_interval,
            state_ancient=self.state_ancient,
            log_initial=self.log_initial,
            log_no_reco=self.log_no_reco,
            log_reco=self.log_reco,
            log_emission=self.log_emission,
        )


class ODECore(EigenCore):
    """Growth-aware native core using explicit ODE integration per interval."""

    def _build_per_interval_data(self) -> None:
        n_ref = self.refined.n_refined
        self.alpha_per_interval: list[np.ndarray] = []
        self.M_per_interval: list[np.ndarray | None] = []
        self.partition_per_interval: list[list[list[int]]] = []
        self.popsize_per_interval: list[np.ndarray] = []
        self.Z_per_interval: list[np.ndarray] = []
        self.eig_per_interval: list[EigDecomp] = []
        self.f_per_interval: list[np.ndarray] = []
        self._dense_transition_solutions: list[object | None] = []
        self._f_no_reco_per_interval: list[np.ndarray | None] = []
        self.n_anc_per_interval: list[int] = []

        for interval in range(n_ref):
            epoch_idx = self.refined.epoch_map[interval]
            epoch = self.refined.demo.epochs[epoch_idx]
            partition = epoch.partition
            n_anc = len(partition)
            interval_start = self.refined.refined_boundaries[interval]
            interval_end = self.refined.refined_boundaries[interval + 1]
            pop_sizes = (
                np.ones(n_anc, dtype=np.float64)
                if epoch.pop_sizes is None
                else np.asarray(epoch.pop_sizes, dtype=np.float64)
            )
            alpha = self.trunk.absorption_rates(partition, pop_sizes)
            f_matrix, dense_solution = _integrate_transition_matrix(
                epoch,
                alpha,
                interval_start,
                interval_end,
                dense_output=not self.refined.is_pulse(interval),
            )
            z_mid = _extended_rate_matrix_at_time(
                epoch,
                alpha,
                interval_start if self.refined.is_pulse(interval) else 0.5 * (interval_start + interval_end),
            )

            self.alpha_per_interval.append(alpha)
            self.M_per_interval.append(epoch.migration_matrix)
            self.partition_per_interval.append(partition)
            self.popsize_per_interval.append(pop_sizes)
            self.Z_per_interval.append(z_mid)
            self.eig_per_interval.append(EigDecomp.from_matrix(z_mid))
            self.f_per_interval.append(f_matrix)
            self._dense_transition_solutions.append(dense_solution)
            self._f_no_reco_per_interval.append(None)
            self.n_anc_per_interval.append(n_anc)

    def _compute_log_no_reco(self) -> None:
        log_no_reco = np.full(self.n_states, LOG_ZERO, dtype=np.float64)
        n_ref = self.refined.n_refined
        p_no: list[list[np.ndarray]] = [[None] * n_ref for _ in range(n_ref)]
        q_no: list[list[np.ndarray]] = [[None] * n_ref for _ in range(n_ref)]

        for interval in range(n_ref):
            p_no[interval][interval] = np.eye(
                self.n_anc_per_interval[interval],
                dtype=np.float64,
            )
            for next_interval in range(interval + 1, n_ref):
                n_curr = self.n_anc_per_interval[next_interval - 1]
                n_next = self.n_anc_per_interval[next_interval]
                f_non = self._f_no_reco(next_interval - 1)[:n_curr, :n_curr]
                merge_map = np.zeros((n_curr, n_next), dtype=np.float64)
                e1 = self.refined.epoch_map[next_interval - 1]
                e2 = self.refined.epoch_map[next_interval]
                if e1 == e2:
                    merge_map[: min(n_curr, n_next), : min(n_curr, n_next)] = np.eye(
                        min(n_curr, n_next)
                    )
                else:
                    part1 = self.partition_per_interval[next_interval - 1]
                    part2 = self.partition_per_interval[next_interval]
                    for ancient_one in range(n_curr):
                        members_one = set(part1[ancient_one])
                        for ancient_two in range(n_next):
                            members_two = set(part2[ancient_two])
                            if members_one.issubset(members_two):
                                merge_map[ancient_one, ancient_two] = 1.0
                                break
                p_no[interval][next_interval] = p_no[interval][next_interval - 1] @ f_non @ merge_map

        for start_interval in range(n_ref):
            for end_interval in range(start_interval, n_ref):
                n_end = self.n_anc_per_interval[end_interval]
                f_abs = self._f_no_reco(end_interval)[:n_end, n_end : 2 * n_end]
                q_no[start_interval][end_interval] = p_no[start_interval][end_interval] @ f_abs

        for state in range(self.n_states):
            interval = int(self.state_interval[state])
            ancient = int(self.state_ancient[state])
            joint = q_no[0][interval][self.start_anc, ancient]
            if joint <= 0 or self.log_initial[state] <= LOG_ZERO / 2:
                continue
            log_no_reco[state] = np.log(max(float(joint), 1e-300)) - self.log_initial[state]
        self.log_no_reco = log_no_reco

    def _f_no_reco(self, interval: int) -> np.ndarray:
        cached = self._f_no_reco_per_interval[interval]
        if cached is not None:
            return cached
        epoch = self.refined.demo.epochs[self.refined.epoch_map[interval]]
        interval_start = self.refined.refined_boundaries[interval]
        interval_end = self.refined.refined_boundaries[interval + 1]
        f_matrix, _ = _integrate_transition_matrix(
            epoch,
            self.alpha_per_interval[interval],
            interval_start,
            interval_end,
            extra_loss=self.rho,
            dense_output=False,
        )
        self._f_no_reco_per_interval[interval] = f_matrix
        return f_matrix

    def _expected_absorption_times(self) -> np.ndarray:
        expected_times = np.zeros(self.n_states, dtype=np.float64)
        for state in range(self.n_states):
            interval = int(self.state_interval[state])
            ancient = int(self.state_ancient[state])
            state_prob = float(np.exp(self.log_initial[state]))
            if state_prob <= EPS or self.refined.is_pulse(interval):
                continue
            solution = self._dense_transition_solutions[interval]
            if solution is None:
                continue
            n_anc = self.n_anc_per_interval[interval]
            interval_start = float(self.refined.refined_boundaries[interval])
            interval_end = float(self.refined.refined_boundaries[interval + 1])
            grid = np.linspace(interval_start, interval_end, num=33, dtype=np.float64)
            start_distribution = np.zeros(n_anc, dtype=np.float64)
            start_distribution[:] = self.P[0][interval][self.start_anc]
            densities = np.zeros_like(grid)
            for grid_idx, time_point in enumerate(grid):
                transition = np.asarray(solution(float(time_point))).reshape((2 * n_anc, 2 * n_anc))
                non_absorbing = np.clip(
                    np.real(transition[:n_anc, :n_anc]),
                    0.0,
                    np.inf,
                )
                non_absorbing_dist = start_distribution @ non_absorbing
                absorption_rates = _absorption_rates_at_time(
                    self.alpha_per_interval[interval],
                    self.refined.demo.epochs[self.refined.epoch_map[interval]],
                    float(time_point),
                )
                densities[grid_idx] = non_absorbing_dist[ancient] * absorption_rates[ancient]
            numerator = np.trapezoid(grid * densities, grid)
            denominator = np.trapezoid(densities, grid)
            if denominator > EPS:
                expected_times[state] = float(numerator / denominator)
            else:
                expected_times[state] = 0.5 * (interval_start + interval_end)
        return expected_times

    def _compute_log_emission(self) -> None:
        log_em = np.full(
            (self.n_states, self.n_alleles, self.n_alleles),
            LOG_ZERO,
            dtype=np.float64,
        )
        expected_times = self._expected_absorption_times()
        for state in range(self.n_states):
            if self.log_initial[state] <= LOG_ZERO / 2:
                continue
            t_mean = max(float(expected_times[state]), 0.0)
            p_mut = matrix_exp_eig(self._mutation_generator, 2.0 * t_mean)
            p_mut = np.clip(np.real(p_mut), 1e-300, 1.0)
            p_mut /= np.maximum(p_mut.sum(axis=1, keepdims=True), 1e-300)
            log_em[state] = np.log(p_mut)
        self.log_emission = log_em


def _build_native_core(
    refined: RefinedDemography,
    trunk: SimpleTrunk,
    observed_present_deme: int,
    mutation_matrix: np.ndarray,
    theta: float,
    rho: float,
) -> tuple[EigenCore | ODECore, str]:
    has_growth = _refined_has_growth(refined)
    if has_growth and _refined_has_pulse(refined):
        raise NotImplementedError(
            "Native diCal2 does not support combining pulse migration with ODECore."
        )
    if has_growth:
        return (
            ODECore(
                refined=refined,
                trunk=trunk,
                observed_present_deme=observed_present_deme,
                mutation_matrix=mutation_matrix,
                theta=theta,
                rho=rho,
            ),
            "ode",
        )
    return (
        EigenCore(
            refined=refined,
            trunk=trunk,
            observed_present_deme=observed_present_deme,
            mutation_matrix=mutation_matrix,
            theta=theta,
            rho=rho,
        ),
        "eigen",
    )


# ===========================================================================
# Forward-backward algorithm (log-space, with reco/noReco split)
# ===========================================================================


def forward_log(
    core: CoreMatrices,
    obs_additional: np.ndarray,
    obs_trunk: np.ndarray,
    step_sizes: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Log-space forward algorithm.

    Parameters
    ----------
    core : CoreMatrices
        Precomputed HMM matrices.
    obs_additional : (L,) int array
        Allele observed at each locus on the additional lineage.
    obs_trunk : (L,) int array
        Allele observed at each locus on the trunk consensus.

    Returns
    -------
    logF : (L, n_states) array
    log_likelihood : float
    """
    L = len(obs_additional)
    S = core.n_states
    if step_sizes is None:
        step_sizes = np.ones(L, dtype=np.int64)
    logF = np.full((L, S), LOG_ZERO, dtype=np.float64)

    # Locus 0
    em0 = _block_emission_log(core, obs_additional[0], obs_trunk[0])
    logF[0] = core.log_initial + em0

    for ll in range(1, L):
        prev = logF[ll - 1]
        log_no_reco_step, log_reco_step = _scaled_transition_logs(core, int(step_sizes[ll - 1]))
        # Recombination contribution: logsumexp over src of log_reco[src, dst] + prev[src]
        # Vectorized: log_reco.T @ exp(prev) in log space
        # logR[dst] = logsumexp(log_reco[:, dst] + prev)
        logR = logsumexp(log_reco_step + prev[:, None], axis=0)
        # No-recombination contribution
        log_nr = log_no_reco_step + prev
        # Combine
        combined = np.logaddexp(log_nr, logR)
        em = _block_emission_log(core, obs_additional[ll], obs_trunk[ll])
        logF[ll] = combined + em

    log_likelihood = logsumexp(logF[L - 1])
    return logF, log_likelihood


def backward_log(
    core: CoreMatrices,
    obs_additional: np.ndarray,
    obs_trunk: np.ndarray,
    step_sizes: np.ndarray | None = None,
) -> np.ndarray:
    """Log-space backward algorithm."""
    L = len(obs_additional)
    S = core.n_states
    if step_sizes is None:
        step_sizes = np.ones(L, dtype=np.int64)
    logB = np.full((L, S), LOG_ZERO, dtype=np.float64)
    logB[L - 1] = 0.0

    for ll in range(L - 2, -1, -1):
        nxt_em = _block_emission_log(core, obs_additional[ll + 1], obs_trunk[ll + 1])
        nxt = logB[ll + 1] + nxt_em
        log_no_reco_step, log_reco_step = _scaled_transition_logs(core, int(step_sizes[ll]))
        # Recombination: from src, sum over dst of log_reco[src, dst] + nxt[dst]
        logR = logsumexp(log_reco_step + nxt[None, :], axis=1)
        log_nr = log_no_reco_step + nxt
        logB[ll] = np.logaddexp(log_nr, logR)
    return logB


def _block_emission_log(
    core: CoreMatrices,
    obs_additional: np.ndarray | int,
    obs_trunk: np.ndarray | int,
) -> np.ndarray:
    S = core.n_states
    add = np.atleast_1d(obs_additional)
    trunk = np.atleast_1d(obs_trunk)
    log_em = np.zeros(S, dtype=np.float64)
    for a_o, a_t in zip(add, trunk):
        if a_t < 0 or a_o < 0:
            continue
        log_em += core.log_emission[np.arange(S), int(a_t), int(a_o)]
    return log_em


def _expanded_state_block_emission_log(
    core: CoreMatrices,
    base_state: int,
    obs_additional: np.ndarray | int,
    obs_trunk: np.ndarray | int,
) -> float:
    add = np.atleast_1d(obs_additional)
    trunk = np.atleast_1d(obs_trunk)
    value = 0.0
    for a_o, a_t in zip(add, trunk):
        if a_t < 0 or a_o < 0:
            continue
        value += core.log_emission[base_state, int(a_t), int(a_o)]
    return float(value)


def _scaled_transition_logs(
    core: CoreMatrices,
    step_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    log_no = core.log_no_reco * step_size
    log_re = np.zeros_like(core.log_reco)
    no_reco_prob = np.exp(np.clip(log_no, -700, 0))
    reco_prob = np.clip(1.0 - no_reco_prob, 1e-300, 1.0)
    for src in range(core.n_states):
        log_re[src] = np.log(reco_prob[src]) + core.log_initial
    return log_no, log_re


def _scaled_transition_logs_expanded(
    expanded: ExpandedCore,
    step_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    log_no = expanded.log_no_reco * step_size
    log_re = np.zeros_like(expanded.log_reco)
    no_reco_prob = np.exp(np.clip(log_no, -700, 0))
    reco_prob = np.clip(1.0 - no_reco_prob, 1e-300, 1.0)
    for src in range(len(expanded.expanded_to_base)):
        dst_base = expanded.expanded_to_base
        log_re[src] = (
            np.log(reco_prob[src])
            + expanded.log_reco[src]
            - logsumexp(expanded.log_reco[src])
        )
    return log_no, log_re


# ===========================================================================
# Expected sufficient statistics (E-step)
# ===========================================================================


@dataclass
class ExpectedCounts:
    """Expected sufficient statistics from a single CSD forward-backward."""

    log_likelihood: float
    initial_expect: np.ndarray  # (n_states,)
    no_reco_expect: np.ndarray  # (n_states,)
    reco_expect: np.ndarray  # (n_states, n_states)
    emission_expect: np.ndarray  # (n_states, n_alleles, n_alleles)


@dataclass
class ExpandedCore:
    """Demo-state x trunk-haplotype expansion of core probabilities."""

    base_to_expanded: list[list[int]]
    expanded_to_base: np.ndarray
    log_initial: np.ndarray
    log_no_reco: np.ndarray
    log_reco: np.ndarray
    trunk_sequences: np.ndarray


def expected_counts(
    core: CoreMatrices,
    obs_additional: np.ndarray,
    obs_trunk: np.ndarray,
    n_alleles: int,
    step_sizes: np.ndarray | None = None,
) -> ExpectedCounts:
    """Run forward-backward and accumulate sufficient statistics."""
    L = len(obs_additional)
    S = core.n_states
    if step_sizes is None:
        step_sizes = np.ones(L, dtype=np.int64)

    logF, ll = forward_log(core, obs_additional, obs_trunk, step_sizes=step_sizes)
    logB = backward_log(core, obs_additional, obs_trunk, step_sizes=step_sizes)

    # Posterior at each locus
    log_post = logF + logB - ll  # (L, S)
    post = np.exp(log_post)

    # Initial expectations
    initial_expect = post[0].copy()

    # Emission expectations
    emission_expect = np.zeros((S, n_alleles, n_alleles), dtype=np.float64)
    for ll_i in range(L):
        add_block = np.atleast_1d(obs_additional[ll_i])
        trunk_block = np.atleast_1d(obs_trunk[ll_i])
        for a_o, a_t in zip(add_block, trunk_block):
            if a_t >= 0 and a_o >= 0:
                emission_expect[:, int(a_t), int(a_o)] += post[ll_i]

    # Transition expectations
    no_reco_expect = np.zeros(S, dtype=np.float64)
    reco_expect = np.zeros((S, S), dtype=np.float64)

    for ll_i in range(L - 1):
        em_next = _block_emission_log(core, obs_additional[ll_i + 1], obs_trunk[ll_i + 1])
        log_no_reco_step, log_reco_step = _scaled_transition_logs(core, int(step_sizes[ll_i]))
        # No recombination: stays in same state
        # xi_noreco[s] = exp(logF[l, s] + log_no_reco[s] + em_next[s] + logB[l+1, s] - ll)
        xi_no = np.exp(
            logF[ll_i] + log_no_reco_step + em_next + logB[ll_i + 1] - ll
        )
        no_reco_expect += xi_no

        # Recombination: src → dst
        # xi_reco[src, dst] = exp(logF[l, src] + log_reco[src, dst] + em_next[dst] + logB[l+1, dst] - ll)
        xi_re = np.exp(
            logF[ll_i][:, None]
            + log_reco_step
            + em_next[None, :]
            + logB[ll_i + 1][None, :]
            - ll
        )
        reco_expect += xi_re

    return ExpectedCounts(
        log_likelihood=ll,
        initial_expect=initial_expect,
        no_reco_expect=no_reco_expect,
        reco_expect=reco_expect,
        emission_expect=emission_expect,
    )


def _build_expanded_core(
    core_obj: EigenCore | ODECore,
    core: CoreMatrices,
    trunk: SimpleTrunk,
    trunk_idxs: list[int],
    grouped_trunk_sequences: dict[int, np.ndarray],
) -> ExpandedCore:
    base_to_expanded: list[list[int]] = [[] for _ in range(core.n_states)]
    expanded_to_base: list[int] = []
    log_initial: list[float] = []
    log_no_reco: list[float] = []
    trunk_seq_rows: list[np.ndarray] = []
    state_present: list[int] = []
    hap_fraction_logs: list[float] = []

    present_totals: dict[int, float] = {}
    for h in trunk_idxs:
        present = trunk.config.haplotype_populations[h]
        if present < 0:
            continue
        present_totals[present] = present_totals.get(present, 0.0) + float(
            trunk.config.haplotype_multiplicities[h, present]
        )

    for base_state in range(core.n_states):
        interval = int(core.state_interval[base_state])
        ancient = int(core.state_ancient[base_state])
        partition = core_obj.partition_per_interval[interval][ancient]
        for present in partition:
            if present_totals.get(present, 0.0) <= 0:
                continue
            frac = trunk.fraction_ancient_to_present(
                core_obj.partition_per_interval[interval],
                present,
                ancient,
            )
            if frac <= 0:
                continue
            for h in trunk_idxs:
                if trunk.config.haplotype_populations[h] != present:
                    continue
                mult = float(trunk.config.haplotype_multiplicities[h, present])
                if mult <= 0:
                    continue
                exp_idx = len(expanded_to_base)
                base_to_expanded[base_state].append(exp_idx)
                expanded_to_base.append(base_state)
                log_initial.append(
                    core.log_initial[base_state]
                    + np.log(frac)
                    + np.log(mult / present_totals[present])
                )
                log_no_reco.append(core.log_no_reco[base_state])
                trunk_seq_rows.append(grouped_trunk_sequences[h].astype(np.int64))
                state_present.append(present)
                hap_fraction_logs.append(np.log(mult / present_totals[present]))

    n_expanded = len(expanded_to_base)
    log_reco = np.full((n_expanded, n_expanded), LOG_ZERO, dtype=np.float64)
    for src in range(n_expanded):
        src_base = expanded_to_base[src]
        for dst in range(n_expanded):
            dst_base = expanded_to_base[dst]
            log_reco[src, dst] = core.log_reco[src_base, dst_base]
            interval = int(core.state_interval[dst_base])
            ancient = int(core.state_ancient[dst_base])
            frac = trunk.fraction_ancient_to_present(
                core_obj.partition_per_interval[interval],
                state_present[dst],
                ancient,
            )
            log_reco[src, dst] += np.log(max(frac, 1e-300)) + hap_fraction_logs[dst]

    return ExpandedCore(
        base_to_expanded=base_to_expanded,
        expanded_to_base=np.array(expanded_to_base, dtype=np.int64),
        log_initial=np.array(log_initial, dtype=np.float64),
        log_no_reco=np.array(log_no_reco, dtype=np.float64),
        log_reco=log_reco,
        trunk_sequences=np.array(trunk_seq_rows, dtype=np.int64),
    )


def _expanded_expected_counts(
    core: CoreMatrices,
    expanded: ExpandedCore,
    obs_additional: np.ndarray,
    n_alleles: int,
    step_sizes: np.ndarray | None = None,
) -> ExpectedCounts:
    L = len(obs_additional)
    H = len(expanded.expanded_to_base)
    if step_sizes is None:
        step_sizes = np.ones(L, dtype=np.int64)

    logF = np.full((L, H), LOG_ZERO, dtype=np.float64)
    em0 = np.zeros(H, dtype=np.float64)
    for h in range(H):
        em0[h] = _expanded_state_block_emission_log(
            core,
            int(expanded.expanded_to_base[h]),
            obs_additional[0],
            expanded.trunk_sequences[h][0],
        )
    logF[0] = expanded.log_initial + em0

    for ll in range(1, L):
        log_no_reco_step, log_reco_step = _scaled_transition_logs_expanded(
            expanded,
            int(step_sizes[ll - 1]),
        )
        logR = logsumexp(log_reco_step + logF[ll - 1][:, None], axis=0)
        log_nr = log_no_reco_step + logF[ll - 1]
        em = np.zeros(H, dtype=np.float64)
        for h in range(H):
            em[h] = _expanded_state_block_emission_log(
                core,
                int(expanded.expanded_to_base[h]),
                obs_additional[ll],
                expanded.trunk_sequences[h][ll],
            )
        logF[ll] = np.logaddexp(log_nr, logR) + em

    ll = logsumexp(logF[-1])
    logB = np.full((L, H), LOG_ZERO, dtype=np.float64)
    logB[-1] = 0.0
    for ll_i in range(L - 2, -1, -1):
        nxt_em = np.zeros(H, dtype=np.float64)
        for h in range(H):
            nxt_em[h] = _expanded_state_block_emission_log(
                core,
                int(expanded.expanded_to_base[h]),
                obs_additional[ll_i + 1],
                expanded.trunk_sequences[h][ll_i + 1],
            )
        nxt = logB[ll_i + 1] + nxt_em
        log_no_reco_step, log_reco_step = _scaled_transition_logs_expanded(
            expanded,
            int(step_sizes[ll_i]),
        )
        logR = logsumexp(log_reco_step + nxt[None, :], axis=1)
        log_nr = log_no_reco_step + nxt
        logB[ll_i] = np.logaddexp(log_nr, logR)

    post = np.exp(logF + logB - ll)
    initial_expect = np.zeros(core.n_states, dtype=np.float64)
    no_reco_expect = np.zeros(core.n_states, dtype=np.float64)
    reco_expect = np.zeros((core.n_states, core.n_states), dtype=np.float64)
    emission_expect = np.zeros((core.n_states, n_alleles, n_alleles), dtype=np.float64)

    for h in range(H):
        initial_expect[expanded.expanded_to_base[h]] += post[0, h]
    for ll_i in range(L):
        add_block = np.atleast_1d(obs_additional[ll_i])
        for h in range(H):
            base = expanded.expanded_to_base[h]
            trunk_block = expanded.trunk_sequences[h][ll_i]
            for a_o, a_t in zip(add_block, np.atleast_1d(trunk_block)):
                if a_t >= 0 and a_o >= 0:
                    emission_expect[base, int(a_t), int(a_o)] += post[ll_i, h]

    for ll_i in range(L - 1):
        nxt_em = np.zeros(H, dtype=np.float64)
        for h in range(H):
            nxt_em[h] = _expanded_state_block_emission_log(
                core,
                int(expanded.expanded_to_base[h]),
                obs_additional[ll_i + 1],
                expanded.trunk_sequences[h][ll_i + 1],
            )
        log_no_reco_step, log_reco_step = _scaled_transition_logs_expanded(
            expanded,
            int(step_sizes[ll_i]),
        )
        xi_no = np.exp(logF[ll_i] + log_no_reco_step + nxt_em + logB[ll_i + 1] - ll)
        for h in range(H):
            no_reco_expect[expanded.expanded_to_base[h]] += xi_no[h]
        xi_re = np.exp(
            logF[ll_i][:, None]
            + log_reco_step
            + nxt_em[None, :]
            + logB[ll_i + 1][None, :]
            - ll
        )
        for src in range(H):
            src_base = expanded.expanded_to_base[src]
            for dst in range(H):
                dst_base = expanded.expanded_to_base[dst]
                reco_expect[src_base, dst_base] += xi_re[src, dst]

    return ExpectedCounts(
        log_likelihood=float(ll),
        initial_expect=initial_expect,
        no_reco_expect=no_reco_expect,
        reco_expect=reco_expect,
        emission_expect=emission_expect,
    )


# ===========================================================================
# Composite likelihood: pairwise & leave-one-out
# ===========================================================================


def _trunk_consensus(
    sequences: np.ndarray,
    trunk_haps: list[int],
) -> np.ndarray:
    """Majority-vote consensus across trunk haplotypes.

    For pairwise CSDs, this is just the single trunk haplotype's sequence.
    """
    if len(trunk_haps) == 1:
        return sequences[trunk_haps[0]].astype(np.int64)
    sub = sequences[trunk_haps]  # (n_trunk, L)
    # Per-column mode (works for small allele counts)
    consensus = np.empty(sub.shape[1], dtype=np.int64)
    for ll in range(sub.shape[1]):
        present = sub[:, ll]
        present = present[present >= 0]
        if len(present) == 0:
            consensus[ll] = -1
        else:
            consensus[ll] = int(np.bincount(present).argmax())
    return consensus


def _group_observations(
    sequence: np.ndarray,
    loci_per_hmm_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    if loci_per_hmm_step <= 1:
        return sequence.astype(np.int64), np.ones(len(sequence), dtype=np.int64)
    blocks = []
    step_sizes = []
    for start in range(0, len(sequence), loci_per_hmm_step):
        block = sequence[start : start + loci_per_hmm_step].astype(np.int64)
        step_len = len(block)
        if len(block) < loci_per_hmm_step:
            padded = np.full(loci_per_hmm_step, -1, dtype=np.int64)
            padded[: len(block)] = block
            block = padded
        blocks.append(block)
        step_sizes.append(step_len)
    return np.array(blocks, dtype=np.int64), np.array(step_sizes, dtype=np.int64)


def _enumerate_csd_pairs(n_hap: int, mode: str) -> list[tuple[int, list[int]]]:
    """Return list of (additional_idx, trunk_indices) for one CSD pass.

    Parameters
    ----------
    n_hap : int
        Number of haplotypes.
    mode : {'pcl', 'lol', 'pac'}
        Composite-likelihood scheme.
    """
    if mode == "pcl":
        # All ordered pairs
        return [(i, [j]) for i in range(n_hap) for j in range(n_hap) if j != i]
    if mode == "lol":
        # Leave-one-out
        return [(i, [j for j in range(n_hap) if j != i]) for i in range(n_hap)]
    if mode == "pac":
        # PAC: single random permutation, each haplotype conditional on
        # the previous ones (skip the first which has no trunk)
        order = list(range(n_hap))
        return [(order[i], order[:i]) for i in range(1, n_hap)]
    raise ValueError(f"Unknown composite likelihood mode: {mode}")


# ===========================================================================
# Demographic model parameterization for the M-step
# ===========================================================================


@dataclass
class DemoParameters:
    """Free demographic parameters being optimized."""

    ordered_param_ids: list[int]
    free_boundary_groups: list[list[int]]
    boundary_values: np.ndarray
    free_pop_size_groups: list[list[tuple[int, int]]]
    pop_size_values: np.ndarray
    free_migration_groups: list[list[tuple[int, int, int]]]
    migration_values: np.ndarray
    free_growth_rate_groups: list[list[tuple[int, int]]]
    growth_rate_values: np.ndarray
    boundary_param_ids: list[int]
    pop_param_ids: list[int]
    migration_param_ids: list[int]
    growth_param_ids: list[int]
    ordered_lower_bounds: np.ndarray | None = None
    ordered_upper_bounds: np.ndarray | None = None

    def n_params(self) -> int:
        return (
            len(self.boundary_values)
            + len(self.pop_size_values)
            + len(self.migration_values)
            + len(self.growth_rate_values)
        )

    def ordered_param_values(self) -> np.ndarray:
        values: list[float] = []
        bound_map = {param_id: idx for idx, param_id in enumerate(self.boundary_param_ids)}
        pop_map = {param_id: idx for idx, param_id in enumerate(self.pop_param_ids)}
        mig_map = {param_id: idx for idx, param_id in enumerate(self.migration_param_ids)}
        growth_map = {param_id: idx for idx, param_id in enumerate(self.growth_param_ids)}
        for param_id in self.ordered_param_ids:
            if param_id in bound_map:
                values.append(float(self.boundary_values[bound_map[param_id]]))
            elif param_id in pop_map:
                values.append(float(self.pop_size_values[pop_map[param_id]]))
            elif param_id in mig_map:
                values.append(float(self.migration_values[mig_map[param_id]]))
            elif param_id in growth_map:
                values.append(float(self.growth_rate_values[growth_map[param_id]]))
        return np.array(values, dtype=np.float64)

    def set_ordered_param_values(self, values: np.ndarray) -> None:
        if len(values) != len(self.ordered_param_ids):
            raise ValueError(
                f"Expected {len(self.ordered_param_ids)} parameters, got {len(values)}"
            )
        bound_map = {param_id: idx for idx, param_id in enumerate(self.boundary_param_ids)}
        pop_map = {param_id: idx for idx, param_id in enumerate(self.pop_param_ids)}
        mig_map = {param_id: idx for idx, param_id in enumerate(self.migration_param_ids)}
        growth_map = {param_id: idx for idx, param_id in enumerate(self.growth_param_ids)}
        for param_id, value in zip(self.ordered_param_ids, values):
            if param_id in bound_map:
                self.boundary_values[bound_map[param_id]] = float(value)
            elif param_id in pop_map:
                self.pop_size_values[pop_map[param_id]] = float(value)
            elif param_id in mig_map:
                self.migration_values[mig_map[param_id]] = float(value)
            elif param_id in growth_map:
                self.growth_rate_values[growth_map[param_id]] = float(value)

    def clip_to_bounds(self) -> None:
        if self.ordered_lower_bounds is None or self.ordered_upper_bounds is None:
            return
        values = self.ordered_param_values()
        values = np.clip(values, self.ordered_lower_bounds, self.ordered_upper_bounds)
        self.set_ordered_param_values(values)

    def pack_opt_params(self) -> np.ndarray:
        parts = []
        if len(self.boundary_values):
            parts.append(np.log(np.maximum(self.boundary_values, 1e-6)))
        if len(self.pop_size_values):
            parts.append(np.log(np.maximum(self.pop_size_values, 1e-3)))
        if len(self.migration_values):
            parts.append(np.log(np.maximum(self.migration_values, 1e-8)))
        if len(self.growth_rate_values):
            parts.append(self.growth_rate_values.copy())
        if not parts:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(parts)

    def unpack_opt_params(self, opt_params: np.ndarray) -> None:
        n_bound = len(self.boundary_values)
        n_pop = len(self.pop_size_values)
        n_mig = len(self.migration_values)
        if n_bound:
            self.boundary_values = np.clip(np.exp(opt_params[:n_bound]), 1e-6, 1e3)
        if n_pop:
            start = n_bound
            self.pop_size_values = np.clip(
                np.exp(opt_params[start : start + n_pop]),
                1e-3,
                1e3,
            )
        if n_mig:
            start = n_bound + n_pop
            self.migration_values = np.clip(
                np.exp(opt_params[start : start + n_mig]),
                1e-8,
                1e6,
            )
        if len(self.growth_rate_values):
            self.growth_rate_values = opt_params[n_bound + n_pop + n_mig :].copy()

    def to_demo(self, demo_template: DiCal2Demo) -> DiCal2Demo:
        """Build a fresh DiCal2Demo from the current parameter values."""
        epoch_boundaries = demo_template.epoch_boundaries.copy()
        for group, val in zip(self.free_boundary_groups, self.boundary_values):
            for boundary_idx in group:
                epoch_boundaries[boundary_idx] = val

        new_epochs = []
        for e_idx, ep in enumerate(demo_template.epochs):
            new_pop = ep.pop_sizes.copy() if ep.pop_sizes is not None else None
            new_mig = (
                ep.migration_matrix.copy()
                if ep.migration_matrix is not None
                else None
            )
            new_pulse = (
                ep.pulse_migration.copy()
                if ep.pulse_migration is not None
                else None
            )
            new_epochs.append(
                DiCal2Epoch(
                    start=float(epoch_boundaries[e_idx]),
                    end=float(epoch_boundaries[e_idx + 1]),
                    partition=[list(g) for g in ep.partition],
                    pop_sizes=new_pop,
                    pop_size_param_ids=(
                        None if ep.pop_size_param_ids is None else list(ep.pop_size_param_ids)
                    ),
                    migration_matrix=new_mig,
                    migration_param_ids=(
                        None
                        if ep.migration_param_ids is None
                        else [list(row) for row in ep.migration_param_ids]
                    ),
                    pulse_migration=new_pulse,
                    growth_rates=(
                        ep.growth_rates.copy() if ep.growth_rates is not None else None
                    ),
                    growth_rate_param_ids=(
                        None
                        if ep.growth_rate_param_ids is None
                        else list(ep.growth_rate_param_ids)
                    ),
                )
            )
        for group, val in zip(self.free_pop_size_groups, self.pop_size_values):
            for e_idx, a_idx in group:
                if new_epochs[e_idx].pop_sizes is not None:
                    new_epochs[e_idx].pop_sizes[a_idx] = val
        for group, val in zip(self.free_migration_groups, self.migration_values):
            affected_epochs: set[int] = set()
            for e_idx, src_idx, dst_idx in group:
                if new_epochs[e_idx].migration_matrix is not None:
                    new_epochs[e_idx].migration_matrix[src_idx, dst_idx] = val
                    affected_epochs.add(e_idx)
            for e_idx in affected_epochs:
                mig = new_epochs[e_idx].migration_matrix
                if mig is None:
                    continue
                for row_idx in range(mig.shape[0]):
                    off_diag = mig[row_idx, :row_idx].sum() + mig[row_idx, row_idx + 1 :].sum()
                    mig[row_idx, row_idx] = -off_diag
        for group, val in zip(self.free_growth_rate_groups, self.growth_rate_values):
            for e_idx, a_idx in group:
                if new_epochs[e_idx].growth_rates is None:
                    new_epochs[e_idx].growth_rates = np.zeros(
                        len(new_epochs[e_idx].partition),
                        dtype=np.float64,
                    )
                new_epochs[e_idx].growth_rates[a_idx] = val
        return DiCal2Demo(
            epoch_boundaries=epoch_boundaries,
            boundary_param_ids=(
                None
                if demo_template.boundary_param_ids is None
                else list(demo_template.boundary_param_ids)
            ),
            epochs=new_epochs,
            n_present_demes=demo_template.n_present_demes,
        )


def _ordered_values_within_bounds(
    values: np.ndarray,
    params: DemoParameters,
) -> bool:
    if params.ordered_lower_bounds is None or params.ordered_upper_bounds is None:
        return True
    return bool(
        np.all(values >= params.ordered_lower_bounds)
        and np.all(values <= params.ordered_upper_bounds)
    )


def _demo_is_valid(demo: DiCal2Demo) -> bool:
    boundaries = np.asarray(demo.epoch_boundaries, dtype=np.float64)
    if len(boundaries) < 2:
        return False
    if not np.isclose(boundaries[0], 0.0):
        return False
    if not np.isinf(boundaries[-1]):
        return False
    if len(demo.epochs) != len(boundaries) - 1:
        return False

    for idx, epoch in enumerate(demo.epochs):
        start = float(boundaries[idx])
        end = float(boundaries[idx + 1])
        is_pulse = epoch.migration_matrix is None
        if is_pulse:
            if not np.isclose(end, start):
                return False
        else:
            if end <= start + EPS:
                return False
        if epoch.pop_sizes is not None:
            pop_sizes = np.asarray(epoch.pop_sizes, dtype=np.float64)
            if np.any(~np.isfinite(pop_sizes)) or np.any(pop_sizes <= 0.0):
                return False
        if epoch.growth_rates is not None:
            growth_rates = np.asarray(epoch.growth_rates, dtype=np.float64)
            if np.any(~np.isfinite(growth_rates)):
                return False
            if epoch.pop_sizes is not None and len(growth_rates) != len(epoch.pop_sizes):
                return False
        if epoch.migration_matrix is not None:
            mig = np.asarray(epoch.migration_matrix, dtype=np.float64)
            if mig.ndim != 2 or mig.shape[0] != mig.shape[1]:
                return False
            off_diag = mig.copy()
            np.fill_diagonal(off_diag, 0.0)
            if np.any(~np.isfinite(off_diag)) or np.any(off_diag < 0.0):
                return False
    return True


def _demo_from_params_or_none(
    params: DemoParameters,
    demo_template: DiCal2Demo,
) -> DiCal2Demo | None:
    ordered = params.ordered_param_values()
    if not _ordered_values_within_bounds(ordered, params):
        return None
    demo = params.to_demo(demo_template)
    if not _demo_is_valid(demo):
        return None
    return demo


def _default_demo_for_single_pop(
    n_intervals: int,
    boundaries: np.ndarray,
) -> DiCal2Demo:
    """Build a default piecewise-constant single-population demography."""
    epochs = []
    for i in range(len(boundaries) - 1):
        epochs.append(
            DiCal2Epoch(
                start=float(boundaries[i]),
                end=float(boundaries[i + 1]),
                partition=[[0]],
                pop_sizes=np.array([1.0], dtype=np.float64),
                pop_size_param_ids=[i],
                migration_matrix=None,
                pulse_migration=None,
                growth_rates=np.array([0.0], dtype=np.float64),
                growth_rate_param_ids=[None],
            )
        )
    return DiCal2Demo(
        epoch_boundaries=boundaries.copy(),
        boundary_param_ids=[None] * len(boundaries),
        epochs=epochs,
        n_present_demes=1,
    )


def _build_free_params(demo: DiCal2Demo) -> DemoParameters:
    """Collect free demographic parameters from a diCal2 demo."""
    boundary_groups_by_id: dict[int, list[int]] = {}
    boundary_values_by_id: dict[int, float] = {}
    pop_groups_by_id: dict[int, list[tuple[int, int]]] = {}
    pop_values_by_id: dict[int, float] = {}
    migration_groups_by_id: dict[int, list[tuple[int, int, int]]] = {}
    migration_values_by_id: dict[int, float] = {}
    growth_groups_by_id: dict[int, list[tuple[int, int]]] = {}
    growth_values_by_id: dict[int, float] = {}
    saw_explicit_ids = False

    if demo.boundary_param_ids is not None:
        for boundary_idx, param_id in enumerate(demo.boundary_param_ids):
            if boundary_idx == 0 or boundary_idx == len(demo.epoch_boundaries) - 1:
                continue
            if param_id is None:
                continue
            saw_explicit_ids = True
            boundary_groups_by_id.setdefault(param_id, []).append(boundary_idx)
            boundary_values_by_id.setdefault(
                param_id,
                float(demo.epoch_boundaries[boundary_idx]),
            )

    for e_idx, ep in enumerate(demo.epochs):
        if ep.pop_sizes is not None:
            if ep.pop_size_param_ids is not None:
                for a_idx, param_id in enumerate(ep.pop_size_param_ids):
                    if param_id is None:
                        continue
                    saw_explicit_ids = True
                    pop_groups_by_id.setdefault(param_id, []).append((e_idx, a_idx))
                    pop_values_by_id.setdefault(param_id, float(ep.pop_sizes[a_idx]))
            elif not saw_explicit_ids:
                pop_groups_by_id[e_idx] = [(e_idx, a_idx) for a_idx in range(len(ep.pop_sizes))]
                pop_values_by_id[e_idx] = float(ep.pop_sizes[0])
        if ep.migration_matrix is not None and ep.migration_param_ids is not None:
            for src_idx, row in enumerate(ep.migration_param_ids):
                for dst_idx, param_id in enumerate(row):
                    if src_idx == dst_idx or param_id is None:
                        continue
                    saw_explicit_ids = True
                    migration_groups_by_id.setdefault(param_id, []).append((e_idx, src_idx, dst_idx))
                    migration_values_by_id.setdefault(
                        param_id,
                        float(ep.migration_matrix[src_idx, dst_idx]),
                    )
        if ep.growth_rates is not None and ep.growth_rate_param_ids is not None:
            for a_idx, param_id in enumerate(ep.growth_rate_param_ids):
                if param_id is None:
                    continue
                saw_explicit_ids = True
                growth_groups_by_id.setdefault(param_id, []).append((e_idx, a_idx))
                growth_values_by_id.setdefault(param_id, float(ep.growth_rates[a_idx]))

    if not saw_explicit_ids:
        next_id = 0
        for boundary_idx in range(1, len(demo.epoch_boundaries) - 1):
            boundary_groups_by_id[next_id] = [boundary_idx]
            boundary_values_by_id[next_id] = float(demo.epoch_boundaries[boundary_idx])
            next_id += 1
        pop_groups_by_id = {}
        pop_values_by_id = {}
        for e_idx, ep in enumerate(demo.epochs):
            if ep.pop_sizes is None:
                continue
            for a_idx in range(len(ep.pop_sizes)):
                pop_groups_by_id[next_id] = [(e_idx, a_idx)]
                pop_values_by_id[next_id] = float(ep.pop_sizes[a_idx])
                next_id += 1

    boundary_ids = sorted(boundary_groups_by_id)
    pop_ids = sorted(pop_groups_by_id)
    migration_ids = sorted(migration_groups_by_id)
    growth_ids = sorted(growth_groups_by_id)
    ordered_ids = sorted(set(boundary_ids) | set(pop_ids) | set(migration_ids) | set(growth_ids))
    return DemoParameters(
        ordered_param_ids=ordered_ids,
        free_boundary_groups=[boundary_groups_by_id[param_id] for param_id in boundary_ids],
        boundary_values=np.array(
            [boundary_values_by_id[param_id] for param_id in boundary_ids],
            dtype=np.float64,
        ),
        free_pop_size_groups=[pop_groups_by_id[param_id] for param_id in pop_ids],
        pop_size_values=np.array([pop_values_by_id[param_id] for param_id in pop_ids], dtype=np.float64),
        free_migration_groups=[
            migration_groups_by_id[param_id] for param_id in migration_ids
        ],
        migration_values=np.array(
            [migration_values_by_id[param_id] for param_id in migration_ids],
            dtype=np.float64,
        ),
        free_growth_rate_groups=[
            growth_groups_by_id[param_id] for param_id in growth_ids
        ],
        growth_rate_values=np.array(
            [growth_values_by_id[param_id] for param_id in growth_ids],
            dtype=np.float64,
        ),
        boundary_param_ids=boundary_ids,
        pop_param_ids=pop_ids,
        migration_param_ids=migration_ids,
        growth_param_ids=growth_ids,
    )


# ===========================================================================
# Q-function evaluation for the M-step
# ===========================================================================


def _q_function(
    params: DemoParameters,
    demo_template: DiCal2Demo,
    interval_boundaries: np.ndarray,
    counts_list: list[ExpectedCounts],
    csd_setups: list[tuple[SimpleTrunk, int]],
    mutation_matrix: np.ndarray,
    theta: float,
    rho: float,
) -> float:
    """Negative Q-function (for minimization).

    Builds new HMM matrices from the candidate parameters and computes

        Q = sum [E[init] @ log π_new
              + E[noReco] @ log_noReco_new
              + sum E[reco] * log_reco_new
              + sum E[emit] * log_emission_new]

    summed across all CSD configurations.
    """
    new_demo = _demo_from_params_or_none(params, demo_template)
    if new_demo is None:
        return float(np.inf)
    refined = refine_demography(new_demo, interval_boundaries)

    total_q = 0.0
    for (trunk, observed_present), counts in zip(csd_setups, counts_list):
        core_obj, _ = _build_native_core(
            refined=refined,
            trunk=trunk,
            observed_present_deme=observed_present,
            mutation_matrix=mutation_matrix,
            theta=theta,
            rho=rho,
        )
        core = core_obj.core_matrices()
        # The expected counts may have a different state space if the
        # refined grid changed: assume same here (boundaries are fixed).
        if core.n_states != counts.n_states if hasattr(counts, "n_states") else False:
            continue

        q_init = float(np.sum(counts.initial_expect * core.log_initial))
        q_no = float(np.sum(counts.no_reco_expect * core.log_no_reco))
        q_re = float(np.sum(counts.reco_expect * core.log_reco))
        q_em = float(np.sum(counts.emission_expect * core.log_emission))
        total_q += q_init + q_no + q_re + q_em

    return -total_q


def _parse_bounds(
    bounds: str | list[tuple[float, float]] | None,
    n_params: int,
) -> list[tuple[float, float]] | None:
    if bounds is None:
        return None
    if isinstance(bounds, str):
        pairs = []
        for item in bounds.split(";"):
            lo, hi = item.split(",")
            pairs.append((float(lo), float(hi)))
        bounds = pairs
    parsed = list(bounds)
    if len(parsed) != n_params:
        raise ValueError(f"Expected {n_params} bounds, got {len(parsed)}")
    return parsed


def _evaluate_total_log_likelihood(
    sequences: np.ndarray,
    config: DiCal2Config,
    demo: DiCal2Demo,
    interval_boundaries: np.ndarray,
    mutation_matrix: np.ndarray,
    theta: float,
    rho: float,
    composite_mode: str,
    n_alleles: int,
    loci_per_hmm_step: int,
) -> float:
    refined = refine_demography(demo, interval_boundaries)
    total_ll = 0.0
    for additional_idx, trunk_idxs in _enumerate_csd_pairs(sequences.shape[0], composite_mode):
        if not trunk_idxs:
            continue
        present_deme = config.haplotype_populations[additional_idx]
        trunk = SimpleTrunk(
            config=config,
            additional_hap_idx=additional_idx,
            trunk_hap_indices=trunk_idxs,
        )
        core_obj, _ = _build_native_core(
            refined=refined,
            trunk=trunk,
            observed_present_deme=present_deme,
            mutation_matrix=mutation_matrix,
            theta=theta,
            rho=rho,
        )
        core = core_obj.core_matrices()
        obs_add, step_sizes = _group_observations(
            sequences[additional_idx],
            loci_per_hmm_step,
        )
        grouped_trunks = {
            h: _group_observations(sequences[h], loci_per_hmm_step)[0]
            for h in trunk_idxs
        }
        expanded = _build_expanded_core(core_obj, core, trunk, trunk_idxs, grouped_trunks)
        total_ll += _expanded_expected_counts(
            core,
            expanded,
            obs_add,
            n_alleles,
            step_sizes=step_sizes,
        ).log_likelihood
    return float(total_ll)


def _clone_demo_parameters(params: DemoParameters) -> DemoParameters:
    return DemoParameters(
        ordered_param_ids=list(params.ordered_param_ids),
        free_boundary_groups=[list(group) for group in params.free_boundary_groups],
        boundary_values=params.boundary_values.copy(),
        free_pop_size_groups=[list(group) for group in params.free_pop_size_groups],
        pop_size_values=params.pop_size_values.copy(),
        free_migration_groups=[list(group) for group in params.free_migration_groups],
        migration_values=params.migration_values.copy(),
        free_growth_rate_groups=[list(group) for group in params.free_growth_rate_groups],
        growth_rate_values=params.growth_rate_values.copy(),
        boundary_param_ids=list(params.boundary_param_ids),
        pop_param_ids=list(params.pop_param_ids),
        migration_param_ids=list(params.migration_param_ids),
        growth_param_ids=list(params.growth_param_ids),
        ordered_lower_bounds=(
            None if params.ordered_lower_bounds is None else params.ordered_lower_bounds.copy()
        ),
        ordered_upper_bounds=(
            None if params.ordered_upper_bounds is None else params.ordered_upper_bounds.copy()
        ),
    )


def _transformed_bounds_for_optimizer(
    params: DemoParameters,
    parsed_bounds: list[tuple[float, float]] | None,
) -> list[tuple[float, float]] | None:
    if parsed_bounds is None:
        return None
    transformed_bounds = []
    bound_map = {
        param_id: (lo, hi)
        for param_id, (lo, hi) in zip(params.ordered_param_ids, parsed_bounds)
    }
    for param_id in params.boundary_param_ids:
        lo, hi = bound_map[param_id]
        transformed_bounds.append((np.log(max(lo, 1e-6)), np.log(max(hi, 1e-6))))
    for param_id in params.pop_param_ids:
        lo, hi = bound_map[param_id]
        transformed_bounds.append((np.log(max(lo, 1e-6)), np.log(max(hi, 1e-6))))
    for param_id in params.migration_param_ids:
        lo, hi = bound_map[param_id]
        transformed_bounds.append((np.log(max(lo, 1e-8)), np.log(max(hi, 1e-8))))
    for param_id in params.growth_param_ids:
        lo, hi = bound_map[param_id]
        transformed_bounds.append((lo, hi))
    return transformed_bounds


def _q_function_ordered(
    ordered_params: np.ndarray,
    params_template: DemoParameters,
    demo_template: DiCal2Demo,
    interval_boundaries: np.ndarray,
    counts_list: list[ExpectedCounts],
    csd_setups: list[tuple[SimpleTrunk, int]],
    mutation_matrix: np.ndarray,
    theta: float,
    rho: float,
) -> float:
    candidate = _clone_demo_parameters(params_template)
    ordered_params = np.asarray(ordered_params, dtype=np.float64)
    if not _ordered_values_within_bounds(ordered_params, candidate):
        return float(np.inf)
    candidate.set_ordered_param_values(ordered_params)
    return _q_function(
        candidate,
        demo_template,
        interval_boundaries,
        counts_list,
        csd_setups,
        mutation_matrix,
        theta,
        rho,
    )


def _relative_error(value_one: float, value_two: float, epsilon: float = EPS) -> float:
    rel_err = abs(value_one - value_two)
    if rel_err < epsilon and abs(value_two) < epsilon:
        return 0.0
    return rel_err / abs(value_two)


def _max_relative_error(
    vector_one: np.ndarray,
    vector_two: np.ndarray,
    epsilon: float = EPS,
) -> float:
    return max(
        _relative_error(float(v1), float(v2), epsilon)
        for v1, v2 in zip(vector_one, vector_two)
    )


def _reflect_at_bounds(
    value: float,
    bounds: tuple[float, float] | None,
) -> float:
    if bounds is None:
        return value
    lower, upper = bounds
    reflected = value
    if reflected < lower:
        reflected = lower + (lower - reflected)
    elif reflected > upper:
        reflected = upper - (reflected - upper)
    if reflected < lower or reflected > upper:
        reflected = 0.5 * (lower + upper)
    return reflected


def _ordered_bounds(
    params: DemoParameters,
) -> list[tuple[float, float]] | None:
    if params.ordered_lower_bounds is None or params.ordered_upper_bounds is None:
        return None
    return [
        (float(lo), float(hi))
        for lo, hi in zip(params.ordered_lower_bounds, params.ordered_upper_bounds)
    ]


def _initial_step_sizes(
    curr_point: np.ndarray,
    nm_fraction: float | None,
    rng: np.random.Generator,
) -> np.ndarray | None:
    if nm_fraction is None:
        return None
    step_sizes = np.empty_like(curr_point, dtype=np.float64)
    for idx, value in enumerate(curr_point):
        signed_step = float(nm_fraction) * float(value)
        if rng.random() >= 0.5:
            signed_step *= -1.0
        step_sizes[idx] = max(1e-8, signed_step)
    return step_sizes


def _initial_simplex(
    curr_point: np.ndarray,
    step_sizes: np.ndarray | None,
) -> np.ndarray | None:
    if step_sizes is None:
        return None
    simplex = [curr_point.astype(np.float64, copy=True)]
    for idx, step in enumerate(step_sizes):
        vertex = curr_point.astype(np.float64, copy=True)
        vertex[idx] = vertex[idx] + step
        simplex.append(vertex)
    return np.array(simplex, dtype=np.float64)


def _update_point(
    curr_point: np.ndarray,
    objective_function,
    number_iterations_mstep: int | None,
    relative_error_m: float | None,
    nm_fraction: float | None,
    use_param_rel_err_m: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    initial_simplex = _initial_simplex(
        curr_point,
        _initial_step_sizes(curr_point, nm_fraction, rng),
    )

    options: dict[str, object] = {"disp": False}
    if initial_simplex is not None:
        options["initial_simplex"] = initial_simplex
    if number_iterations_mstep is not None:
        options["maxiter"] = int(number_iterations_mstep)
    elif relative_error_m is not None:
        if use_param_rel_err_m:
            options["xatol"] = float(relative_error_m)
            options["fatol"] = 0.0
        else:
            options["xatol"] = 0.0
            options["fatol"] = float(relative_error_m)
    result = minimize(
        objective_function,
        np.asarray(curr_point, dtype=np.float64),
        method="Nelder-Mead",
        options=options,
    )
    return np.asarray(result.x, dtype=np.float64)


def _update_point_coordinatewise(
    curr_point: np.ndarray,
    objective_function,
    number_iterations_mstep: int | None,
    relative_error_m: float | None,
    nm_fraction: float | None,
    use_param_rel_err_m: bool,
    coordinate_order: tuple[int, ...] | None,
    rng: np.random.Generator,
) -> np.ndarray:
    local_curr_point = np.asarray(curr_point, dtype=np.float64).copy()
    if coordinate_order is None:
        permutation = list(rng.permutation(len(local_curr_point)))
    else:
        permutation = list(coordinate_order)
    for idx in permutation:
        def clamped_objective(arg: np.ndarray) -> float:
            point = local_curr_point.copy()
            point[idx] = float(arg[0])
            return float(objective_function(point))

        local_curr_point[idx] = _update_point(
            np.array([local_curr_point[idx]], dtype=np.float64),
            clamped_objective,
            number_iterations_mstep=number_iterations_mstep,
            relative_error_m=relative_error_m,
            nm_fraction=nm_fraction,
            use_param_rel_err_m=use_param_rel_err_m,
            rng=rng,
        )[0]
    return local_curr_point


def _marginal_standard_deviation(points: list[np.ndarray]) -> np.ndarray:
    if len(points) == 1:
        return np.zeros_like(points[0], dtype=np.float64)
    stacked = np.vstack(points)
    return np.sqrt(np.mean((stacked - stacked.mean(axis=0)) ** 2, axis=0))


def _meta_get_new_point(
    parent: np.ndarray,
    sd: np.ndarray,
    bounds: list[tuple[float, float]] | None,
    stretch_proportion: float,
    disperse_factor: float,
    rng: np.random.Generator,
) -> np.ndarray:
    new_point = np.empty_like(parent, dtype=np.float64)
    if rng.random() > stretch_proportion:
        for idx in range(len(new_point)):
            gaussian = (
                np.sqrt(max(abs(parent[idx]), EPS) * max(sd[idx], EPS))
                * disperse_factor
                * rng.normal()
            )
            point_bounds = None if bounds is None else bounds[idx]
            new_point[idx] = _reflect_at_bounds(parent[idx] + gaussian, point_bounds)
        return new_point

    avg_sd = float(np.exp(np.mean(np.log(np.maximum(sd, EPS)))))
    stretch_factor = _reflect_at_bounds(
        avg_sd * disperse_factor * rng.normal() * 5.0 + 1.0,
        (0.0, np.inf),
    )
    for idx in range(len(new_point)):
        point_bounds = None if bounds is None else bounds[idx]
        new_point[idx] = _reflect_at_bounds(parent[idx] * stretch_factor, point_bounds)
    return new_point


def _meta_next_generation(
    generation_results: list[tuple[float, np.ndarray]],
    params: DemoParameters,
    *,
    meta_keep_best: int,
    meta_num_points: int,
    stretch_proportion: float,
    disperse_factor: float,
    sd_percentage_if_zero: float,
    rng: np.random.Generator,
) -> np.ndarray:
    sorted_results = sorted(generation_results, key=lambda result: result[0], reverse=True)
    kept_points = [
        np.asarray(point, dtype=np.float64).copy()
        for _, point in sorted_results[: max(meta_keep_best, 1)]
    ]
    sd = _marginal_standard_deviation(kept_points)
    best_point = kept_points[0]
    for idx in range(len(sd)):
        if sd[idx] < EPS:
            sd[idx] = sd_percentage_if_zero * max(abs(best_point[idx]), 1.0)
    next_generation = [point.copy() for point in kept_points]
    bounds = _ordered_bounds(params)
    while len(next_generation) < meta_num_points:
        parent = kept_points[int(rng.random() * len(kept_points))]
        new_point = _meta_get_new_point(
            parent,
            sd,
            bounds,
            stretch_proportion=stretch_proportion,
            disperse_factor=disperse_factor,
            rng=rng,
        )
        candidate = _clone_demo_parameters(params)
        candidate.set_ordered_param_values(new_point)
        candidate.clip_to_bounds()
        next_generation.append(candidate.ordered_param_values())
    return np.array(next_generation, dtype=np.float64)


def _run_dical2_em(
    *,
    demo: DiCal2Demo,
    params: DemoParameters,
    sequences: np.ndarray,
    config: DiCal2Config,
    interval_boundaries: np.ndarray,
    mutation_matrix: np.ndarray,
    theta: float,
    rho: float,
    composite_mode: str,
    n_alleles: int,
    loci_per_hmm_step: int,
    number_iterations_em: int,
    number_iterations_mstep: int | None,
    relative_error_e: float | None,
    relative_error_m: float | None,
    use_param_rel_err: bool,
    use_param_rel_err_m: bool,
    coordinatewise_mstep: bool,
    coordinate_order: tuple[int, ...] | None,
    nm_fraction: float | None,
    rng: np.random.Generator,
) -> tuple[DemoParameters, list[dict], float, str]:
    n_hap = sequences.shape[0]
    csd_pairs = _enumerate_csd_pairs(n_hap, composite_mode)
    rounds: list[dict] = []
    prev_ll = -np.inf
    prev_point: np.ndarray | None = None
    selected_core_type = "eigen"

    for em_iter in range(number_iterations_em + 1):
        refined = refine_demography(params.to_demo(demo), interval_boundaries)

        counts_list: list[ExpectedCounts] = []
        csd_setups: list[tuple[SimpleTrunk, int]] = []
        total_ll = 0.0

        for additional_idx, trunk_idxs in csd_pairs:
            if not trunk_idxs:
                continue
            present_deme = config.haplotype_populations[additional_idx]
            trunk = SimpleTrunk(
                config=config,
                additional_hap_idx=additional_idx,
                trunk_hap_indices=trunk_idxs,
            )
            core_obj, selected_core_type = _build_native_core(
                refined=refined,
                trunk=trunk,
                observed_present_deme=present_deme,
                mutation_matrix=mutation_matrix,
                theta=theta,
                rho=rho,
            )
            core = core_obj.core_matrices()
            obs_add, step_sizes = _group_observations(
                sequences[additional_idx],
                loci_per_hmm_step,
            )
            grouped_trunks = {
                h: _group_observations(sequences[h], loci_per_hmm_step)[0]
                for h in trunk_idxs
            }
            expanded = _build_expanded_core(
                core_obj,
                core,
                trunk,
                trunk_idxs,
                grouped_trunks,
            )

            counts = _expanded_expected_counts(
                core,
                expanded,
                obs_add,
                n_alleles,
                step_sizes=step_sizes,
            )
            counts.n_states = core.n_states  # type: ignore[attr-defined]
            counts_list.append(counts)
            csd_setups.append((trunk, present_deme))
            total_ll += counts.log_likelihood

        rounds.append(
            {
                "iteration": em_iter,
                "log_likelihood": total_ll,
                "epoch_boundaries": params.boundary_values.copy(),
                "pop_sizes": params.pop_size_values.copy(),
                "growth_rates": params.growth_rate_values.copy(),
                "ordered_params": params.ordered_param_values().copy(),
            }
        )

        curr_point = params.ordered_param_values()
        stop = not np.isfinite(total_ll) or em_iter >= number_iterations_em
        if not stop and relative_error_e is not None and prev_point is not None:
            rel_err = (
                _max_relative_error(curr_point, prev_point)
                if use_param_rel_err
                else _relative_error(total_ll, prev_ll)
            )
            stop = rel_err < relative_error_e
        if stop:
            prev_ll = total_ll
            break

        objective_params = _clone_demo_parameters(params)
        objective = lambda point: _q_function_ordered(
            np.asarray(point, dtype=np.float64),
            objective_params,
            demo,
            interval_boundaries,
            counts_list,
            csd_setups,
            mutation_matrix,
            theta,
            rho,
        )
        try:
            if coordinatewise_mstep:
                next_point = _update_point_coordinatewise(
                    curr_point,
                    objective,
                    number_iterations_mstep=number_iterations_mstep,
                    relative_error_m=relative_error_m,
                    nm_fraction=nm_fraction,
                    use_param_rel_err_m=use_param_rel_err_m,
                    coordinate_order=coordinate_order,
                    rng=rng,
                )
            else:
                next_point = _update_point(
                    curr_point,
                    objective,
                    number_iterations_mstep=number_iterations_mstep,
                    relative_error_m=relative_error_m,
                    nm_fraction=nm_fraction,
                    use_param_rel_err_m=use_param_rel_err_m,
                    rng=rng,
                )
            next_params = _clone_demo_parameters(params)
            next_params.set_ordered_param_values(next_point)
            if _demo_from_params_or_none(next_params, demo) is not None:
                params = next_params
        except Exception as exc:  # pragma: no cover - safety net
            logger.warning("M-step failed at iteration %d: %s", em_iter, exc)

        prev_point = curr_point.copy()
        prev_ll = total_ll

    return params, rounds, float(prev_ll), selected_core_type


# ===========================================================================
# Main entry point: dical2()
# ===========================================================================


@dataclass
class DiCal2Result:
    """Results from running diCal2."""

    time: np.ndarray
    pop_sizes: np.ndarray  # (n_epochs, n_ancient_demes) — flat for single pop
    ne: np.ndarray  # for single pop convenience
    log_likelihood: float
    n_iterations: int
    rounds: list[dict] = field(default_factory=list)


def dical2(
    data: SmcData,
    n_intervals: int = 11,
    max_t: float = 4.0,
    alpha: float = 0.1,
    n_em_iterations: int = 10,
    composite_mode: str = "pac",
    loci_per_hmm_step: int = 1,
    mu: float = 1.25e-8,
    generation_time: float = 25.0,
    n_ref: float | None = None,
    start_point: np.ndarray | list[float] | None = None,
    meta_start_file: str | None = None,
    meta_num_iterations: int = 1,
    meta_keep_best: int = 1,
    meta_num_points: int | None = None,
    bounds: str | list[tuple[float, float]] | None = None,
    seed: int | None = None,
    implementation: str = "auto",
    upstream_options: dict | None = None,
    native_options: dict | None = None,
) -> SmcData:
    """Run diCal2 demographic inference.

    Parameters
    ----------
    data : SmcData
        Data container from ``smckit.io.read_dical2``. Must have
        ``data.sequences`` of shape ``(n_haplotypes, seq_length)`` and
        ``data.uns`` populated with ``mutation_matrix``, ``demo``,
        ``config``, ``n_alleles``.
    n_intervals : int
        Number of refined coalescent time intervals.
    max_t : float
        Maximum coalescent time (units of 2N₀).
    alpha : float
        Resolution near t=0.
    n_em_iterations : int
        Number of EM iterations.
    composite_mode : {'pcl', 'lol', 'pac'}
        Composite-likelihood scheme.
    loci_per_hmm_step : int
        Number of loci to group into one HMM step, matching diCal2's
        ``--lociPerHmmStep`` option approximately.
    mu : float
        Per-base mutation rate (for converting to absolute time).
    generation_time : float
        Years per generation.
    n_ref : float, optional
        Reference effective population size for scaling. If None, derived
        from theta and ``mu``.
    start_point : array-like, optional
        Initial values for placeholder parameters in diCal2 ``?N`` order.
    meta_start_file : str, optional
        Tab-separated file of candidate starting points in diCal2 ``?N`` order.
    meta_num_iterations : int
        Number of meta-start generations to run when ``meta_start_file`` is
        provided. ``1`` means evaluate only the supplied points.
    meta_keep_best : int
        Number of best points to retain between meta-start generations.
    meta_num_points : int, optional
        Number of points per meta-start generation. Defaults to the number of
        rows in ``meta_start_file``.
    bounds : str or list of pairs, optional
        Parameter bounds in diCal2 placeholder order, e.g. ``'0.1,1;0.2,2'``.
    seed : int, optional
        RNG seed (currently unused — kept for API symmetry).
    implementation : {"auto", "native", "upstream"}
        Algorithm provenance selector. ``"native"`` runs the in-repo diCal2
        port. ``"upstream"`` currently raises because the public upstream bridge
        is not exposed yet. ``"auto"`` resolves to the best available implementation.

    Returns
    -------
    SmcData
        Same container with results stored in ``data.results['dical2']``.
    """
    implementation = normalize_implementation(implementation)
    implementation_used = choose_implementation(
        implementation,
        upstream_available=method_upstream_available("dical2"),
    )
    source_paths = data.uns.get("source_paths", {})
    upstream_required_inputs = ["param_file", "demo_file", "config_file", "reference_file", "sequences"]
    if implementation == "auto" and implementation_used == "upstream":
        if any(not source_paths.get(key) for key in upstream_required_inputs):
            implementation_used = "native"
    allowed_method_option_keys = {
        "bounds",
        "composite_mode",
        "compositeLikelihood",
        "coordinate_order",
        "coordinateOrder",
        "coordinatewise_mstep",
        "coordinateWiseMStep",
        "disable_coordinatewise_mstep",
        "disableCoordinateWiseMStep",
        "interval_type",
        "intervalType",
        "interval_params",
        "intervalParams",
        "loci_per_hmm_step",
        "lociPerHmmStep",
        "meta_disperse_factor",
        "metaDisperseFactor",
        "meta_keep_best",
        "metaKeepBest",
        "meta_num_iterations",
        "metaNumIterations",
        "meta_num_points",
        "metaNumPoints",
        "meta_sd_percentage_if_zero",
        "metaSDPercentageIfZero",
        "meta_start_file",
        "metaStartFile",
        "meta_stretch_proportion",
        "metaStretchProportion",
        "nm_fraction",
        "nmFraction",
        "number_iterations_em",
        "numberIterationsEM",
        "number_iterations_mstep",
        "numberIterationsMstep",
        "relative_error_e",
        "relativeErrorE",
        "relative_error_m",
        "relativeErrorM",
        "seed",
        "start_point",
        "startPoint",
        "use_param_rel_err",
        "useParamRelErr",
        "use_param_rel_err_m",
        "useParamRelErrM",
    }
    native_method_options = {} if native_options is None else dict(native_options)
    if native_method_options:
        unsupported = sorted(set(native_method_options) - allowed_method_option_keys)
        if unsupported:
            unsupported_list = ", ".join(unsupported)
            raise TypeError(f"Unsupported dical2 native_options keys: {unsupported_list}")
    upstream_method_options = {} if upstream_options is None else dict(upstream_options)
    cli_args = list(upstream_method_options.pop("cli_args", []))
    if upstream_method_options:
        unsupported = sorted(set(upstream_method_options) - allowed_method_option_keys)
        if unsupported:
            unsupported_list = ", ".join(unsupported)
            raise TypeError(f"Unsupported dical2 upstream_options keys: {unsupported_list}")

    resolved_native = _resolve_dical2_options(
        n_intervals=n_intervals,
        max_t=max_t,
        alpha=alpha,
        n_em_iterations=n_em_iterations,
        composite_mode=composite_mode,
        loci_per_hmm_step=loci_per_hmm_step,
        start_point=start_point,
        meta_start_file=meta_start_file,
        meta_num_iterations=meta_num_iterations,
        meta_keep_best=meta_keep_best,
        meta_num_points=meta_num_points,
        bounds=bounds,
        seed=seed,
        method_options=native_method_options,
    )
    resolved_upstream = _resolve_dical2_options(
        n_intervals=n_intervals,
        max_t=max_t,
        alpha=alpha,
        n_em_iterations=n_em_iterations,
        composite_mode=composite_mode,
        loci_per_hmm_step=loci_per_hmm_step,
        start_point=start_point,
        meta_start_file=meta_start_file,
        meta_num_iterations=meta_num_iterations,
        meta_keep_best=meta_keep_best,
        meta_num_points=meta_num_points,
        bounds=bounds,
        seed=seed,
        method_options=upstream_method_options,
    )
    if implementation_used == "upstream":
        return _dical2_upstream(
            data,
            resolved=resolved_upstream,
            cli_args=cli_args,
            implementation_requested=implementation,
        )

    if data.sequences is None:
        raise ValueError("data.sequences is None — call read_dical2 first.")
    sequences = np.asarray(data.sequences, dtype=np.int8)
    n_hap, seq_len = sequences.shape

    n_alleles = int(data.uns.get("n_alleles", 2))
    mutation_matrix = data.uns.get("mutation_matrix")
    if mutation_matrix is None:
        mutation_matrix = np.ones((n_alleles, n_alleles), dtype=np.float64)
        np.fill_diagonal(mutation_matrix, -(n_alleles - 1))
    mutation_matrix = np.asarray(mutation_matrix, dtype=np.float64)

    config: DiCal2Config | None = data.uns.get("config")
    if config is None:
        config = DiCal2Config(
            seq_length=seq_len,
            n_alleles=n_alleles,
            n_populations=1,
            haplotype_populations=[0] * n_hap,
            haplotypes_to_include=[True] * n_hap,
            haplotype_multiplicities=np.ones((n_hap, 1), dtype=np.int64),
            sample_sizes=np.array([n_hap], dtype=np.int64),
        )

    demo: DiCal2Demo | None = data.uns.get("demo")
    interval_boundaries = _resolve_interval_boundaries(demo, config, resolved_native)
    if demo is None:
        demo = _default_demo_for_single_pop(len(interval_boundaries) - 1, interval_boundaries)

    params = _build_free_params(demo)
    if params.n_params() == 0:
        raise ValueError("No free parameters in demographic model.")
    parsed_bounds = _parse_bounds(resolved_native.bounds, len(params.ordered_param_ids))
    if parsed_bounds is not None:
        params.ordered_lower_bounds = np.array([lo for lo, _ in parsed_bounds], dtype=np.float64)
        params.ordered_upper_bounds = np.array([hi for _, hi in parsed_bounds], dtype=np.float64)
    theta = float(data.params.get("theta", 0.0005))
    rho = float(data.params.get("rho", 0.0005))
    rng = np.random.default_rng(resolved_native.seed)

    best_params: DemoParameters | None = None
    best_rounds: list[dict] = []
    best_ll = -np.inf
    best_core_type = "eigen"

    if resolved_native.meta_start_file is not None:
        candidate_points = np.loadtxt(
            resolved_native.meta_start_file,
            dtype=np.float64,
            ndmin=2,
        )
        effective_meta_num_points = resolved_native.meta_num_points
        if effective_meta_num_points is None:
            effective_meta_num_points = candidate_points.shape[0]
        candidate_points = candidate_points[:effective_meta_num_points].copy()
        for meta_iter in range(max(resolved_native.meta_num_iterations, 1)):
            generation_results: list[tuple[float, np.ndarray]] = []
            for row in candidate_points:
                local_params = _clone_demo_parameters(params)
                local_params.set_ordered_param_values(np.asarray(row, dtype=np.float64))
                if _demo_from_params_or_none(local_params, demo) is None:
                    continue
                local_params, rounds, ll, core_type = _run_dical2_em(
                    demo=demo,
                    params=local_params,
                    sequences=sequences,
                    config=config,
                    interval_boundaries=interval_boundaries,
                    mutation_matrix=mutation_matrix,
                    theta=theta,
                    rho=rho,
                    composite_mode=resolved_native.composite_mode,
                    n_alleles=n_alleles,
                    loci_per_hmm_step=resolved_native.loci_per_hmm_step,
                    number_iterations_em=resolved_native.number_iterations_em,
                    number_iterations_mstep=resolved_native.number_iterations_mstep,
                    relative_error_e=resolved_native.relative_error_e,
                    relative_error_m=resolved_native.relative_error_m,
                    use_param_rel_err=resolved_native.use_param_rel_err,
                    use_param_rel_err_m=resolved_native.use_param_rel_err_m,
                    coordinatewise_mstep=resolved_native.coordinatewise_mstep,
                    coordinate_order=resolved_native.coordinate_order,
                    nm_fraction=resolved_native.nm_fraction,
                    rng=rng,
                )
                generation_results.append((ll, local_params.ordered_param_values()))
                if ll > best_ll:
                    best_ll = ll
                    best_rounds = rounds
                    best_params = _clone_demo_parameters(local_params)
                    best_core_type = core_type
            if not generation_results:
                raise ValueError("No valid diCal2 meta-start points remained after applying bounds.")
            if meta_iter == max(resolved_native.meta_num_iterations, 1) - 1:
                break
            candidate_points = _meta_next_generation(
                generation_results,
                params,
                meta_keep_best=resolved_native.meta_keep_best,
                meta_num_points=effective_meta_num_points,
                stretch_proportion=resolved_native.meta_stretch_proportion,
                disperse_factor=resolved_native.meta_disperse_factor,
                sd_percentage_if_zero=resolved_native.meta_sd_percentage_if_zero,
                rng=rng,
            )
    else:
        if resolved_native.start_point is not None:
            params.set_ordered_param_values(np.asarray(resolved_native.start_point, dtype=np.float64))
            if _demo_from_params_or_none(params, demo) is None:
                raise ValueError("diCal2 start_point is outside bounds or yields an invalid demography.")
        params, best_rounds, best_ll, best_core_type = _run_dical2_em(
            demo=demo,
            params=params,
            sequences=sequences,
            config=config,
            interval_boundaries=interval_boundaries,
            mutation_matrix=mutation_matrix,
            theta=theta,
            rho=rho,
            composite_mode=resolved_native.composite_mode,
            n_alleles=n_alleles,
            loci_per_hmm_step=resolved_native.loci_per_hmm_step,
            number_iterations_em=resolved_native.number_iterations_em,
            number_iterations_mstep=resolved_native.number_iterations_mstep,
            relative_error_e=resolved_native.relative_error_e,
            relative_error_m=resolved_native.relative_error_m,
            use_param_rel_err=resolved_native.use_param_rel_err,
            use_param_rel_err_m=resolved_native.use_param_rel_err_m,
            coordinatewise_mstep=resolved_native.coordinatewise_mstep,
            coordinate_order=resolved_native.coordinate_order,
            nm_fraction=resolved_native.nm_fraction,
            rng=rng,
        )
        best_params = params

    assert best_params is not None
    params = best_params
    rounds = best_rounds
    prev_ll = best_ll

    # Final pop sizes per epoch (single deme assumption for the convenience field)
    final_demo = params.to_demo(demo)
    pop_sizes_arr = [ep.pop_sizes.copy() for ep in final_demo.epochs if ep.pop_sizes is not None]
    growth_rates_arr = [
        (
            ep.growth_rates.copy()
            if ep.growth_rates is not None
            else np.zeros(len(ep.partition), dtype=np.float64)
        )
        for ep in final_demo.epochs
        if ep.pop_sizes is not None
    ]
    pop_sizes_flat = np.array([sizes[0] for sizes in pop_sizes_arr], dtype=np.float64)
    growth_rates_flat = np.array([rates[0] for rates in growth_rates_arr], dtype=np.float64)

    # Convert to absolute Ne and time
    if n_ref is None:
        # theta = 4 * N_ref * mu  ⇒  N_ref = theta / (4 * mu)
        n_ref = theta / (4.0 * mu) if mu > 0 else 1e4
    ne = pop_sizes_flat * n_ref
    structured_ne = [sizes * n_ref for sizes in pop_sizes_arr]
    time = interval_boundaries[:-1] * 2.0 * n_ref  # in generations
    time_years = time * generation_time

    data.results["dical2"] = annotate_result({
        "time": time,
        "time_years": time_years,
        "ne": ne,
        "structured_ne": structured_ne,
        "pop_sizes": pop_sizes_flat,
        "epoch_pop_sizes": pop_sizes_arr,
        "growth_rates": growth_rates_flat,
        "epoch_growth_rates": growth_rates_arr,
        "log_likelihood": float(prev_ll),
        "n_iterations": len(rounds),
        "rounds": rounds,
        "interval_boundaries": interval_boundaries,
        "n_ref": float(n_ref),
        "composite_mode": resolved_native.composite_mode,
        "loci_per_hmm_step": resolved_native.loci_per_hmm_step,
        "ordered_params": params.ordered_param_values(),
        "param_ids": list(params.ordered_param_ids),
        "core_type": best_core_type,
        "resolved_options": _resolved_options_metadata(resolved_native),
    }, implementation_requested=implementation, implementation_used=implementation_used)
    data.params.setdefault("mu", mu)
    data.params.setdefault("generation_time", generation_time)
    return data


def _parse_dical2_stdout(stdout: str) -> tuple[list[dict], dict | None]:
    em_path: list[dict] = []
    best: dict | None = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split("\t")
        if len(parts) < 3:
            continue
        try:
            log_likelihood = float(parts[0])
            elapsed_ms = float(parts[1])
            params = [float(x) for x in parts[2:-1]]
            run_id = parts[-1]
        except ValueError:
            continue
        row = {
            "log_likelihood": log_likelihood,
            "elapsed_ms": elapsed_ms,
            "params": params,
            "id": run_id,
        }
        em_path.append(row)
        if best is None or log_likelihood > best["log_likelihood"]:
            best = row
    return em_path, best


def _bounds_to_string(bounds: str | list[tuple[float, float]] | None) -> str | None:
    if bounds is None:
        return None
    if isinstance(bounds, str):
        return bounds
    return ";".join(f"{lo},{hi}" for lo, hi in bounds)


def _interval_type_cli_name(interval_type: str | None) -> str | None:
    if interval_type is None:
        return None
    if interval_type == "loguniform":
        return "logUniform"
    if interval_type == "customfixed":
        return "customFixed"
    return interval_type


def _dical2_upstream(
    data: SmcData,
    *,
    resolved: DiCal2ResolvedOptions,
    cli_args: list[str],
    implementation_requested: str,
) -> SmcData:
    import smckit.upstream as upstream_api

    source_paths = data.uns.get("source_paths", {})
    jar = Path(__file__).resolve().parents[3] / "vendor/diCal2/diCal2.jar"
    if not jar.exists():
        require_upstream_available("dical2")
    if not method_upstream_available("dical2"):
        message = _dical2_java_help()
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        raise RuntimeError(message)
    required = ["param_file", "demo_file", "config_file", "reference_file", "sequences"]
    missing = [key for key in required if not source_paths.get(key)]
    if missing:
        raise ValueError(
            "Upstream diCal2 requires path-backed inputs from read_dical2(...): "
            + ", ".join(missing)
        )
    resolved_paths = {
        key: str(Path(value).resolve())
        for key, value in source_paths.items()
        if value is not None
    }
    java_path = upstream_api.status("dical2")["runtime"]["path"]
    if java_path is None:
        raise RuntimeError(_dical2_java_help())
    cmd = [
        str(java_path),
        "-jar",
        str(jar),
        "--paramFile",
        resolved_paths["param_file"],
        "--demoFile",
        resolved_paths["demo_file"],
        "--vcfFile",
        resolved_paths["sequences"],
        "--vcfFilterPassString",
        ".",
        "--vcfReferenceFile",
        resolved_paths["reference_file"],
        "--configFile",
        resolved_paths["config_file"],
        "--lociPerHmmStep",
        str(int(resolved.loci_per_hmm_step)),
        "--compositeLikelihood",
        str(resolved.composite_mode),
        "--numberIterationsEM",
        str(int(resolved.number_iterations_em)),
        "--seed",
        str(0 if resolved.seed is None else int(resolved.seed)),
    ]
    if resolved_paths.get("rates_file"):
        cmd.extend(["--ratesFile", resolved_paths["rates_file"]])
    if resolved.number_iterations_mstep is not None:
        cmd.extend(["--numberIterationsMstep", str(int(resolved.number_iterations_mstep))])
    elif resolved.relative_error_m is not None:
        cmd.extend(["--relativeErrorM", str(float(resolved.relative_error_m))])
    else:
        cmd.extend(["--numberIterationsMstep", "1"])
    if resolved.relative_error_e is not None:
        cmd.extend(["--relativeErrorE", str(float(resolved.relative_error_e))])
    if resolved.start_point is not None:
        cmd.extend(["--startPoint", ",".join(str(float(x)) for x in resolved.start_point)])
    if resolved.meta_start_file is not None:
        cmd.extend(["--metaStartFile", str(Path(resolved.meta_start_file).resolve())])
        cmd.extend(["--metaNumIterations", str(int(resolved.meta_num_iterations))])
        cmd.extend(["--metaKeepBest", str(int(resolved.meta_keep_best))])
        if resolved.meta_num_points is not None:
            cmd.extend(["--metaNumPoints", str(int(resolved.meta_num_points))])
    interval_type = _interval_type_cli_name(resolved.interval_type)
    if interval_type is not None and resolved.interval_params is not None:
        cmd.extend(["--intervalType", interval_type, "--intervalParams", resolved.interval_params])
    bounds_str = _bounds_to_string(resolved.bounds)
    if bounds_str is not None:
        cmd.extend(["--bounds", bounds_str])
    if not resolved.coordinatewise_mstep:
        cmd.append("--disableCoordinateWiseMStep")
    if resolved.coordinate_order is not None:
        cmd.extend(["--coordinateOrder", ",".join(str(x) for x in resolved.coordinate_order)])
    if resolved.nm_fraction is not None:
        cmd.extend(["--nmFraction", str(float(resolved.nm_fraction))])
    if resolved.use_param_rel_err:
        cmd.append("--useParamRelErr")
    if resolved.use_param_rel_err_m:
        cmd.append("--useParamRelErrM")
    cmd.extend(cli_args)
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=jar.parent)
    if proc.returncode != 0:
        raise RuntimeError(
            "Upstream diCal2 backend failed.\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    em_path, best = _parse_dical2_stdout(proc.stdout)
    data.results["dical2"] = annotate_result(
        {
            "backend": "upstream",
            "log_likelihood": np.nan if best is None else float(best["log_likelihood"]),
            "best_params": None if best is None else np.asarray(best["params"], dtype=np.float64),
            "em_path": em_path,
            "resolved_options": _resolved_options_metadata(resolved),
            "upstream": standard_upstream_metadata(
                "dical2",
                effective_args={
                    "compositeLikelihood": resolved.composite_mode,
                    "lociPerHmmStep": int(resolved.loci_per_hmm_step),
                    "seed": 0 if resolved.seed is None else int(resolved.seed),
                    "cli_args": cli_args,
                },
                extra={
                    "java": str(java_path),
                    "jar": str(jar),
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                },
            ),
        },
        implementation_requested=implementation_requested,
        implementation_used="upstream",
    )
    return data
