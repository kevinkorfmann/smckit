"""Validate smckit SMC++ against msprime simulation with known demography.

Simulates coalescent data under a constant-size population, converts to a
run-length encoded SMC++-style observation stream, runs inference, and checks:
1. Theta/N0 estimation accuracy (Watterson's estimator)
2. HMM mathematical correctness (normalization, stochasticity)
3. Ne recovery within expected tolerance for the current CSFS model
"""


import copy
import numpy as np
import pytest

from smckit._core import SmcData
from smckit.tl._smcpp import (
    _compute_psmc_style_transition,
    _forward_spans,
    _lineage_rate_matrix,
    _log_likelihood_spans,
    _propagate_lineages,
    _resolve_upstream_smcpp_python,
    _sfs_weights,
    compute_csfs,
    compute_hmm_params,
    compute_time_intervals,
    smcpp,
)

# Try to import msprime; skip if not available
msprime = pytest.importorskip("msprime")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N0 = 10000
MU = 1.25e-8
R = 1e-8
GEN_TIME = 25
N_HAPLOTYPES = 12
N_UNDIST = N_HAPLOTYPES - 1
K = 8
MAX_T = 10.0


def _legacy_one_distinguished(data: SmcData) -> SmcData:
    data = copy.deepcopy(data)
    data.uns["n_distinguished"] = 1
    data.uns["records"] = [
        {**record, "n_distinguished": 1}
        for record in data.uns["records"]
    ]
    return data


@pytest.fixture(scope="module")
def simulated_data():
    """Simulate 10 Mb under constant population, convert to SMC++ format."""
    seq_length = 10_000_000

    demography = msprime.Demography()
    demography.add_population(name="pop", initial_size=N0)

    ts = msprime.sim_ancestry(
        samples={"pop": N_HAPLOTYPES // 2},
        demography=demography,
        sequence_length=seq_length,
        recombination_rate=R,
        random_seed=42,
    )
    ts = msprime.sim_mutations(ts, rate=MU, random_seed=42)

    observations = []
    prev_pos = 0
    for variant in ts.variants():
        geno = variant.genotypes
        if np.any(geno > 1):
            continue
        pos = int(variant.site.position)
        a = int(geno[0])
        b = int(np.sum(geno[1:]))
        total = a + b
        if total == 0 or total == N_HAPLOTYPES:
            continue
        gap = max(pos - prev_pos, 0)
        if gap > 0:
            observations.append((gap, 0, 0))
        observations.append((1, a, b))
        prev_pos = pos + 1
    if prev_pos < seq_length:
        observations.append((seq_length - prev_pos, 0, 0))

    data = SmcData(
        uns={
            "records": [{"name": "sim_chr1", "observations": observations}],
            "n_undist": N_UNDIST,
        },
    )
    return data, observations


# ---------------------------------------------------------------------------
# Mathematical component tests
# ---------------------------------------------------------------------------


def test_lineage_counting_monotone():
    """Expected lineage count must decrease over time."""
    Q = _lineage_rate_matrix(N_UNDIST)
    t = compute_time_intervals(K, MAX_T)
    eta = np.ones(K)
    p = _propagate_lineages(Q, N_UNDIST, t, eta)
    j_vec = np.arange(1, N_UNDIST + 1, dtype=np.float64)
    E_A = [float(j_vec @ p[k]) for k in range(K + 1)]
    for i in range(K):
        assert E_A[i] >= E_A[i + 1] - 1e-10


def test_sfs_weights_normalized():
    """SFS weights must sum to 1 for each lineage count."""
    w = _sfs_weights(N_UNDIST)
    for j in range(N_UNDIST):
        row_sum = w[j].sum()
        if row_sum > 0:
            assert abs(row_sum - 1.0) < 1e-10


def test_sigma_distribution_normalized():
    """Initial distribution from PSMC-style transition must sum to 1."""
    t = compute_time_intervals(K, MAX_T)
    eta = np.ones(K)
    _, sigma, _, _ = _compute_psmc_style_transition(N_UNDIST, K, t, eta, 0.0002)
    assert abs(sigma.sum() - 1.0) < 1e-4


def test_csfs_columns_normalized():
    """CSFS columns (excl. missing) must sum to 1."""
    t = compute_time_intervals(K, MAX_T)
    eta = np.ones(K)
    _, _, avg_t, E_A = _compute_psmc_style_transition(N_UNDIST, K, t, eta, 0.0002)
    e = compute_csfs(N_UNDIST, K, t, eta, theta=0.0005, avg_t=avg_t, E_A_mid=E_A)
    for k in range(K):
        col_sum = e[:-1, k].sum()
        assert abs(col_sum - 1.0) < 1e-4


def test_transition_matrix_stochastic():
    """Transition matrix rows must sum to 1."""
    t = compute_time_intervals(K, MAX_T)
    eta = np.ones(K)
    a, _, _, _ = _compute_psmc_style_transition(N_UNDIST, K, t, eta, 0.0002)
    for k in range(K):
        assert abs(a[k].sum() - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Forward algorithm test
# ---------------------------------------------------------------------------


def test_forward_log_likelihood_finite(simulated_data):
    """Forward algorithm must produce finite negative log-likelihood."""
    _, observations = simulated_data
    t = compute_time_intervals(K, MAX_T)
    eta = np.ones(K)
    theta = 4 * N0 * MU
    rho_base = (R / MU) * theta / 2.0
    hp = compute_hmm_params(eta, N_UNDIST, t, theta, rho_base)
    _, s_list = _forward_spans(hp, observations[:500])
    ll = _log_likelihood_spans(s_list)
    assert np.isfinite(ll)
    assert ll < 0


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_smcpp_theta_estimation(simulated_data):
    """Watterson's estimator should recover theta within 20% of truth."""
    data, _ = simulated_data
    data = _legacy_one_distinguished(data)
    data = smcpp(
        data,
        n_intervals=K,
        max_t=MAX_T,
        mu=MU,
        recombination_rate=R,
        generation_time=GEN_TIME,
        regularization=10.0,
        max_iterations=50,
        seed=42,
        backend="native",
    )
    res = data.results["smcpp"]
    theta_true = 4 * N0 * MU
    rel_diff = abs(res["theta"] - theta_true) / theta_true
    assert rel_diff < 0.20, f"Theta relative diff {rel_diff:.3f} > 0.20"
    assert res["n_distinguished"] == 1


@pytest.mark.slow
def test_smcpp_n0_estimation(simulated_data):
    """N0 estimate should be within 20% of truth."""
    data, _ = simulated_data
    data = _legacy_one_distinguished(data)
    data = smcpp(
        data,
        n_intervals=K,
        max_t=MAX_T,
        mu=MU,
        recombination_rate=R,
        generation_time=GEN_TIME,
        regularization=10.0,
        max_iterations=50,
        seed=42,
        backend="native",
    )
    res = data.results["smcpp"]
    rel_diff = abs(res["n0"] - N0) / N0
    assert rel_diff < 0.20, f"N0 relative diff {rel_diff:.3f} > 0.20"


@pytest.mark.slow
def test_smcpp_ne_within_order_of_magnitude(simulated_data):
    """All recovered Ne values should be within 10x of truth."""
    data, _ = simulated_data
    data = _legacy_one_distinguished(data)
    data = smcpp(
        data,
        n_intervals=K,
        max_t=MAX_T,
        mu=MU,
        recombination_rate=R,
        generation_time=GEN_TIME,
        regularization=10.0,
        max_iterations=50,
        seed=42,
        backend="native",
    )
    ne = data.results["smcpp"]["ne"]
    within_10x = np.mean((ne > N0 / 10) & (ne < N0 * 10))
    assert within_10x == 1.0, f"Only {within_10x:.0%} of Ne within 10x of truth"


@pytest.mark.slow
@pytest.mark.skipif(
    _resolve_upstream_smcpp_python() is None,
    reason="Upstream SMC++ side environment not available",
)
def test_smcpp_native_vs_upstream_agreement_on_default_onepop_path():
    """Default native SMC++ should stay close to the upstream one-pop path."""
    observations = []
    for i in range(120):
        observations.append((4999, 0, 0))
        observations.append((1, i % 3, (i * 2) % 6))

    data = SmcData(
        uns={
            "records": [{"name": "default_onepop", "observations": observations, "n_undist": 5}],
            "n_undist": 5,
        },
    )

    native = smcpp(
        copy.deepcopy(data),
        n_intervals=4,
        regularization=10.0,
        max_iterations=2,
        seed=42,
        backend="native",
    ).results["smcpp"]
    upstream = smcpp(
        copy.deepcopy(data),
        n_intervals=4,
        regularization=10.0,
        max_iterations=2,
        seed=42,
        backend="upstream",
    ).results["smcpp"]

    t_min = max(
        float(np.min(native["time"][native["time"] > 0])),
        float(np.min(upstream["time"][upstream["time"] > 0])),
    )
    t_max = min(float(np.max(native["time"])), float(np.max(upstream["time"])))
    grid = np.geomspace(t_min, t_max, 200)

    def _step_eval(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(times, query, side="right") - 1
        idx = np.clip(idx, 0, len(values) - 1)
        return values[idx]

    native_ne = _step_eval(np.asarray(native["time"]), np.asarray(native["ne"]), grid)
    upstream_ne = _step_eval(np.asarray(upstream["time"]), np.asarray(upstream["ne"]), grid)

    rel = np.abs(native_ne - upstream_ne) / np.maximum(np.abs(upstream_ne), 1e-12)
    log_corr = float(np.corrcoef(np.log(native_ne), np.log(upstream_ne))[0, 1])
    scale_ratio = float(np.median(native_ne) / np.median(upstream_ne))
    median_log10_error = float(np.median(np.abs(np.log10(native_ne / upstream_ne))))

    assert native["n_distinguished"] == 2
    assert upstream["n_distinguished"] == 2
    assert native["preprocessing"]["applied"] is True
    assert native["observation_scale"] == 100.0
    assert upstream["optimization"]["history"]
    assert log_corr >= 0.999
    assert 0.998 < scale_ratio < 1.002
    assert median_log10_error < 0.001
    assert float(np.median(rel)) < 0.002


@pytest.mark.slow
@pytest.mark.skipif(
    _resolve_upstream_smcpp_python() is None,
    reason="Upstream SMC++ side environment not available",
)
def test_smcpp_two_distinguished_native_vs_upstream_is_tracked():
    observations = []
    for i in range(120):
        observations.append((4999, 0, 0))
        observations.append((1, i % 3, (i * 2) % 6))

    data = SmcData(
        uns={
            "records": [{"name": "synthetic2", "observations": observations, "n_undist": 5, "n_distinguished": 2}],
            "n_undist": 5,
            "n_distinguished": 2,
        },
    )

    native = smcpp(
        copy.deepcopy(data),
        n_intervals=4,
        regularization=10.0,
        max_iterations=2,
        seed=42,
        backend="native",
    ).results["smcpp"]
    upstream = smcpp(
        copy.deepcopy(data),
        n_intervals=4,
        regularization=10.0,
        max_iterations=2,
        seed=42,
        backend="upstream",
    ).results["smcpp"]

    grid = np.geomspace(
        max(float(np.min(native["time"])), float(np.min(upstream["time"]))),
        min(float(np.max(native["time"])), float(np.max(upstream["time"]))),
        200,
    )

    def _step_eval(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(times, query, side="right") - 1
        idx = np.clip(idx, 0, len(values) - 1)
        return values[idx]

    native_ne = _step_eval(np.asarray(native["time"]), np.asarray(native["ne"]), grid)
    upstream_ne = _step_eval(np.asarray(upstream["time"]), np.asarray(upstream["ne"]), grid)

    rel = np.abs(native_ne - upstream_ne) / np.maximum(np.abs(upstream_ne), 1e-12)
    log_corr = float(np.corrcoef(np.log(native_ne), np.log(upstream_ne))[0, 1])
    scale_ratio = float(np.median(native_ne) / np.median(upstream_ne))
    median_log10_error = float(np.median(np.abs(np.log10(native_ne / upstream_ne))))

    assert native["observation_scale"] == 100.0
    assert upstream["optimization"]["history"]
    assert log_corr >= 0.94
    assert 0.97 < scale_ratio < 1.04
    assert median_log10_error < 0.02
    assert float(np.median(rel)) < 0.04
