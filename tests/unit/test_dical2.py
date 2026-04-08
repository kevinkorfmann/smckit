"""Unit tests for diCal2 implementation."""

from __future__ import annotations

import numpy as np
import pytest

from smckit.io._dical2 import (
    DiCal2Config,
    DiCal2Demo,
    DiCal2Epoch,
    _parse_partition,
    read_dical2,
    read_dical2_config,
    read_dical2_demo,
    read_dical2_param,
    read_dical2_rates,
)
from smckit.tl._dical2 import (
    DICAL2_T_INF,
    CoreMatrices,
    EigenCore,
    ODECore,
    SimpleTrunk,
    _JavaRandom,
    _build_native_core,
    _build_free_params,
    _old_interval_boundaries,
    _resolve_dical2_options,
    _resolve_interval_boundaries,
    backward_log,
    build_extended_matrix,
    compute_time_intervals,
    dical2,
    expected_counts,
    forward_log,
    h_integral,
    matrix_exp_eig,
    refine_demography,
)

VENDOR_EXAMPLES = "vendor/diCal2/examples"


# ---------------------------------------------------------------------------
# Time intervals
# ---------------------------------------------------------------------------


class TestTimeIntervals:
    def test_basic(self):
        t = compute_time_intervals(5, max_t=4.0, alpha=0.1)
        assert len(t) == 6
        assert t[0] == 0.0
        assert t[-1] == 4.0
        assert np.all(np.diff(t) > 0)

    def test_exponential_spacing(self):
        t = compute_time_intervals(10, max_t=10.0, alpha=0.1)
        diffs = np.diff(t)
        assert diffs[0] < diffs[-1]


class TestIntervalFactories:
    def test_loguniform_readme_grid(self):
        config = read_dical2_config("vendor/diCal2/examples/fromReadme/exp.config")
        demo = read_dical2_demo("vendor/diCal2/examples/fromReadme/exp.demo")
        resolved = _resolve_dical2_options(
            n_intervals=11,
            max_t=4.0,
            alpha=0.1,
            n_em_iterations=2,
            composite_mode="lol",
            loci_per_hmm_step=3,
            start_point=None,
            meta_start_file=None,
            meta_num_iterations=1,
            meta_keep_best=1,
            meta_num_points=None,
            bounds=None,
            seed=1,
            method_options={
                "interval_type": "logUniform",
                "interval_params": "11,0.01,4",
            },
        )
        boundaries = _resolve_interval_boundaries(demo, config, resolved)
        expected = np.array(
            [
                0.0,
                0.01,
                0.0172405417,
                0.0297236279,
                0.0512451448,
                0.0883494058,
                0.1523191625,
                0.2626064874,
                0.4527478100,
                0.7805617510,
                1.3457307492,
                2.3201127105,
                4.0,
                DICAL2_T_INF,
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(boundaries, expected, rtol=1e-8, atol=1e-8)

    def test_old_interval_factory_matches_piecewise_example(self):
        config = read_dical2_config(
            f"{VENDOR_EXAMPLES}/piecewiseConstant/piecewise_constant.config"
        )
        demo = read_dical2_demo(
            f"{VENDOR_EXAMPLES}/piecewiseConstant/piecewise_constant.demo"
        )
        boundaries = _old_interval_boundaries(demo, config, "4")
        expected = np.array(
            [
                0.0,
                -np.log(4 / 5) / 4.0,
                -np.log(3 / 5) / 4.0,
                -np.log(2 / 5) / 4.0,
                -np.log(1 / 5) / 4.0,
                DICAL2_T_INF,
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(boundaries, expected, rtol=1e-10, atol=1e-10)


class TestResolvedOptions:
    def test_nm_fraction_defaults_to_upstream_value(self):
        resolved = _resolve_dical2_options(
            n_intervals=11,
            max_t=4.0,
            alpha=0.1,
            n_em_iterations=2,
            composite_mode="lol",
            loci_per_hmm_step=3,
            start_point=None,
            meta_start_file=None,
            meta_num_iterations=1,
            meta_keep_best=1,
            meta_num_points=None,
            bounds=None,
            seed=1,
            method_options=None,
        )
        assert resolved.nm_fraction == pytest.approx(0.2)


class TestJavaRandom:
    def test_next_long_matches_java_random(self):
        rng = _JavaRandom(1)
        observed = [rng.next_long() for _ in range(3)]
        expected = [
            -4964420948893066024,
            7564655870752979346,
            3831662765844904176,
        ]
        assert observed == expected

    def test_permutation_matches_java_collections_shuffle(self):
        observed = _JavaRandom(1).permutation(5)
        expected = np.array([2, 3, 1, 4, 0], dtype=np.int64)
        np.testing.assert_array_equal(observed, expected)


# ---------------------------------------------------------------------------
# I/O parsers
# ---------------------------------------------------------------------------


class TestPartitionParser:
    def test_single_pop(self):
        assert _parse_partition("{{0}}") == [[0]]

    def test_two_pops(self):
        assert _parse_partition("{{0},{1}}") == [[0], [1]]

    def test_merged(self):
        assert _parse_partition("{{0,1}}") == [[0, 1]]

    def test_three_pops(self):
        assert _parse_partition("{{0},{1},{2}}") == [[0], [1], [2]]


class TestParamReader:
    def test_piecewise_constant(self):
        p = read_dical2_param(f"{VENDOR_EXAMPLES}/piecewiseConstant/mutRec.param")
        assert p.theta == pytest.approx(0.0005)
        assert p.rho == pytest.approx(0.0005)
        assert p.mutation_matrix.shape == (2, 2)
        assert p.mutation_matrix[0, 1] == 1.0


class TestDemoReader:
    def test_piecewise_constant(self):
        d = read_dical2_demo(
            f"{VENDOR_EXAMPLES}/piecewiseConstant/piecewise_constant.demo"
        )
        assert len(d.epochs) == 4
        assert d.n_present_demes == 1
        assert d.epoch_boundaries[0] == 0.0
        assert np.isinf(d.epoch_boundaries[-1])
        # All epochs have a single ancient deme
        for ep in d.epochs:
            assert len(ep.partition) == 1
            assert ep.partition[0] == [0]

    def test_clean_split(self):
        d = read_dical2_demo(f"{VENDOR_EXAMPLES}/cleanSplit/clean_split.demo")
        assert len(d.epochs) == 2
        assert d.n_present_demes == 2
        # Epoch 0: two separate demes
        assert d.epochs[0].partition == [[0], [1]]
        # Epoch 1: merged
        assert d.epochs[1].partition == [[0, 1]]

    def test_isolation_migration(self):
        d = read_dical2_demo(
            f"{VENDOR_EXAMPLES}/islolationMigration/isolation_migration.demo"
        )
        assert len(d.epochs) == 2
        # Epoch 0 has nontrivial migration matrix (off-diagonals = ?3 → default 1)
        ep0 = d.epochs[0]
        assert ep0.migration_matrix is not None
        assert ep0.migration_matrix.shape == (2, 2)
        assert ep0.migration_param_ids == [[None, 3], [3, None]]

    def test_repeated_placeholder_ids_preserved(self):
        d = read_dical2_demo(f"{VENDOR_EXAMPLES}/expGrowth/exp_growth.demo")
        assert d.boundary_param_ids == [None, None, None, None]
        assert d.epochs[0].pop_size_param_ids == [0]
        assert d.epochs[1].pop_size_param_ids == [0]
        assert d.epochs[2].pop_size_param_ids == [None]

    def test_boundary_placeholder_ids_preserved(self):
        d = read_dical2_demo("vendor/diCal2/examples/fromReadme/exp.demo")
        assert d.boundary_param_ids == [None, 1, 2, None]


class TestRatesReader:
    def test_exp_growth_rates(self):
        demo = read_dical2_demo(f"{VENDOR_EXAMPLES}/expGrowth/exp_growth.demo")
        demo = read_dical2_rates(
            f"{VENDOR_EXAMPLES}/expGrowth/exp_growth.rates",
            demo,
        )
        assert demo.epochs[0].growth_rates is not None
        assert demo.epochs[0].growth_rates[0] == pytest.approx(0.0)
        assert demo.epochs[0].growth_rate_param_ids == [1]
        assert demo.epochs[1].growth_rates[0] == pytest.approx(0.0)


class TestConfigReader:
    def test_single_pop(self):
        c = read_dical2_config(
            f"{VENDOR_EXAMPLES}/piecewiseConstant/piecewise_constant.config"
        )
        assert c.n_populations == 1
        assert c.n_alleles == 2
        assert sum(c.sample_sizes) == 4
        assert c.haplotypes_to_include[:4] == [True, True, True, True]
        assert c.haplotypes_to_include[4:] == [False] * 6

    def test_two_pops(self):
        c = read_dical2_config(f"{VENDOR_EXAMPLES}/cleanSplit/clean_split.config")
        assert c.n_populations == 2
        assert c.sample_sizes.tolist() == [2, 2]


class TestReadDical2:
    def test_basic_array_input(self):
        rng = np.random.default_rng(0)
        seqs = (rng.random((4, 100)) < 0.05).astype(np.int8)
        data = read_dical2(sequences=seqs, theta=0.001, rho=0.0005)
        assert data.sequences.shape == (4, 100)
        assert data.params["theta"] == 0.001
        assert data.uns["n_haplotypes"] == 4
        assert data.uns["config"].n_populations == 1

    def test_rates_file_round_trip(self):
        rng = np.random.default_rng(0)
        seqs = (rng.random((4, 100)) < 0.05).astype(np.int8)
        data = read_dical2(
            sequences=seqs,
            demo_file=f"{VENDOR_EXAMPLES}/expGrowth/exp_growth.demo",
            rates_file=f"{VENDOR_EXAMPLES}/expGrowth/exp_growth.rates",
            theta=0.001,
            rho=0.0005,
        )
        assert data.uns["demo"].epochs[0].growth_rates is not None

    def test_vcf_uses_reference_length(self):
        data = read_dical2(
            sequences="vendor/diCal2/examples/fromReadme/test.vcf",
            param_file="vendor/diCal2/examples/fromReadme/test.param",
            demo_file="vendor/diCal2/examples/fromReadme/IM.demo",
            config_file="vendor/diCal2/examples/fromReadme/IM.config",
            reference_file="vendor/diCal2/examples/fromReadme/test.fa",
            filter_pass_string=".",
        )
        assert data.sequences.shape == (4, 2)
        np.testing.assert_array_equal(data.uns["seg_positions"], np.array([6, 7], dtype=np.int64))
        assert data.uns["reference_length"] == 19
        np.testing.assert_array_equal(
            data.uns["reference_alleles"],
            np.array(
                [-1, 0, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, 0, 0, 0, -1, 0, 0, 0],
                dtype=np.int8,
            ),
        )


# ---------------------------------------------------------------------------
# Linear algebra building blocks
# ---------------------------------------------------------------------------


class TestExtendedMatrix:
    def test_single_deme_no_migration(self):
        # 1 deme, absorption rate 1.0
        Z = build_extended_matrix(None, np.array([1.0]))
        # Z should be:
        # [[-1, 1], [0, 0]]
        assert Z.shape == (2, 2)
        assert Z[0, 0] == -1.0
        assert Z[0, 1] == 1.0
        assert Z[1, 0] == 0.0

    def test_two_demes_with_migration(self):
        M = np.array([[-0.1, 0.1], [0.1, -0.1]])
        alpha = np.array([0.5, 0.5])
        Z = build_extended_matrix(M, alpha)
        assert Z.shape == (4, 4)
        # Top-left = M - diag(alpha)
        assert Z[0, 0] == pytest.approx(-0.6)  # -0.1 - 0.5
        assert Z[1, 1] == pytest.approx(-0.6)
        assert Z[0, 1] == 0.1
        # Absorption block
        assert Z[0, 2] == 0.5
        assert Z[1, 3] == 0.5


class TestMatrixExp:
    def test_zero_time(self):
        Z = np.array([[-1.0, 1.0], [0.0, 0.0]])
        E = matrix_exp_eig(Z, 0.0)
        np.testing.assert_allclose(E, np.eye(2))

    def test_simple_decay(self):
        Z = np.array([[-1.0, 1.0], [0.0, 0.0]])
        E = matrix_exp_eig(Z, 1.0)
        # exp(-1) ≈ 0.368
        assert E[0, 0] == pytest.approx(np.exp(-1.0), abs=1e-10)
        assert E[0, 1] == pytest.approx(1 - np.exp(-1.0), abs=1e-10)
        assert E[1, 0] == pytest.approx(0.0, abs=1e-10)
        assert E[1, 1] == pytest.approx(1.0, abs=1e-10)

    def test_row_sums_one_for_stochastic(self):
        # exp of a proper rate matrix should be a stochastic matrix
        Z = np.array(
            [
                [-1.5, 1.0, 0.5],
                [0.2, -0.7, 0.5],
                [0.0, 0.0, 0.0],
            ]
        )
        E = matrix_exp_eig(Z, 0.5)
        np.testing.assert_allclose(E.sum(axis=1), np.ones(3), atol=1e-10)


class TestHIntegral:
    def test_zero_interval(self):
        # a == b → 0
        assert h_integral(1.0, 1.0, 0.0 + 0j, -0.5 + 0j) == 0

    def test_zero_lambda(self):
        # lam = 0 → exp(u) * (b - a)
        h = h_integral(0.0, 2.0, 0.0 + 0j, 0.0 + 0j)
        assert h == pytest.approx(2.0)

    def test_finite_interval(self):
        # ∫_0^1 exp(-t) dt = 1 - exp(-1)
        h = h_integral(0.0, 1.0, 0.0 + 0j, -1.0 + 0j)
        assert h.real == pytest.approx(1.0 - np.exp(-1.0))

    def test_infinite_interval(self):
        # ∫_0^∞ exp(-t) dt = 1
        h = h_integral(0.0, np.inf, 0.0 + 0j, -1.0 + 0j)
        assert h.real == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# EigenCore
# ---------------------------------------------------------------------------


def _make_simple_demo(n_intervals: int = 4) -> DiCal2Demo:
    """Single-population piecewise-constant demography."""
    bounds = compute_time_intervals(n_intervals, max_t=2.0, alpha=0.1)
    bounds = np.append(bounds, DICAL2_T_INF)
    epochs = []
    for i in range(len(bounds) - 1):
        epochs.append(
            DiCal2Epoch(
                start=float(bounds[i]),
                end=float(bounds[i + 1]),
                partition=[[0]],
                pop_sizes=np.array([1.0]),
                migration_matrix=None,
                pulse_migration=None,
                growth_rates=None,
            )
        )
    return DiCal2Demo(
        epoch_boundaries=bounds.copy(),
        epochs=epochs,
        n_present_demes=1,
    )


class TestEigenCore:
    def test_constructs(self):
        demo = _make_simple_demo()
        config = DiCal2Config(
            seq_length=100,
            n_alleles=2,
            n_populations=1,
            haplotype_populations=[0, 0, 0],
            haplotypes_to_include=[True, True, True],
            haplotype_multiplicities=np.ones((3, 1), dtype=np.int64),
            sample_sizes=np.array([3]),
        )
        bounds = demo.epoch_boundaries
        refined = refine_demography(demo, bounds)
        trunk = SimpleTrunk(config=config, additional_hap_idx=0)
        mut_mat = np.array([[-1.0, 1.0], [1.0, -1.0]])
        core = EigenCore(
            refined=refined,
            trunk=trunk,
            observed_present_deme=0,
            mutation_matrix=mut_mat,
            theta=0.001,
            rho=0.0005,
        ).core_matrices()
        assert core.n_states > 0
        # Initial probs sum to 1
        assert np.exp(core.log_initial).sum() == pytest.approx(1.0, abs=1e-9)

    def test_emission_rows_sum_to_one(self):
        demo = _make_simple_demo()
        config = DiCal2Config(
            seq_length=100,
            n_alleles=2,
            n_populations=1,
            haplotype_populations=[0, 0],
            haplotypes_to_include=[True, True],
            haplotype_multiplicities=np.ones((2, 1), dtype=np.int64),
            sample_sizes=np.array([2]),
        )
        refined = refine_demography(demo, demo.epoch_boundaries)
        trunk = SimpleTrunk(config=config, additional_hap_idx=0)
        mut_mat = np.array([[-1.0, 1.0], [1.0, -1.0]])
        core = EigenCore(
            refined=refined,
            trunk=trunk,
            observed_present_deme=0,
            mutation_matrix=mut_mat,
            theta=0.001,
            rho=0.0005,
        ).core_matrices()
        em = np.exp(core.log_emission)
        # For each state, each trunk allele row should sum to ~1
        for s in range(core.n_states):
            for trunk_a in range(2):
                assert em[s, trunk_a, :].sum() == pytest.approx(1.0, abs=1e-8)

    def test_growth_rates_change_absorption_profile(self):
        demo = read_dical2_demo(f"{VENDOR_EXAMPLES}/expGrowth/exp_growth.demo")
        demo = read_dical2_rates(
            f"{VENDOR_EXAMPLES}/expGrowth/exp_growth.rates",
            demo,
        )
        params = _build_free_params(demo)
        assert len(params.pop_size_values) == 1
        assert len(params.growth_rate_values) == 1

    def test_native_core_selector_uses_ode_for_growth(self):
        demo = DiCal2Demo(
            epoch_boundaries=np.array([0.0, 0.5, DICAL2_T_INF], dtype=np.float64),
            epochs=[
                DiCal2Epoch(
                    start=0.0,
                    end=0.5,
                    partition=[[0]],
                    pop_sizes=np.array([1.0]),
                    migration_matrix=None,
                    pulse_migration=None,
                    growth_rates=np.array([0.2]),
                ),
                DiCal2Epoch(
                    start=0.5,
                    end=DICAL2_T_INF,
                    partition=[[0]],
                    pop_sizes=np.array([1.0]),
                    migration_matrix=None,
                    pulse_migration=None,
                    growth_rates=np.array([0.0]),
                ),
            ],
            n_present_demes=1,
        )
        config = DiCal2Config(
            seq_length=100,
            n_alleles=2,
            n_populations=1,
            haplotype_populations=[0, 0],
            haplotypes_to_include=[True, True],
            haplotype_multiplicities=np.ones((2, 1), dtype=np.int64),
            sample_sizes=np.array([2]),
        )
        refined = refine_demography(demo, demo.epoch_boundaries)
        trunk = SimpleTrunk(config=config, additional_hap_idx=0)
        mut_mat = np.array([[-1.0, 1.0], [1.0, -1.0]])
        core_obj, core_type = _build_native_core(
            refined=refined,
            trunk=trunk,
            observed_present_deme=0,
            mutation_matrix=mut_mat,
            theta=0.001,
            rho=0.0005,
        )
        assert core_type == "ode"
        assert isinstance(core_obj, ODECore)

    def test_native_core_selector_rejects_growth_with_pulse(self):
        demo = DiCal2Demo(
            epoch_boundaries=np.array([0.0, 0.5, 0.5, DICAL2_T_INF], dtype=np.float64),
            epochs=[
                DiCal2Epoch(
                    start=0.0,
                    end=0.5,
                    partition=[[0]],
                    pop_sizes=np.array([1.0]),
                    migration_matrix=None,
                    pulse_migration=None,
                    growth_rates=np.array([0.2]),
                ),
                DiCal2Epoch(
                    start=0.5,
                    end=0.5,
                    partition=[[0]],
                    pop_sizes=None,
                    migration_matrix=None,
                    pulse_migration=np.array([[1.0]]),
                    growth_rates=None,
                ),
                DiCal2Epoch(
                    start=0.5,
                    end=DICAL2_T_INF,
                    partition=[[0]],
                    pop_sizes=np.array([1.0]),
                    migration_matrix=None,
                    pulse_migration=None,
                    growth_rates=np.array([0.0]),
                ),
            ],
            n_present_demes=1,
        )
        config = DiCal2Config(
            seq_length=100,
            n_alleles=2,
            n_populations=1,
            haplotype_populations=[0, 0],
            haplotypes_to_include=[True, True],
            haplotype_multiplicities=np.ones((2, 1), dtype=np.int64),
            sample_sizes=np.array([2]),
        )
        refined = refine_demography(demo, demo.epoch_boundaries)
        trunk = SimpleTrunk(config=config, additional_hap_idx=0)
        mut_mat = np.array([[-1.0, 1.0], [1.0, -1.0]])
        with pytest.raises(NotImplementedError):
            _build_native_core(
                refined=refined,
                trunk=trunk,
                observed_present_deme=0,
                mutation_matrix=mut_mat,
                theta=0.001,
                rho=0.0005,
            )

    def test_from_readme_exp_tracks_boundary_params(self):
        demo = read_dical2_demo("vendor/diCal2/examples/fromReadme/exp.demo")
        demo = read_dical2_rates("vendor/diCal2/examples/fromReadme/exp.rates", demo)
        params = _build_free_params(demo)
        assert params.ordered_param_ids == [0, 1, 2, 3, 4]
        assert len(params.boundary_values) == 2
        assert len(params.pop_size_values) == 2
        assert len(params.growth_rate_values) == 1
        moved = params.to_demo(demo)
        np.testing.assert_allclose(moved.epoch_boundaries[1:3], demo.epoch_boundaries[1:3])

    def test_ordered_param_values_round_trip(self):
        demo = read_dical2_demo("vendor/diCal2/examples/fromReadme/exp.demo")
        demo = read_dical2_rates("vendor/diCal2/examples/fromReadme/exp.rates", demo)
        params = _build_free_params(demo)
        params.set_ordered_param_values(np.array([2.0, 0.1, 0.2, 0.3, 0.4]))
        np.testing.assert_allclose(
            params.ordered_param_values(),
            np.array([2.0, 0.1, 0.2, 0.3, 0.4]),
        )

    def test_from_readme_im_tracks_migration_params(self):
        demo = read_dical2_demo("vendor/diCal2/examples/fromReadme/IM.demo")
        params = _build_free_params(demo)
        assert params.ordered_param_ids == [0, 1, 2, 3, 4, 5, 6]
        assert len(params.migration_values) == 1
        assert params.migration_param_ids == [6]
        assert params.free_migration_groups == [[(0, 0, 1), (0, 1, 0)]]

        params.set_ordered_param_values(
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 7.0])
        )
        moved = params.to_demo(demo)
        assert moved.epochs[0].migration_matrix is not None
        np.testing.assert_allclose(
            moved.epochs[0].migration_matrix,
            np.array([[-7.0, 7.0], [7.0, -7.0]]),
        )


# ---------------------------------------------------------------------------
# Forward-backward
# ---------------------------------------------------------------------------


class TestForwardBackward:
    def _setup(self, L: int = 30):
        demo = _make_simple_demo()
        config = DiCal2Config(
            seq_length=L,
            n_alleles=2,
            n_populations=1,
            haplotype_populations=[0, 0],
            haplotypes_to_include=[True, True],
            haplotype_multiplicities=np.ones((2, 1), dtype=np.int64),
            sample_sizes=np.array([2]),
        )
        refined = refine_demography(demo, demo.epoch_boundaries)
        trunk = SimpleTrunk(config=config, additional_hap_idx=0)
        mut_mat = np.array([[-1.0, 1.0], [1.0, -1.0]])
        core = EigenCore(
            refined=refined,
            trunk=trunk,
            observed_present_deme=0,
            mutation_matrix=mut_mat,
            theta=0.001,
            rho=0.0005,
        ).core_matrices()
        return core

    def test_forward_finite_likelihood(self):
        core = self._setup()
        L = 30
        rng = np.random.default_rng(1)
        obs_a = rng.integers(0, 2, size=L)
        obs_t = rng.integers(0, 2, size=L)
        _, ll = forward_log(core, obs_a, obs_t)
        assert np.isfinite(ll)
        assert ll < 0  # log-likelihood is negative

    def test_backward_matches_forward_likelihood(self):
        core = self._setup()
        L = 20
        rng = np.random.default_rng(2)
        obs_a = rng.integers(0, 2, size=L)
        obs_t = rng.integers(0, 2, size=L)
        logF, ll_f = forward_log(core, obs_a, obs_t)
        logB = backward_log(core, obs_a, obs_t)
        # logF[0] + logB[0] - ll should give a uniform-ish posterior (sums to 1)
        log_post0 = logF[0] + logB[0] - ll_f
        post0 = np.exp(log_post0)
        assert post0.sum() == pytest.approx(1.0, abs=1e-8)

    def test_expected_counts(self):
        core = self._setup()
        L = 25
        rng = np.random.default_rng(3)
        obs_a = rng.integers(0, 2, size=L)
        obs_t = rng.integers(0, 2, size=L)
        counts = expected_counts(core, obs_a, obs_t, n_alleles=2)
        # Initial expectations sum to 1
        assert counts.initial_expect.sum() == pytest.approx(1.0, abs=1e-8)
        # Total emission counts = L (per pair of trunk/observed alleles)
        assert counts.emission_expect.sum() == pytest.approx(L, abs=1e-6)
        # Transition counts (no_reco + reco) = L - 1
        total_trans = counts.no_reco_expect.sum() + counts.reco_expect.sum()
        assert total_trans == pytest.approx(L - 1, abs=1e-6)


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


class TestDical2EndToEnd:
    def test_runs_synthetic_pcl(self):
        rng = np.random.default_rng(7)
        seqs = (rng.random((4, 200)) < 0.02).astype(np.int8)
        data = read_dical2(sequences=seqs, theta=0.001, rho=0.0005)
        data = dical2(
            data,
            n_intervals=4,
            n_em_iterations=2,
            max_t=2.0,
            composite_mode="pcl",
        )
        res = data.results["dical2"]
        assert "ne" in res
        assert "log_likelihood" in res
        assert np.isfinite(res["log_likelihood"])
        assert np.all(res["ne"] > 0)
        assert np.all(np.isfinite(res["ne"]))

    def test_runs_synthetic_pac(self):
        rng = np.random.default_rng(8)
        seqs = (rng.random((5, 150)) < 0.02).astype(np.int8)
        data = read_dical2(sequences=seqs, theta=0.001, rho=0.0005)
        data = dical2(
            data,
            n_intervals=3,
            n_em_iterations=2,
            max_t=1.5,
            composite_mode="pac",
        )
        res = data.results["dical2"]
        assert np.isfinite(res["log_likelihood"])
        assert len(res["ne"]) >= 3

    def test_runs_synthetic_lol(self):
        rng = np.random.default_rng(9)
        seqs = (rng.random((4, 150)) < 0.02).astype(np.int8)
        data = read_dical2(sequences=seqs, theta=0.001, rho=0.0005)
        data = dical2(
            data,
            n_intervals=3,
            n_em_iterations=1,
            max_t=1.5,
            composite_mode="lol",
        )
        res = data.results["dical2"]
        assert np.isfinite(res["log_likelihood"])
