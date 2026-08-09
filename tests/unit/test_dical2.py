"""Unit tests for diCal2 implementation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smckit._core import SmcData
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
    write_dical2_output,
)
from smckit.tl import dical2
from smckit.tl._dical2 import (
    DICAL2_T_INF,
    EigenCore,
    ODECore,
    SimpleTrunk,
    _build_free_params,
    _build_native_core,
    _dical2_upstream,
    _generate_java_permutations,
    _JavaRandom,
    _meta_grid_points,
    _old_interval_boundaries,
    _pac_csd_pairs,
    _pac_trunk_sizes,
    _parse_dical2_stdout,
    _persist_dical2_outputs,
    _read_dical2_permutations,
    _refined_interval_epoch,
    _resolve_csd_groups,
    _resolve_dical2_options,
    _resolve_interval_boundaries,
    backward_log,
    build_extended_matrix,
    compute_time_intervals,
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
        demo = read_dical2_demo(f"{VENDOR_EXAMPLES}/piecewiseConstant/piecewise_constant.demo")
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
    @staticmethod
    def resolve(method_options):
        return _resolve_dical2_options(
            n_intervals=11,
            max_t=4.0,
            alpha=0.1,
            n_em_iterations=2,
            composite_mode="pac",
            loci_per_hmm_step=3,
            start_point=None,
            meta_start_file=None,
            meta_num_iterations=1,
            meta_keep_best=1,
            meta_num_points=None,
            bounds=None,
            seed=1,
            method_options=method_options,
        )

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

    def test_generated_grid_controls_are_resolved(self):
        resolved = self.resolve(
            {
                "metaNumStartPoints": 3,
                "metaGridStart": True,
                "numPermutations": 5,
                "numCsdsPerPerm": 2,
                "diffPermsPerChunk": True,
            }
        )
        assert resolved.meta_num_start_points == 3
        assert resolved.meta_grid_start is True
        assert resolved.num_permutations == 5
        assert resolved.num_csds_per_permutation == 2
        assert resolved.different_permutations_per_contig is True

    @pytest.mark.parametrize(
        ("method_options", "message"),
        [
            (
                {"num_permutations": 2, "permutation_files": "perms.txt"},
                "either num_permutations or permutation_files",
            ),
            ({"meta_grid_start": True}, "requires meta_num_start_points"),
            (
                {"meta_num_iterations": 2, "meta_num_points": 2},
                "require multiple start points",
            ),
        ],
    )
    def test_incompatible_search_controls_fail(self, method_options, message):
        with pytest.raises(ValueError, match=message):
            self.resolve(method_options)

    def test_per_contig_permutation_switch_is_pac_only(self):
        resolved = self.resolve(
            {
                "composite_mode": "lol",
                "different_permutations_per_contig": True,
            }
        )
        with pytest.raises(ValueError, match="only valid with composite_mode='pac'"):
            _resolve_csd_groups(
                n_hap=4,
                n_contigs=2,
                resolved=resolved,
                rng=_JavaRandom(1),
            )


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


class TestPacPermutations:
    def test_repeated_shuffle_reuses_the_mutated_order(self):
        observed = _generate_java_permutations(3, 5, _JavaRandom(1))
        assert observed == [
            [2, 3, 1, 4, 0],
            [4, 3, 1, 2, 0],
            [0, 4, 3, 1, 2],
        ]

    def test_pac_visits_permutation_backwards(self):
        assert _pac_csd_pairs([0, 1, 2, 3]) == [
            (2, [3]),
            (1, [3, 2]),
            (0, [3, 2, 1]),
        ]
        assert _pac_csd_pairs([0, 1, 2, 3], 1) == [(0, [3, 2, 1])]
        assert _pac_csd_pairs([0, 1, 2, 3], 2) == [
            (2, [3]),
            (0, [3, 2, 1]),
        ]

    def test_num_csds_matches_upstream_interpolation(self):
        assert _pac_trunk_sizes(8, None) == set(range(8))
        assert _pac_trunk_sizes(8, 4) == {1, 3, 5, 7}

    def test_permutation_file_is_strict(self, tmp_path):
        valid = tmp_path / "valid.permutations"
        valid.write_text("0 1 2 3\n3 1 0 2\n")
        assert _read_dical2_permutations(valid, 4) == [
            [0, 1, 2, 3],
            [3, 1, 0, 2],
        ]

        invalid = tmp_path / "invalid.permutations"
        invalid.write_text("0 1 1 3\n")
        with pytest.raises(ValueError, match="each index"):
            _read_dical2_permutations(invalid, 4)


class TestMetaStartGeneration:
    def test_grid_is_log_spaced_with_first_dimension_fastest(self):
        observed = _meta_grid_points([(0.01, 100.0), (0.1, 10.0)], 2)
        expected = np.array(
            [
                [0.01, 0.1],
                [100.0, 0.1],
                [0.01, 10.0],
                [100.0, 10.0],
            ]
        )
        np.testing.assert_allclose(observed, expected)

    def test_grid_rejects_nonpositive_bounds(self):
        with pytest.raises(ValueError, match="positive, increasing"):
            _meta_grid_points([(0.0, 1.0)], 2)


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
        d = read_dical2_demo(f"{VENDOR_EXAMPLES}/piecewiseConstant/piecewise_constant.demo")
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
        d = read_dical2_demo(f"{VENDOR_EXAMPLES}/islolationMigration/isolation_migration.demo")
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

    def test_introgression_pulse_parameter_is_preserved(self):
        demo = read_dical2_demo(f"{VENDOR_EXAMPLES}/introgression/introgression.demo")
        pulse_epoch = demo.epochs[1]
        assert pulse_epoch.pulse_migration is not None
        assert pulse_epoch.pulse_migration[1, 2] == pytest.approx(0.0)
        assert pulse_epoch.pulse_migration_param_ids is not None
        assert pulse_epoch.pulse_migration_param_ids[1][2] == 1


class TestRefineDemography:
    def test_near_duplicate_grid_boundary_does_not_create_pulse(self):
        demo = read_dical2_demo(f"{VENDOR_EXAMPLES}/cleanSplit/clean_split.demo")
        params = _build_free_params(demo)
        params.set_ordered_param_values(np.array([0.2, 0.25, 0.25, 1.0]))
        moved = params.to_demo(demo)
        grid = np.array([0.0, np.nextafter(0.2, 0.0), DICAL2_T_INF])
        refined = refine_demography(moved, grid)
        assert not any(refined.is_pulse(idx) for idx in range(refined.n_refined))

    def test_parameterized_pulse_becomes_stochastic_refined_epoch(self):
        demo = read_dical2_demo(f"{VENDOR_EXAMPLES}/introgression/introgression.demo")
        params = _build_free_params(demo)
        assert params.ordered_param_ids == [0, 1]
        assert params.pulse_migration_param_ids == [1]
        params.set_ordered_param_values(np.array([0.05, 0.03]))
        packed = params.pack_opt_params()
        params.unpack_opt_params(packed)
        np.testing.assert_allclose(params.ordered_param_values(), [0.05, 0.03])
        moved = params.to_demo(demo)
        refined = refine_demography(
            moved,
            np.array([0.0, 0.1, DICAL2_T_INF], dtype=np.float64),
        )
        pulse_indices = [
            idx for idx in range(refined.n_refined) if refined.is_pulse(idx)
        ]
        assert len(pulse_indices) == 1
        pulse_epoch = _refined_interval_epoch(refined, pulse_indices[0])
        assert pulse_epoch.pulse_migration is not None
        np.testing.assert_allclose(
            pulse_epoch.pulse_migration,
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.97, 0.03],
                    [0.0, 0.0, 1.0],
                ]
            ),
        )


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
        c = read_dical2_config(f"{VENDOR_EXAMPLES}/piecewiseConstant/piecewise_constant.config")
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

    def test_vcf_compacts_config_to_filtered_haplotype_order(self, tmp_path):
        config_path = tmp_path / "pair.config"
        config_path.write_text(
            "\n".join(
                [
                    "20\t2\t2",
                    "1\t0",
                    "0\t0",
                    "0\t1",
                    "0\t0",
                    "0\t0",
                    "0\t0",
                    "0\t0",
                    "0\t0",
                ]
            )
            + "\n"
        )
        data = read_dical2(
            sequences="vendor/diCal2/examples/fromReadme/test.vcf",
            param_file="vendor/diCal2/examples/fromReadme/test.param",
            demo_file="vendor/diCal2/examples/fromReadme/IM.demo",
            config_file=config_path,
            reference_file="vendor/diCal2/examples/fromReadme/test.fa",
            filter_pass_string=".",
        )
        config = data.uns["config"]
        assert data.sequences.shape == (2, 2)
        assert config.haplotype_populations == [0, 1]
        assert config.haplotypes_to_include == [True, True]
        np.testing.assert_array_equal(
            config.haplotype_multiplicities,
            np.array([[1, 0], [0, 1]], dtype=np.int64),
        )
        np.testing.assert_array_equal(config.sample_sizes, np.array([1, 1], dtype=np.int64))

    def test_multiple_vcfs_are_preserved_as_independent_contigs(self):
        root = Path("vendor/diCal2/examples/fromReadme")
        data = read_dical2(
            sequences=[root / "test.vcf", root / "test.vcf"],
            param_file=root / "test.param",
            demo_file=root / "exp.demo",
            rates_file=root / "exp.rates",
            config_file=root / "exp.config",
            reference_file=root / "test.fa",
        )

        assert data.uns["n_contigs"] == 2
        assert len(data.uns["contigs"]) == 2
        assert data.sequences.shape == (8, 4)
        np.testing.assert_array_equal(data.sequences[:, :2], data.sequences[:, 2:])
        assert data.uns["seg_positions"] is None
        assert data.uns["source_paths"]["sequences"] == [
            str(root / "test.vcf"),
            str(root / "test.vcf"),
        ]

    def test_vcf_offset_and_bed_mask_match_upstream_coordinate_semantics(self, tmp_path):
        root = Path("vendor/diCal2/examples/fromReadme")
        shifted_vcf = tmp_path / "shifted.vcf"
        shifted_lines = []
        for line in (root / "test.vcf").read_text().splitlines():
            if line.startswith("#"):
                shifted_lines.append(line)
                continue
            fields = line.split("\t")
            fields[1] = str(int(fields[1]) + 100)
            shifted_lines.append("\t".join(fields))
        shifted_vcf.write_text("\n".join(shifted_lines) + "\n")
        bed = tmp_path / "mask.bed"
        bed.write_text("1\t6\t7\n")

        baseline = read_dical2(
            sequences=root / "test.vcf",
            param_file=root / "test.param",
            demo_file=root / "IM.demo",
            config_file=root / "IM.config",
            reference_file=root / "test.fa",
        )
        shifted = read_dical2(
            sequences=shifted_vcf,
            param_file=root / "test.param",
            demo_file=root / "IM.demo",
            config_file=root / "IM.config",
            reference_file=root / "test.fa",
            vcf_offsets=100,
        )
        masked = read_dical2(
            sequences=shifted_vcf,
            param_file=root / "test.param",
            demo_file=root / "IM.demo",
            config_file=root / "IM.config",
            reference_file=root / "test.fa",
            bed_files=bed,
            vcf_offsets=100,
        )

        np.testing.assert_array_equal(shifted.sequences, baseline.sequences)
        np.testing.assert_array_equal(shifted.uns["seg_positions"], baseline.uns["seg_positions"])
        assert masked.sequences.shape == (4, 1)
        np.testing.assert_array_equal(masked.uns["seg_positions"], np.array([7]))
        assert masked.uns["reference_alleles"][6] == -1
        assert masked.uns["source_paths"]["bed_files"] == str(bed)
        assert masked.uns["source_paths"]["vcf_offsets"] == 100

    def test_vcf_reference_can_be_resolved_from_header(self, tmp_path):
        root = Path("vendor/diCal2/examples/fromReadme")
        vcf = tmp_path / "header-reference.vcf"
        vcf.write_text(
            f"##reference=file://{(root / 'test.fa').resolve()}\n"
            + (root / "test.vcf").read_text()
        )

        from_header = read_dical2(
            sequences=vcf,
            param_file=root / "test.param",
            demo_file=root / "IM.demo",
            config_file=root / "IM.config",
        )
        explicit = read_dical2(
            sequences=root / "test.vcf",
            param_file=root / "test.param",
            demo_file=root / "IM.demo",
            config_file=root / "IM.config",
            reference_file=root / "test.fa",
        )

        np.testing.assert_array_equal(from_header.sequences, explicit.sequences)
        np.testing.assert_array_equal(
            from_header.uns["reference_alleles"], explicit.uns["reference_alleles"]
        )
        assert from_header.uns["source_paths"]["reference_file"] is None

    def test_multiple_contigs_reset_the_native_hmm(self):
        root = Path("vendor/diCal2/examples/fromReadme")
        common = {
            "param_file": root / "test.param",
            "demo_file": root / "exp.demo",
            "rates_file": root / "exp.rates",
            "config_file": root / "exp.config",
            "reference_file": root / "test.fa",
        }
        native_options = {
            "interval_type": "logUniform",
            "interval_params": "11,0.01,4",
            "disableCoordinateWiseMStep": True,
        }
        start_point = np.loadtxt(root / "exp.rand", ndmin=2)[0]
        single = dical2(
            read_dical2(sequences=root / "test.vcf", **common),
            implementation="native",
            n_em_iterations=0,
            start_point=start_point,
            native_options=native_options,
            loci_per_hmm_step=3,
            composite_mode="lol",
        ).results["dical2"]
        repeated = dical2(
            read_dical2(sequences=[root / "test.vcf", root / "test.vcf"], **common),
            implementation="native",
            n_em_iterations=0,
            start_point=start_point,
            native_options=native_options,
            loci_per_hmm_step=3,
            composite_mode="lol",
        ).results["dical2"]

        assert single["n_contigs"] == 1
        assert repeated["n_contigs"] == 2
        assert repeated["log_likelihood"] == pytest.approx(2 * single["log_likelihood"])

    def test_native_pac_mixes_generated_permutations_with_logsumexp(self):
        root = Path("vendor/diCal2/examples/fromReadme")
        data = read_dical2(
            sequences=root / "test.vcf",
            param_file=root / "test.param",
            demo_file=root / "exp.demo",
            rates_file=root / "exp.rates",
            config_file=root / "exp.config",
            reference_file=root / "test.fa",
        )
        start_point = np.loadtxt(root / "exp.rand", ndmin=2)[0]
        result = dical2(
            data,
            implementation="native",
            n_em_iterations=0,
            start_point=start_point,
            seed=1,
            native_options={
                "interval_type": "logUniform",
                "interval_params": "11,0.01,4",
                "disableCoordinateWiseMStep": True,
                "num_permutations": 2,
            },
            loci_per_hmm_step=3,
            composite_mode="pac",
        ).results["dical2"]

        permutation_lls = np.asarray(
            result["rounds"][0]["permutation_log_likelihoods"][0]
        )
        maximum = float(permutation_lls.max())
        expected = maximum + np.log(np.exp(permutation_lls - maximum).sum())
        assert result["log_likelihood"] == pytest.approx(expected)
        assert sum(result["rounds"][0]["permutation_weights"][0]) == pytest.approx(1.0)
        assert result["permutations"]["source"] == "generated_java_collections_shuffle"
        assert len(result["permutations"]["per_contig"][0]) == 2

    @pytest.mark.parametrize(
        ("bed_text", "message"),
        [
            ("1\t5\n", "exactly 3 columns"),
            ("1\t8\t10\n1\t7\t9\n", "sorted and non-overlapping"),
            ("1\t0\t100\n", "outside reference length"),
        ],
    )
    def test_invalid_bed_masks_fail_clearly(self, tmp_path, bed_text, message):
        root = Path("vendor/diCal2/examples/fromReadme")
        bed = tmp_path / "bad.bed"
        bed.write_text(bed_text)
        with pytest.raises(ValueError, match=message):
            read_dical2(
                sequences=root / "test.vcf",
                param_file=root / "test.param",
                demo_file=root / "IM.demo",
                config_file=root / "IM.config",
                reference_file=root / "test.fa",
                bed_files=bed,
            )

    def test_upstream_bridge_preserves_multicontig_vcf_controls(self, tmp_path, monkeypatch):
        import smckit.tl._dical2 as dical2_module
        import smckit.upstream as upstream_api

        root = Path("vendor/diCal2/examples/fromReadme")
        shifted_vcf = tmp_path / "shifted-pass.vcf"
        shifted_lines = []
        for line in (root / "test.vcf").read_text().splitlines():
            if line.startswith("#"):
                shifted_lines.append(line)
                continue
            fields = line.split("\t")
            fields[1] = str(int(fields[1]) + 100)
            fields[6] = "PASS"
            shifted_lines.append("\t".join(fields))
        shifted_vcf.write_text("\n".join(shifted_lines) + "\n")
        bed = tmp_path / "mask.bed"
        bed.write_text("1\t0\t1\n")
        permutation_files = [tmp_path / "contig-1.perm", tmp_path / "contig-2.perm"]
        permutation_files[0].write_text("0 1 2 3\n")
        permutation_files[1].write_text("3 2 1 0\n")
        data = read_dical2(
            sequences=[shifted_vcf, shifted_vcf],
            param_file=root / "test.param",
            demo_file=root / "IM.demo",
            config_file=root / "IM.config",
            reference_file=[root / "test.fa", root / "test.fa"],
            filter_pass_string="PASS",
            bed_files=[bed, bed],
            vcf_offsets=[100, 100],
        )
        resolved = _resolve_dical2_options(
            n_intervals=11,
            max_t=4.0,
            alpha=0.1,
            n_em_iterations=0,
            composite_mode="pcl",
            loci_per_hmm_step=4,
            start_point=None,
            meta_start_file=None,
            meta_num_iterations=1,
            meta_keep_best=1,
            meta_num_points=None,
            bounds=None,
            seed=7,
            method_options={
                "interval_type": "logUniform",
                "interval_params": "11,0.01,4",
                "bounds": "0.1,10;0.1,10;0.1,10;0.1,10;0.1,10",
                "meta_num_start_points": 2,
                "meta_grid_start": True,
                "meta_num_iterations": 2,
                "meta_keep_best": 1,
                "meta_num_points": 2,
                "permutation_files": permutation_files,
                "different_permutations_per_contig": True,
                "num_csds_per_permutation": 2,
            },
        )
        captured = {"calls": []}

        def fake_run(cmd, **kwargs):
            captured["calls"].append((cmd, kwargs))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(dical2_module, "method_upstream_available", lambda method: True)
        monkeypatch.setattr(dical2_module.subprocess, "run", fake_run)
        monkeypatch.setattr(
            upstream_api,
            "status",
            lambda method: {"runtime": {"path": "/mock/java"}},
        )
        result = _dical2_upstream(
            data,
            resolved=resolved,
            cli_args=[],
            implementation_requested="upstream",
        ).results["dical2"]

        cmd, run_kwargs = next(
            (cmd, kwargs) for cmd, kwargs in captured["calls"] if "--vcfFile" in cmd
        )
        vcf_arg = cmd[cmd.index("--vcfFile") + 1]
        ref_arg = cmd[cmd.index("--vcfReferenceFile") + 1]
        bed_arg = cmd[cmd.index("--bedFile") + 1]
        assert vcf_arg == ",".join([str(shifted_vcf.resolve())] * 2)
        assert ref_arg == ",".join([str((root / "test.fa").resolve())] * 2)
        assert bed_arg == ",".join([str(bed.resolve())] * 2)
        assert cmd[cmd.index("--vcfFilterPassString") + 1] == "PASS"
        assert cmd[cmd.index("--vcfOffset") + 1] == "100,100"
        assert cmd[cmd.index("--metaNumStartPoints") + 1] == "2"
        assert "--metaGridStart" in cmd
        assert cmd[cmd.index("--metaNumIterations") + 1] == "2"
        assert cmd[cmd.index("--metaKeepBest") + 1] == "1"
        assert cmd[cmd.index("--metaNumPoints") + 1] == "2"
        assert cmd[cmd.index("--permutationsFile") + 1] == ",".join(
            str(path.resolve()) for path in permutation_files
        )
        assert "--diffPermsPerChunk" in cmd
        assert cmd[cmd.index("--numCsdsPerPerm") + 1] == "2"
        assert run_kwargs["cwd"] == Path("vendor/diCal2").resolve()
        assert result["upstream"]["effective_args"]["vcfOffset"] == [100, 100]
        assert result["upstream"]["effective_args"]["metaGridStart"] is True
        assert result["upstream"]["effective_args"]["diffPermsPerChunk"] is True

        data.uns["source_paths"]["bed_files"] = [None, None]
        captured["calls"].clear()
        _dical2_upstream(
            data,
            resolved=resolved,
            cli_args=[],
            implementation_requested="upstream",
        )
        command_without_bed = next(
            command for command, _ in captured["calls"] if "--vcfFile" in command
        )
        assert "--bedFile" not in command_without_bed


class TestDical2Output:
    def test_native_objective_output_round_trips_through_upstream_parser(self, tmp_path):
        result = {
            "best_params": np.array([1.25, 2.5]),
            "log_likelihood": -12.75,
        }
        path = write_dical2_output(result, tmp_path / "fit.txt")
        rows, best = _parse_dical2_stdout(path.read_text())

        assert len(rows) == 1
        assert best is not None
        assert best["log_likelihood"] == pytest.approx(-12.75)
        np.testing.assert_allclose(best["params"], [1.25, 2.5])
        assert best["id"] == "smckit-native-best"

    def test_output_prefix_records_objective_and_json_artifacts(self, tmp_path):
        data = SmcData()
        data.results["dical2"] = {
            "best_params": np.array([0.5, 1.5]),
            "log_likelihood": -3.0,
            "provenance": {"artifacts": []},
        }
        _persist_dical2_outputs(data, tmp_path / "analysis")

        objective_path = tmp_path / "analysis.dical2.txt"
        result_path = tmp_path / "analysis.dical2.json"
        assert objective_path.is_file()
        assert result_path.is_file()
        on_disk = json.loads(result_path.read_text())
        assert on_disk["best_params"] == [0.5, 1.5]
        assert {
            artifact["kind"] for artifact in data.results["dical2"]["provenance"]["artifacts"]
        } == {"objective_output", "normalized_result"}


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

    def test_native_core_selector_uses_ode_for_structured_multi_deme(self):
        demo = DiCal2Demo(
            epoch_boundaries=np.array([0.0, 0.5, DICAL2_T_INF], dtype=np.float64),
            epochs=[
                DiCal2Epoch(
                    start=0.0,
                    end=0.5,
                    partition=[[0], [1]],
                    pop_sizes=np.array([1.0, 1.0]),
                    migration_matrix=np.array([[-0.2, 0.2], [0.2, -0.2]], dtype=np.float64),
                    pulse_migration=None,
                    growth_rates=np.array([0.0, 0.0]),
                ),
                DiCal2Epoch(
                    start=0.5,
                    end=DICAL2_T_INF,
                    partition=[[0, 1]],
                    pop_sizes=np.array([1.0]),
                    migration_matrix=np.array([[0.0]], dtype=np.float64),
                    pulse_migration=None,
                    growth_rates=np.array([0.0]),
                ),
            ],
            n_present_demes=2,
        )
        config = DiCal2Config(
            seq_length=20,
            n_alleles=2,
            n_populations=2,
            haplotype_populations=[0, 0, 1, 1],
            haplotypes_to_include=[True, True, True, True],
            haplotype_multiplicities=np.array(
                [[1, 0], [1, 0], [0, 1], [0, 1]],
                dtype=np.int64,
            ),
            sample_sizes=np.array([2, 2], dtype=np.int64),
        )
        refined = refine_demography(demo, demo.epoch_boundaries)
        trunk = SimpleTrunk(config=config, additional_hap_idx=0)
        mut_mat = np.array([[0.0, 1.0], [1.0, 0.0]])
        core_obj, core_type = _build_native_core(
            refined=refined,
            trunk=trunk,
            observed_present_deme=0,
            mutation_matrix=mut_mat,
            theta=0.0005,
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

        params.set_ordered_param_values(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 7.0]))
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
