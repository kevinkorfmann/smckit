"""MSMC2 validation against upstream MSMC2 output on a fixed fixture."""

from pathlib import Path

import numpy as np

from smckit.io import read_multihetsep
from smckit.tl import msmc2


DATA = Path(__file__).resolve().parents[2] / "data" / "msmc2_test.multihetsep"

EXPECTED_LEFT_ALL_1 = np.array([
    0.0,
    2.12895e-07,
    4.76226e-07,
    8.01941e-07,
    1.20482e-06,
    1.70314e-06,
    2.31951e-06,
    3.08190e-06,
    4.02490e-06,
    5.19130e-06,
    6.63402e-06,
    8.41853e-06,
    1.06258e-05,
    1.33559e-05,
    1.67329e-05,
    2.09098e-05,
    2.60763e-05,
    3.24667e-05,
    4.03710e-05,
    5.01479e-05,
    6.22409e-05,
    7.71987e-05,
    9.57001e-05,
    1.18585e-04,
    1.46890e-04,
    1.81902e-04,
    2.25207e-04,
    2.78772e-04,
    3.45027e-04,
    4.26977e-04,
    5.28342e-04,
    6.53720e-04,
])

EXPECTED_LAMBDA_ALL_1 = np.array([
    3116.97,
    3116.97,
    5807.48,
    7916.58,
    9748.2,
    11165.7,
    12086.4,
    12533.6,
    12710.8,
    12948.5,
    13529.7,
    14592.9,
    16052.1,
    17625.8,
    18764.2,
    19135.1,
    18948.9,
    18433.5,
    17555.3,
    15947.7,
    13408.1,
    10263.3,
    7298.03,
    5075.44,
    3661.35,
    2948.22,
    3038.24,
    6428.18,
    6428.18,
    15810.6,
    15810.6,
    15810.6,
])

EXPECTED_LAMBDA_ALL_2 = np.array([
    766.798,
    766.798,
    2616.2,
    4768.66,
    7093.35,
    9126.87,
    10467.3,
    10981.9,
    10955.7,
    10930.0,
    11348.9,
    12453.6,
    14367.2,
    16852.3,
    18884.1,
    19721.4,
    19890.9,
    19771.0,
    18914.9,
    16231.0,
    11926.9,
    7632.12,
    4633.83,
    2963.26,
    2123.6,
    1768.98,
    1843.12,
    3128.04,
    3128.04,
    221.472,
    221.472,
    221.472,
])

EXPECTED_LEFT_PAIR_1 = np.array([
    0.0,
    5.17595e-07,
    1.13624e-06,
    1.87568e-06,
    2.75948e-06,
    3.81584e-06,
    5.07844e-06,
    6.58755e-06,
    8.39130e-06,
    1.05472e-05,
    1.31240e-05,
    1.62040e-05,
    1.98853e-05,
    2.42852e-05,
    2.95443e-05,
    3.58301e-05,
    4.33432e-05,
    5.23231e-05,
    6.30563e-05,
    7.58850e-05,
    9.12184e-05,
    1.09546e-04,
    1.31451e-04,
    1.57633e-04,
    1.88927e-04,
    2.26331e-04,
    2.71037e-04,
    3.24472e-04,
    3.88339e-04,
    4.64676e-04,
    5.55918e-04,
    6.64973e-04,
])

EXPECTED_LAMBDA_PAIR_1 = np.array([
    700.604,
    700.604,
    1464.28,
    2207.1,
    3070.45,
    4085.93,
    5314.36,
    6861.15,
    8865.7,
    11415.0,
    14371.1,
    17133.6,
    18756.4,
    18928.7,
    18508.7,
    18249.0,
    17922.5,
    16897.3,
    14888.8,
    12177.1,
    9373.77,
    7002.85,
    5265.74,
    4140.81,
    3558.41,
    3598.85,
    4708.26,
    9152.1,
    9152.1,
    15672.7,
    15672.7,
    15672.7,
])

EXPECTED_LEFT_SKIP_1 = np.array([
    0.0,
    1.95839e-07,
    4.38072e-07,
    7.37691e-07,
    1.10829e-06,
    1.56668e-06,
    2.13367e-06,
    2.83498e-06,
    3.70243e-06,
    4.77538e-06,
    6.10252e-06,
    7.74405e-06,
    9.77447e-06,
    1.22859e-05,
    1.53923e-05,
    1.92346e-05,
    2.39871e-05,
    2.98656e-05,
    3.71366e-05,
    4.61301e-05,
    5.72543e-05,
    7.10138e-05,
    8.80329e-05,
    1.09084e-04,
    1.35122e-04,
    1.67328e-04,
    2.07164e-04,
    2.56438e-04,
    3.17384e-04,
    3.92769e-04,
    4.86012e-04,
    6.01345e-04,
])

EXPECTED_LAMBDA_SKIP_1 = np.array([
    1703.17,
    1703.17,
    3213.53,
    4582.45,
    5997.92,
    7391.38,
    8678.74,
    9767.83,
    10597.8,
    11206.8,
    11774.1,
    12601.8,
    13967.3,
    15844.7,
    17751.7,
    18896.9,
    18928.2,
    18511.0,
    18243.7,
    17615.0,
    15829.0,
    12847.9,
    9438.34,
    6528.31,
    4530.8,
    3404.1,
    3128.1,
    5827.17,
    5827.17,
    16466.7,
    16466.7,
    16466.7,
])

EXPECTED_LEFT_QUANT_1 = np.array([
    -0.0,
    2.27523e-07,
    4.60956e-07,
    7.00614e-07,
    9.46840e-07,
    1.20000e-06,
    1.46050e-06,
    1.72878e-06,
    2.00532e-06,
    2.29063e-06,
    2.58530e-06,
    2.88996e-06,
    3.20532e-06,
    3.53214e-06,
    3.87130e-06,
    4.22376e-06,
    4.59062e-06,
    4.97309e-06,
    5.37256e-06,
    5.79062e-06,
    6.22908e-06,
    6.69004e-06,
    7.17592e-06,
    7.68958e-06,
    8.23440e-06,
    8.81438e-06,
    9.43440e-06,
    1.01004e-05,
    1.08197e-05,
    1.16016e-05,
    1.24582e-05,
    1.34050e-05,
    1.44635e-05,
    1.56635e-05,
    1.70488e-05,
    1.86872e-05,
    2.06926e-05,
    2.32779e-05,
    2.69216e-05,
    3.31507e-05,
])

EXPECTED_LAMBDA_QUANT_1 = np.array([
    1444.91,
    3583.2,
    5416.53,
    6975.62,
    8289.52,
    9390.97,
    10307.8,
    11065.7,
    11696.6,
    12233.6,
    12919.0,
    12919.0,
    13721.1,
    13721.1,
    14546.6,
    14546.6,
    15497.9,
    15497.9,
    16590.3,
    16590.3,
    17749.8,
    17749.8,
    18880.7,
    18880.7,
    19920.9,
    19920.9,
    20895.9,
    20895.9,
    21959.1,
    21959.1,
    23341.5,
    23341.5,
    25235.4,
    25235.4,
    27394.3,
    27394.3,
    27750.1,
    27750.1,
    381.639,
    381.639,
])

EXPECTED_LEFT_MULTI_1 = np.array([
    0.0,
    2.14114e-07,
    4.78952e-07,
    8.06530e-07,
    1.21171e-06,
    1.71288e-06,
    2.33278e-06,
    3.09953e-06,
    4.04793e-06,
    5.22101e-06,
    6.67199e-06,
    8.46671e-06,
    1.06866e-05,
    1.34324e-05,
    1.68286e-05,
    2.10295e-05,
    2.62255e-05,
    3.26525e-05,
    4.06021e-05,
    5.04349e-05,
    6.25971e-05,
    7.76405e-05,
    9.62478e-05,
    1.19263e-04,
    1.47731e-04,
    1.82943e-04,
    2.26496e-04,
    2.80368e-04,
    3.47001e-04,
    4.29421e-04,
    5.31365e-04,
    6.57461e-04,
])

EXPECTED_LAMBDA_MULTI_1 = np.array([
    3432.65,
    3432.65,
    6517.28,
    9034.73,
    11214.6,
    12805.7,
    13876.6,
    14727.9,
    15518.2,
    16138.1,
    16591.4,
    17207.0,
    18303.9,
    19698.5,
    20704.9,
    20746.1,
    19927.7,
    18870.6,
    17765.1,
    16246.1,
    13931.0,
    10889.3,
    7845.34,
    5466.1,
    3914.31,
    3109.8,
    3117.83,
    6367.82,
    6367.82,
    16172.9,
    16172.9,
    16172.9,
])


def _assert_matches_reference(res, expected_left, expected_lambda, expected_mu, expected_rho, expected_ll):
    np.testing.assert_allclose(res["left_boundary"], expected_left, rtol=1e-4, atol=1e-10)
    np.testing.assert_allclose(res["lambda"], expected_lambda, rtol=2e-3, atol=1e-6)
    assert np.corrcoef(res["lambda"], expected_lambda)[0, 1] > 0.999999
    assert abs(res["mu"] - expected_mu) < 1e-8
    assert abs(res["rho"] - expected_rho) < 1e-8
    assert abs(res["log_likelihood"] - expected_ll) < 1e-2


def test_msmc2_matches_upstream_single_iteration():
    data = read_multihetsep(DATA)
    out = msmc2(data, n_iterations=1)
    res = out.results["msmc2"]

    _assert_matches_reference(
        res,
        EXPECTED_LEFT_ALL_1,
        EXPECTED_LAMBDA_ALL_1,
        5.392e-05,
        1.29244e-05,
        -2846.78,
    )


def test_msmc2_matches_upstream_two_iterations():
    data = read_multihetsep(DATA)
    out = msmc2(data, n_iterations=2)
    res = out.results["msmc2"]

    _assert_matches_reference(
        res,
        EXPECTED_LEFT_ALL_1,
        EXPECTED_LAMBDA_ALL_2,
        5.392e-05,
        1.16626e-05,
        -2835.84,
    )


def test_msmc2_matches_upstream_pair_indices_run():
    data = read_multihetsep(DATA, pair_indices=[(0, 1), (2, 3)])
    out = msmc2(data, n_iterations=1)
    res = out.results["msmc2"]

    _assert_matches_reference(
        res,
        EXPECTED_LEFT_PAIR_1,
        EXPECTED_LAMBDA_PAIR_1,
        5.30213e-05,
        1.28721e-05,
        -1181.25,
    )


def test_msmc2_matches_upstream_skip_ambiguous_run():
    data = read_multihetsep(DATA, skip_ambiguous=True)
    out = msmc2(data, n_iterations=1)
    res = out.results["msmc2"]

    _assert_matches_reference(
        res,
        EXPECTED_LEFT_SKIP_1,
        EXPECTED_LAMBDA_SKIP_1,
        4.96e-05,
        1.19858e-05,
        -2761.2,
    )


def test_msmc2_matches_upstream_quantile_boundaries_run():
    data = read_multihetsep(DATA)
    out = msmc2(data, n_iterations=1, quantile_bounds=True, time_pattern="10*1+15*2")
    res = out.results["msmc2"]

    _assert_matches_reference(
        res,
        EXPECTED_LEFT_QUANT_1,
        EXPECTED_LAMBDA_QUANT_1,
        5.392e-05,
        1.62775e-05,
        -2893.14,
    )


def test_msmc2_matches_upstream_multi_file_run(tmp_path):
    lines = DATA.read_text().splitlines()
    chr_a = tmp_path / "chrA.multihetsep"
    chr_b = tmp_path / "chrB.multihetsep"
    chr_a.write_text("\n".join(lines[:50]) + "\n")
    chr_b.write_text("\n".join(lines[50:]) + "\n")

    data = read_multihetsep([chr_a, chr_b])
    out = msmc2(data, n_iterations=1)
    res = out.results["msmc2"]

    np.testing.assert_allclose(res["left_boundary"], EXPECTED_LEFT_MULTI_1, rtol=1e-4, atol=1e-10)
    np.testing.assert_allclose(res["lambda"], EXPECTED_LAMBDA_MULTI_1, rtol=3e-3, atol=1e-6)
    assert np.corrcoef(res["lambda"], EXPECTED_LAMBDA_MULTI_1)[0, 1] > 0.999999
    assert abs(res["mu"] - 5.42286e-05) < 1e-8
    assert abs(res["rho"] - 1.32145e-05) < 1e-8
    assert abs(res["log_likelihood"] - (-2894.83)) < 1e-2
