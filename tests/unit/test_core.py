"""Tests for SmcData container."""

import numpy as np

from smckit._core import SmcData


def test_smcdata_empty():
    data = SmcData()
    assert data.n_sites is None
    assert data.params == {}
    assert data.results == {}


def test_smcdata_with_sequences():
    seq = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 0])
    data = SmcData(sequences=seq)
    assert data.n_sites == 10


def test_smcdata_repr():
    data = SmcData(sequences=np.zeros(100), params={"theta": 0.01})
    r = repr(data)
    assert "n_sites=100" in r
    assert "theta" in r


def test_normalized_result_accessors():
    data = SmcData(
        results={
            "psmc": {
                "time_years": np.array([10.0, 20.0]),
                "ne": np.array([1_000.0, 2_000.0]),
                "log_likelihood": -12.5,
            }
        }
    )
    np.testing.assert_array_equal(data.times("psmc"), [10.0, 20.0])
    np.testing.assert_array_equal(
        data.effective_population_size("psmc"),
        [1_000.0, 2_000.0],
    )
    assert data.log_likelihood("psmc") == -12.5
