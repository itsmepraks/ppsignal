import numpy as np
import pytest

from ppsignal._windows import hann

HANN_REFERENCE = [
    [],
    [1.0],
    [0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.75, 0.75, 0.0],
    [0.0, 0.5, 1.0, 0.5, 0.0],
    [
        0.0,
        0.3454915028125263,
        0.9045084971874737,
        0.9045084971874737,
        0.3454915028125263,
        0.0,
    ],
]


@pytest.mark.parametrize("length", range(len(HANN_REFERENCE)))
def test_hann_kernel_matches_symmetric_reference(length):
    result = hann(np.empty(length, dtype=np.float64))
    expected = np.array(HANN_REFERENCE[length], dtype=np.float64)
    assert result.dtype == np.float64
    assert result.shape == (length,)
    np.testing.assert_allclose(result, expected, rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize("fill", [-7.5, np.nan])
def test_hann_kernel_ignores_input_values(fill):
    shape = np.full(5, fill, dtype=np.float64)
    result = hann(shape)
    np.testing.assert_allclose(result, HANN_REFERENCE[5], rtol=1e-15, atol=1e-15)


def test_hann_kernel_preserves_input_and_does_not_share_memory():
    carrier = np.array([np.nan, -7.5, 0.0, np.nan, 1e300], dtype=np.float64)
    original = carrier.copy()
    result = hann(carrier)
    assert not np.shares_memory(result, carrier)
    # assert_array_equal treats NaNs in matching positions as equal.
    np.testing.assert_array_equal(carrier, original)


@pytest.mark.parametrize(
    "carrier",
    [
        np.zeros((2, 0), dtype=np.float64),
        np.array([[np.nan], [-2.5]], dtype=np.float64),
        np.array([[np.nan, -np.inf, 3.25], [np.inf, -0.0, 1e300]], dtype=np.float64),
        np.full((2, 3, 4), np.nan, dtype=np.float64),
    ],
    ids=["length-0", "length-1", "distinct-rows", "multi-leading-dims"],
)
def test_hann_kernel_broadcasts_over_batched_input(carrier):
    result = hann(carrier)
    expected = np.broadcast_to(HANN_REFERENCE[carrier.shape[-1]], carrier.shape)
    assert result.dtype == np.float64
    assert result.shape == carrier.shape
    np.testing.assert_allclose(result, expected, rtol=1e-15, atol=1e-15)
