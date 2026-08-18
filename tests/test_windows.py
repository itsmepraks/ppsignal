import numpy as np
import pytest

from ppsignal._windows import hann as hann_kernel
from ppsignal.windows import boxcar, hann

EXPECTED = {
    0: [],
    1: [1.0],
    2: [0.0, 0.0],
    3: [0.0, 1.0, 0.0],
    4: [0.0, 0.75, 0.75, 0.0],
    5: [0.0, 0.5, 1.0, 0.5, 0.0],
}

PERIODIC_EXPECTED = {
    0: [],
    1: [1.0],
    2: [0.0, 1.0],
    3: [0.0, 0.75, 0.75],
    4: [0.0, 0.5, 1.0, 0.5],
    5: [
        0.0,
        0.3454915028125263,
        0.9045084971874737,
        0.9045084971874737,
        0.3454915028125263,
    ],
}


@pytest.mark.parametrize("n", sorted(EXPECTED))
def test_hann_symmetric_lengths(n):
    result = hann(n)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, EXPECTED[n], rtol=1e-15, atol=1e-15)


def test_hann_ignores_input_values():
    a = hann_kernel(np.zeros(5))
    b = hann_kernel(np.full(5, -123.456))
    np.testing.assert_allclose(a, b, rtol=1e-15, atol=1e-15)


def test_hann_negative_length_raises_value_error():
    with pytest.raises(ValueError):
        hann(-1)


def test_hann_non_integer_length_raises_type_error():
    with pytest.raises(TypeError):
        hann(2.5)


@pytest.mark.parametrize("n", sorted(PERIODIC_EXPECTED))
def test_hann_periodic_lengths(n):
    result = hann(n, sym=False)
    assert len(result) == n
    assert result.dtype == np.float64
    np.testing.assert_allclose(
        result, PERIODIC_EXPECTED[n], rtol=1e-15, atol=1e-15
    )


@pytest.mark.parametrize("n", [0, 1, 2, 5])
def test_boxcar_lengths(n):
    result = boxcar(n)
    assert result.shape == (n,)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, np.ones(n, dtype=np.float64))


def test_boxcar_symmetric_and_periodic_are_equal_for_length_five():
    np.testing.assert_array_equal(boxcar(5, sym=True), boxcar(5, sym=False))


def test_boxcar_separate_calls_return_different_array_objects():
    a = boxcar(5)
    b = boxcar(5)
    assert a is not b


def test_boxcar_negative_length_raises_value_error():
    with pytest.raises(ValueError):
        boxcar(-1)


def test_boxcar_non_integer_length_raises_type_error():
    with pytest.raises(TypeError):
        boxcar(2.5)


def test_boxcar_accepts_numpy_integer_length():
    result = boxcar(np.int64(3))
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, np.ones(3, dtype=np.float64))
