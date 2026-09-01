import numpy as np
import pytest

from ppsignal._windows import boxcar as boxcar_kernel
from ppsignal._windows import hann as hann_kernel
from ppsignal.windows import bartlett, boxcar, hann

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


def test_boxcar_ignores_input_values():
    a = boxcar_kernel(np.zeros(5))
    b = boxcar_kernel(np.full(5, -123.456))
    np.testing.assert_array_equal(a, np.ones(5))
    np.testing.assert_array_equal(a, b)


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


BARTLETT_EXPECTED = {
    0: [],
    1: [1.0],
    2: [0.0, 0.0],
    3: [0.0, 1.0, 0.0],
    4: [0.0, 2.0 / 3.0, 2.0 / 3.0, 0.0],
    5: [0.0, 0.5, 1.0, 0.5, 0.0],
}

BARTLETT_PERIODIC_EXPECTED = {
    0: [],
    1: [1.0],
    2: [0.0, 1.0],
    3: [0.0, 2.0 / 3.0, 2.0 / 3.0],
    4: [0.0, 0.5, 1.0, 0.5],
    5: [0.0, 0.4, 0.8, 0.8, 0.4],
}


@pytest.mark.parametrize("n", sorted(BARTLETT_EXPECTED))
def test_bartlett_symmetric_lengths(n):
    result = bartlett(n)
    assert result.shape == (n,)
    assert result.dtype == np.float64
    np.testing.assert_allclose(
        result, BARTLETT_EXPECTED[n], rtol=1e-15, atol=1e-15
    )


@pytest.mark.parametrize("n", sorted(BARTLETT_PERIODIC_EXPECTED))
def test_bartlett_periodic_lengths(n):
    result = bartlett(n, sym=False)
    assert result.shape == (n,)
    assert result.dtype == np.float64
    np.testing.assert_allclose(
        result, BARTLETT_PERIODIC_EXPECTED[n], rtol=1e-15, atol=1e-15
    )


def test_bartlett_negative_length_raises_value_error():
    with pytest.raises(ValueError):
        bartlett(-1)


def test_bartlett_non_integer_length_raises_type_error():
    with pytest.raises(TypeError):
        bartlett(2.5)


def test_bartlett_accepts_numpy_integer_length():
    result = bartlett(np.int64(3))
    assert result.dtype == np.float64
    np.testing.assert_allclose(
        result, BARTLETT_EXPECTED[3], rtol=1e-15, atol=1e-15
    )


def test_bartlett_five_symmetric_has_maximum_exactly_one():
    result = bartlett(5)
    assert result.max() == 1.0


def test_bartlett_four_symmetric_maximum_is_less_than_one():
    result = bartlett(4)
    assert result.max() < 1.0


def test_bartlett_four_periodic_has_maximum_exactly_one():
    result = bartlett(4, sym=False)
    assert result.max() == 1.0


def test_bartlett_sixteen_symmetric_is_symmetric():
    result = bartlett(16)
    np.testing.assert_allclose(result, result[::-1], rtol=1e-15, atol=1e-15)


WINDOW_FUNCTIONS = {"bartlett": bartlett, "boxcar": boxcar, "hann": hann}


@pytest.mark.parametrize("n", [0, 1, 2, 5, 16])
@pytest.mark.parametrize("name", sorted(WINDOW_FUNCTIONS))
def test_window_functions_share_common_invariants(name, n):
    window = WINDOW_FUNCTIONS[name]
    result = window(n)
    assert len(result) == n
    assert result.ndim == 1
    assert result.dtype == np.float64
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
    other = window(n)
    assert result is not other
