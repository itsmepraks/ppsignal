import numpy as np
import pytest

from ppsignal._windows import hann as hann_kernel
from ppsignal.windows import hann

EXPECTED = {
    0: [],
    1: [1.0],
    2: [0.0, 0.0],
    3: [0.0, 1.0, 0.0],
    4: [0.0, 0.75, 0.75, 0.0],
    5: [0.0, 0.5, 1.0, 0.5, 0.0],
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


def test_hann_sym_false_raises_not_implemented_error():
    with pytest.raises(NotImplementedError):
        hann(5, sym=False)
