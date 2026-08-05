import numpy as np
import pytest

from ppsignal._windows import hann

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
    template = np.full(n, 3.5)
    result = hann(template)
    np.testing.assert_allclose(result, EXPECTED[n], rtol=1e-15, atol=1e-15)


def test_hann_ignores_input_values():
    a = hann(np.zeros(5))
    b = hann(np.full(5, -123.456))
    np.testing.assert_allclose(a, b, rtol=1e-15, atol=1e-15)
