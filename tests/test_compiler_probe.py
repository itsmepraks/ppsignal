import numpy as np

from ppsignal._compiler_probe import identity


def test_identity_probe_copies_float64_values():
    values = np.array([-2.5, 0.0, 3.25], dtype=np.float64)
    result = identity(values)
    np.testing.assert_array_equal(result, values)
    assert result.dtype == np.float64
    assert result is not values
    assert not np.shares_memory(result, values)


def test_identity_probe_handles_empty_float64_input():
    values = np.array([], dtype=np.float64)
    result = identity(values)
    np.testing.assert_array_equal(result, values)
    assert result.dtype == np.float64
    assert not np.shares_memory(result, values)
