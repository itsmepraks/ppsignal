from operator import index

import numpy as np

try:
    from ppsignal_native import boxcar as _boxcar
    from ppsignal_native import hann as _hann
except ModuleNotFoundError as exc:
    if exc.name != "ppsignal_native":
        raise
    from ppsignal._windows import boxcar as _boxcar
    from ppsignal._windows import hann as _hann


def boxcar(M, sym=True):
    """Return a Boxcar window."""
    length = index(M)
    if length < 0:
        raise ValueError("M must be non-negative")
    return _boxcar(np.empty(length, dtype=np.float64))


def hann(M, sym=True):
    """Return a Hann window.

    Parameters
    ----------
    M : int
        Number of values in the returned window. M must be non-negative.
    sym : bool, default: True
        If True, return a symmetric window for filter design.
        If False, return a periodic window for spectral analysis.

    Returns
    -------
    numpy.ndarray
        One-dimensional Hann window with the numpy.float64 data type.

    Raises
    ------
    ValueError
        M is negative.
    TypeError
        M is not an integer.
    """
    length = index(M)
    if length < 0:
        raise ValueError("M must be non-negative")
    if not sym and length > 1:
        return _hann(np.empty(length + 1, dtype=np.float64))[:-1]
    return _hann(np.empty(length, dtype=np.float64))
