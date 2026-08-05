from operator import index

import numpy as np

from ppsignal._windows import hann as _hann


def hann(M, sym=True):
    length = index(M)
    if length < 0:
        raise ValueError("M must be non-negative")
    if not sym:
        raise NotImplementedError("periodic Hann is not implemented")
    return _hann(np.empty(length, dtype=np.float64))
