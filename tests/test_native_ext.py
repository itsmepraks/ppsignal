from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from postpyc.build import build_file

import ppsignal._windows as windows


def test_compiled_hann_matches_interpreted(tmp_path):
    name = "ppsignal_windows_native_test"
    extension = build_file(
        Path(windows.__file__),
        ext_module=True,
        module_name=name,
        output=tmp_path / f"{name}.so",
    )
    spec = spec_from_file_location(name, extension)
    native = module_from_spec(spec)
    spec.loader.exec_module(native)

    assert isinstance(native.hann, np.ufunc)
    assert native.hann.signature == "(n)->(n)"

    for length in [0, 1, 2, 4, 5, 16]:
        shape = np.zeros(length, dtype=np.float64)
        np.testing.assert_allclose(
            native.hann(shape),
            windows.hann(shape),
            rtol=1e-14,
            atol=1e-15,
        )
