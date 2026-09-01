import shutil
import sys
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest
from postpyc.build import build_file

import ppsignal._windows as windows

cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
pytestmark = pytest.mark.skipif(cc is None, reason="No C compiler available")


def test_compiled_windows_match_interpreted_and_public(tmp_path, monkeypatch):
    name = "ppsignal_windows_native_test"
    extension = build_file(
        Path(windows.__file__),
        ext_module=True,
        module_name=name,
        output=tmp_path / f"{name}{EXTENSION_SUFFIXES[0]}",
    )
    spec = spec_from_file_location(name, extension)
    native = module_from_spec(spec)
    spec.loader.exec_module(native)

    assert isinstance(native.hann, np.ufunc)
    assert native.hann.signature == "(n)->(n)"
    assert isinstance(native.boxcar, np.ufunc)
    assert native.boxcar.signature == "(n)->(n)"
    assert isinstance(native.bartlett, np.ufunc)
    assert native.bartlett.signature == "(n)->(n)"

    for length in [0, 1, 2, 4, 5, 16]:
        shape = np.zeros(length, dtype=np.float64)
        np.testing.assert_allclose(
            native.hann(shape),
            windows.hann(shape),
            rtol=1e-14,
            atol=1e-15,
        )

    for length in [0, 1, 2, 5, 16]:
        shape = np.zeros(length, dtype=np.float64)
        np.testing.assert_array_equal(
            native.boxcar(shape),
            windows.boxcar(shape),
        )

    for length in [0, 1, 2, 4, 5, 16]:
        shape = np.zeros(length, dtype=np.float64)
        np.testing.assert_allclose(
            native.bartlett(shape),
            windows.bartlett(shape),
            rtol=1e-15,
            atol=1e-15,
        )

    monkeypatch.setitem(sys.modules, "ppsignal._windows", native)

    public_name = "ppsignal_windows_public_test"
    public_spec = spec_from_file_location(
        public_name, Path(windows.__file__).with_name("windows.py")
    )
    public = module_from_spec(public_spec)
    public_spec.loader.exec_module(public)

    assert public._hann is native.hann
    assert public._boxcar is native.boxcar
    assert public._bartlett is native.bartlett

    np.testing.assert_allclose(
        public.hann(5),
        [0.0, 0.5, 1.0, 0.5, 0.0],
        rtol=1e-14,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        public.hann(5, sym=False),
        [
            0.0,
            0.3454915028125263,
            0.9045084971874737,
            0.9045084971874737,
            0.3454915028125263,
        ],
        rtol=1e-14,
        atol=1e-15,
    )
    np.testing.assert_array_equal(
        public.boxcar(5),
        [1.0, 1.0, 1.0, 1.0, 1.0],
    )
    np.testing.assert_array_equal(
        public.boxcar(5, sym=False),
        [1.0, 1.0, 1.0, 1.0, 1.0],
    )
    np.testing.assert_allclose(
        public.bartlett(5),
        [0.0, 0.5, 1.0, 0.5, 0.0],
        rtol=1e-15,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        public.bartlett(5, sym=False),
        [0.0, 0.4, 0.8, 0.8, 0.4],
        rtol=1e-15,
        atol=1e-15,
    )
