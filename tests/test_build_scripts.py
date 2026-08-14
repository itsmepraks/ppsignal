import ctypes
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
requires_compiler = pytest.mark.skipif(CC is None, reason="No C compiler available")


class _POSTArray(ctypes.Structure):
    """Minimal ctypes mirror of the compiled `__pp_array` view struct."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("ndim", ctypes.c_int64),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("offset_bytes", ctypes.c_int64),
    ]


def _call_pp_hann(library_path, length):
    lib = ctypes.CDLL(str(library_path))
    pp_hann = lib.pp_hann
    pp_hann.argtypes = [
        ctypes.POINTER(_POSTArray),
        ctypes.POINTER(_POSTArray),
        ctypes.c_int64,
    ]
    pp_hann.restype = None

    shape_buf = (ctypes.c_double * length)()
    out_buf = (ctypes.c_double * length)()
    dims = (ctypes.c_int64 * 1)(length)
    strides = (ctypes.c_int64 * 1)(ctypes.sizeof(ctypes.c_double))

    def _view(buf):
        return _POSTArray(
            data=ctypes.cast(buf, ctypes.c_void_p),
            ndim=1,
            shape=dims,
            strides=strides,
            offset_bytes=0,
        )

    shape_view = _view(shape_buf)
    out_view = _view(out_buf)
    pp_hann(ctypes.byref(shape_view), ctypes.byref(out_view), ctypes.c_int64(length))
    return list(out_buf)


@pytest.mark.parametrize("script", ["scripts/build_native.py", "scripts/build_ext.py"])
def test_build_scripts_fail_cleanly_when_build_dir_is_a_file(tmp_path, script):
    bad_dir = tmp_path / "not_a_directory"
    bad_dir.write_text("not a directory")
    env = os.environ.copy()
    env["PPSIGNAL_BUILD_DIR"] = str(bad_dir)
    result = subprocess.run(
        [sys.executable, script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "FAILED" in result.stdout
    assert len(result.stdout.strip().splitlines()) <= 3
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr


@requires_compiler
def test_build_native_emits_library_header_and_manifest(tmp_path):
    env = os.environ.copy()
    env["PPSIGNAL_BUILD_DIR"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "scripts/build_native.py"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "hann" in result.stdout
    assert (tmp_path / "ppsignal_windows.so").is_file()
    assert (tmp_path / "ppsignal_windows.h").is_file()
    assert (tmp_path / "ppsignal_windows.json").is_file()

    manifest = json.loads((tmp_path / "ppsignal_windows.json").read_text())
    assert manifest["post_abi"] == 1
    exports = {export["name"]: export for export in manifest["exports"]}
    hann_export = exports["hann"]
    assert hann_export["c_symbol"] == "pp_hann"
    assert hann_export["kind"] == "ufunc"
    assert hann_export["ufunc"]["signature"] == "(n)->(n)"

    header = (tmp_path / "ppsignal_windows.h").read_text()
    assert "pp_hann" in header

    hann_values = _call_pp_hann(tmp_path / "ppsignal_windows.so", length=5)
    np.testing.assert_allclose(
        hann_values,
        [0.0, 0.5, 1.0, 0.5, 0.0],
        rtol=1e-14,
        atol=1e-15,
    )


@requires_compiler
def test_build_ext_emits_importable_hann_ufunc(tmp_path):
    env = os.environ.copy()
    env["PPSIGNAL_BUILD_DIR"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "scripts/build_ext.py"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "registered Hann ufunc" in result.stdout
    artifact = tmp_path / f"ppsignal_native{EXTENSION_SUFFIXES[0]}"
    assert artifact.is_file()

    spec = importlib.util.spec_from_file_location("ppsignal_native", artifact)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert isinstance(module.hann, np.ufunc)
    assert module.hann.signature == "(n)->(n)"
    np.testing.assert_allclose(
        module.hann(np.zeros(5, dtype=np.float64)),
        [0.0, 0.5, 1.0, 0.5, 0.0],
        rtol=1e-14,
        atol=1e-15,
    )
