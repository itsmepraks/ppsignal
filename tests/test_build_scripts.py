"""Tests for the ahead-of-time build scripts (scripts/build_native.py, scripts/build_ext.py)."""

import ctypes
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
BUILD_NATIVE = ROOT / "scripts" / "build_native.py"
BUILD_EXT = ROOT / "scripts" / "build_ext.py"

MODULE_NAME = "ppsignal_probe_native"


def _has_c_compiler() -> bool:
    return any(shutil.which(cc) for cc in ("cc", "clang", "gcc"))


def _run_script(script: Path, build_dir) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PPSIGNAL_BUILD_DIR"] = str(build_dir)
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _POSTArray(ctypes.Structure):
    """Minimal ctypes mirror of the compiled `__pp_array` view struct."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("ndim", ctypes.c_int64),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("offset_bytes", ctypes.c_int64),
    ]


def _call_pp_identity(library_path, values):
    lib = ctypes.CDLL(str(library_path))
    pp_identity = lib.pp_identity
    pp_identity.argtypes = [
        ctypes.POINTER(_POSTArray),
        ctypes.POINTER(_POSTArray),
        ctypes.c_int64,
    ]
    pp_identity.restype = None

    length = len(values)
    in_buf = (ctypes.c_double * length)(*values)
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

    in_view = _view(in_buf)
    out_view = _view(out_buf)
    pp_identity(ctypes.byref(in_view), ctypes.byref(out_view), ctypes.c_int64(length))
    return list(out_buf)


@pytest.mark.skipif(not _has_c_compiler(), reason="no C compiler available locally")
def test_build_native_produces_library_header_and_manifest(tmp_path):
    result = _run_script(BUILD_NATIVE, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr

    so_path = tmp_path / "ppsignal_probe.so"
    header_path = tmp_path / "ppsignal_probe.h"
    manifest_path = tmp_path / "ppsignal_probe.json"
    assert so_path.exists(), "missing ppsignal_probe.so"
    assert header_path.exists(), "missing ppsignal_probe.h"
    assert manifest_path.exists(), "missing ppsignal_probe.json"

    manifest = json.loads(manifest_path.read_text())
    assert manifest["post_abi"] == 1

    exports = manifest["exports"]
    identity_export = next((e for e in exports if e["name"] == "identity"), None)
    assert identity_export is not None, f"no 'identity' export in manifest exports: {exports!r}"
    assert identity_export["c_symbol"] == "pp_identity"
    assert identity_export["kind"] == "ufunc"
    assert identity_export["ufunc"]["signature"] == "(n)->(n)"

    header_text = header_path.read_text()
    assert "pp_identity" in header_text

    values = [-2.5, 0.0, 3.25, 10.0]
    assert _call_pp_identity(so_path, values) == values

    assert _call_pp_identity(so_path, []) == []


@pytest.mark.skipif(not _has_c_compiler(), reason="no C compiler available locally")
def test_build_ext_produces_ufunc_extension(tmp_path):
    result = _run_script(BUILD_EXT, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr

    suffix = EXTENSION_SUFFIXES[0]
    ext_path = tmp_path / f"{MODULE_NAME}{suffix}"
    assert ext_path.exists(), f"missing built extension at {ext_path}"

    spec = importlib.util.spec_from_file_location(MODULE_NAME, ext_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    identity = module.identity
    assert isinstance(identity, np.ufunc)
    assert identity.signature == "(n)->(n)"

    values = np.array([-2.5, 0.0, 3.25, 10.0], dtype=np.float64)
    result_arr = identity(values)
    np.testing.assert_array_equal(result_arr, values)
    assert result_arr is not values
    assert not np.shares_memory(result_arr, values)

    empty = np.array([], dtype=np.float64)
    result_empty = identity(empty)
    np.testing.assert_array_equal(result_empty, empty)
    assert not np.shares_memory(result_empty, empty)


@pytest.mark.parametrize("script", [BUILD_NATIVE, BUILD_EXT], ids=["build_native", "build_ext"])
def test_build_script_fails_cleanly_when_build_dir_is_a_file(tmp_path, script):
    bad_build_dir = tmp_path / "not_a_directory"
    bad_build_dir.write_text("occupied")

    result = _run_script(script, bad_build_dir)

    assert result.returncode == 1
    assert "FAILED" in result.stdout
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr
    assert len(result.stdout.splitlines()) <= 3


@pytest.mark.parametrize(
    "script,module_name",
    [(BUILD_NATIVE, "build_native_oserror"), (BUILD_EXT, "build_ext_oserror")],
    ids=["build_native", "build_ext"],
)
def test_build_script_reports_failure_when_mkdtemp_raises(
    monkeypatch, capsys, script, module_name
):
    module = _load_script_module(script, module_name)
    monkeypatch.delenv("PPSIGNAL_BUILD_DIR", raising=False)

    def _boom(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(module.tempfile, "mkdtemp", _boom)

    assert module.main() == 1
    captured = capsys.readouterr()
    assert "FAILED" in captured.out
    assert "no space left on device" in captured.out
    assert "Traceback" not in captured.out


@pytest.fixture
def build_ext_module():
    return _load_script_module(BUILD_EXT, "build_ext_validate_identity")


def test_validate_identity_accepts_correct_identity(build_ext_module):
    identity = lambda values: np.array(values, dtype=np.float64)  # noqa: E731
    assert build_ext_module._validate_identity(identity) is None


def test_validate_identity_rejects_wrong_dtype(build_ext_module):
    identity = lambda values: values.astype(np.float32)  # noqa: E731
    error = build_ext_module._validate_identity(identity)
    assert error is not None
    assert "dtype" in error


def test_validate_identity_rejects_wrong_values(build_ext_module):
    identity = lambda values: values + 1.0  # noqa: E731
    error = build_ext_module._validate_identity(identity)
    assert error is not None


def test_validate_identity_rejects_shared_memory(build_ext_module):
    identity = lambda values: values  # noqa: E731
    error = build_ext_module._validate_identity(identity)
    assert error is not None
    assert "shares memory" in error


# --- Finding 1: stale fixed-name artifacts in a reused PPSIGNAL_BUILD_DIR ---


def test_build_native_leaves_no_stale_artifact_when_build_fails(monkeypatch, capsys, tmp_path):
    module = _load_script_module(BUILD_NATIVE, "build_native_stale_fail")
    monkeypatch.setenv("PPSIGNAL_BUILD_DIR", str(tmp_path))

    so_path = tmp_path / "ppsignal_probe.so"
    header_path = tmp_path / "ppsignal_probe.h"
    manifest_path = tmp_path / "ppsignal_probe.json"
    for sentinel in (so_path, header_path, manifest_path):
        sentinel.write_text("stale-sentinel")

    def _boom(*args, **kwargs):
        raise module.BuildError("compiler exploded")

    monkeypatch.setattr(module, "build_file", _boom)

    assert module.main() == 1
    captured = capsys.readouterr()
    assert "FAILED" in captured.out
    assert "Traceback" not in captured.out
    assert not so_path.exists()
    assert not header_path.exists()
    assert not manifest_path.exists()


def test_build_native_requires_all_expected_artifacts_after_build(monkeypatch, capsys, tmp_path):
    module = _load_script_module(BUILD_NATIVE, "build_native_incomplete")
    monkeypatch.setenv("PPSIGNAL_BUILD_DIR", str(tmp_path))

    def _fake_build_file(source, output, emit_header=False, emit_manifest=False):
        Path(output).write_text("so-only")  # header/manifest intentionally not written

    monkeypatch.setattr(module, "build_file", _fake_build_file)

    assert module.main() == 1
    captured = capsys.readouterr()
    assert "FAILED" in captured.out
    assert "Traceback" not in captured.out


def test_build_ext_leaves_no_stale_artifact_when_build_fails(monkeypatch, capsys, tmp_path):
    module = _load_script_module(BUILD_EXT, "build_ext_stale_fail")
    monkeypatch.setenv("PPSIGNAL_BUILD_DIR", str(tmp_path))

    ext_path = tmp_path / f"{MODULE_NAME}{EXTENSION_SUFFIXES[0]}"
    ext_path.write_text("stale-sentinel")

    def _boom(*args, **kwargs):
        raise module.BuildError("compiler exploded")

    monkeypatch.setattr(module, "build_file", _boom)

    assert module.main() == 1
    captured = capsys.readouterr()
    assert "FAILED" in captured.out
    assert "Traceback" not in captured.out
    assert not ext_path.exists()


def test_build_ext_does_not_attempt_to_load_missing_extension_file(monkeypatch, capsys, tmp_path):
    module = _load_script_module(BUILD_EXT, "build_ext_missing_output")
    monkeypatch.setenv("PPSIGNAL_BUILD_DIR", str(tmp_path))

    def _fake_build_file(*args, **kwargs):
        return None  # "succeeds" without producing the expected extension file

    monkeypatch.setattr(module, "build_file", _fake_build_file)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("must not attempt to load a missing extension artifact")

    monkeypatch.setattr(module.importlib.util, "spec_from_file_location", _fail_if_called)

    assert module.main() == 1
    captured = capsys.readouterr()
    assert "FAILED" in captured.out
    assert "Traceback" not in captured.out


# --- Finding 2: non-ImportError/OSError/AttributeError exceptions during extension load ---


def test_build_ext_validation_runtime_error_reported_cleanly(monkeypatch, capsys, tmp_path):
    module = _load_script_module(BUILD_EXT, "build_ext_validation_raises")
    monkeypatch.setenv("PPSIGNAL_BUILD_DIR", str(tmp_path))

    ext_path = tmp_path / f"{MODULE_NAME}{EXTENSION_SUFFIXES[0]}"

    def _fake_build_file(*args, **kwargs):
        ext_path.write_bytes(b"")

    monkeypatch.setattr(module, "build_file", _fake_build_file)

    class _FakeLoader:
        def exec_module(self, mod):
            mod.identity = np.negative  # a real np.ufunc, passes the isinstance check

    class _FakeSpec:
        loader = _FakeLoader()

    monkeypatch.setattr(
        module.importlib.util, "spec_from_file_location", lambda *a, **k: _FakeSpec()
    )
    monkeypatch.setattr(
        module.importlib.util, "module_from_spec", lambda spec: types.ModuleType("fake")
    )

    def _boom(identity):
        raise RuntimeError("boom from validation")

    monkeypatch.setattr(module, "_validate_identity", _boom)

    assert module.main() == 1
    captured = capsys.readouterr()
    assert "FAILED" in captured.out
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err
