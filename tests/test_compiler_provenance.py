"""Tests for compiler, SDK, and package provenance capture."""

import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "compiler_provenance.py"

REQUIRED_SECTIONS = {"python", "platform", "compiler", "sdk", "packages"}
SDK_KEYS = {"SDKROOT", "CONDA_BUILD_SYSROOT"}
PACKAGE_KEYS = {"numpy", "postpyc", "postyp"}


def _run_script(args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _load_script_module(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _installed_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def test_compiler_provenance_writes_schema_v1_json_with_required_sections(tmp_path):
    output_path = tmp_path / "provenance.json"
    result = _run_script(["--output", str(output_path)])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr
    assert output_path.exists()

    data = json.loads(output_path.read_text())

    assert data["schema_version"] == 1
    assert REQUIRED_SECTIONS <= data.keys()

    python_info = data["python"]
    assert python_info["implementation"] == platform.python_implementation()
    assert python_info["version"] == platform.python_version()
    assert python_info["executable"] == sys.executable

    platform_info = data["platform"]
    assert platform_info["system"] == platform.system()
    assert platform_info["release"] == platform.release()
    assert platform_info["machine"] == platform.machine()

    compiler_info = data["compiler"]
    assert set(compiler_info.keys()) >= {"command", "resolved_path", "version"}
    assert isinstance(compiler_info["command"], str) and compiler_info["command"]
    assert compiler_info["resolved_path"] is None or isinstance(
        compiler_info["resolved_path"], str
    )
    assert compiler_info["version"] is None or isinstance(compiler_info["version"], str)

    sdk_info = data["sdk"]
    assert set(sdk_info.keys()) == SDK_KEYS
    assert sdk_info["SDKROOT"] == os.environ.get("SDKROOT")
    assert sdk_info["CONDA_BUILD_SYSROOT"] == os.environ.get("CONDA_BUILD_SYSROOT")

    packages_info = data["packages"]
    assert set(packages_info.keys()) == PACKAGE_KEYS
    for package_name in PACKAGE_KEYS:
        assert packages_info[package_name] == _installed_version(package_name)


def test_compiler_provenance_reports_none_for_missing_compiler_and_packages(
    monkeypatch, capsys, tmp_path
):
    module = _load_script_module("compiler_provenance_missing")

    monkeypatch.setattr(module.shutil, "which", lambda command: None)

    def _missing_version(name):
        raise module.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(module.importlib.metadata, "version", _missing_version)

    output_path = tmp_path / "provenance.json"
    monkeypatch.setattr(sys, "argv", ["compiler_provenance.py", "--output", str(output_path)])

    assert module.main() == 0

    captured = capsys.readouterr()
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err

    data = json.loads(output_path.read_text())
    assert isinstance(data["compiler"]["command"], str) and data["compiler"]["command"]
    assert data["compiler"]["resolved_path"] is None
    assert data["compiler"]["version"] is None
    for package_name in PACKAGE_KEYS:
        assert data["packages"][package_name] is None


def test_compiler_provenance_creates_missing_parent_directories(tmp_path):
    output_path = tmp_path / "nested" / "dir" / "provenance.json"
    result = _run_script(["--output", str(output_path)])

    assert result.returncode == 0, result.stdout + result.stderr
    assert output_path.exists()


# Compiler command contract: wrappers and flags remain shell-free, and failed
# version commands never report their error output as a compiler version.


def test_compiler_provenance_resolves_first_token_of_ccache_wrapped_cc(monkeypatch):
    module = _load_script_module("compiler_provenance_ccache_wrapper_cc")

    monkeypatch.setenv("CC", "ccache clang")

    which_calls = []

    def _fake_which(command):
        which_calls.append(command)
        return "/usr/bin/ccache" if command == "ccache" else None

    monkeypatch.setattr(module.shutil, "which", _fake_which)

    run_calls = []

    def _fake_run(cmd, **kwargs):
        run_calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="Apple clang version 15.0.0\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    info = module._compiler_info()

    assert which_calls == ["ccache"], "expected only the first (executable) token to be resolved"
    assert len(run_calls) == 1
    cmd, kwargs = run_calls[0]
    assert cmd == ["/usr/bin/ccache", "clang", "--version"], (
        "expected remaining tokens preserved and `--version` appended"
    )
    assert not kwargs.get("shell", False), "subprocess call must remain shell-free"
    assert info["command"] == "ccache clang", "original CC string must remain recorded"
    assert info["resolved_path"] == "/usr/bin/ccache"


def test_compiler_provenance_resolves_first_token_of_cc_with_leading_flags(monkeypatch):
    module = _load_script_module("compiler_provenance_flagged_cc")

    monkeypatch.setenv("CC", "clang -arch arm64")

    which_calls = []

    def _fake_which(command):
        which_calls.append(command)
        return "/usr/bin/clang" if command == "clang" else None

    monkeypatch.setattr(module.shutil, "which", _fake_which)

    run_calls = []

    def _fake_run(cmd, **kwargs):
        run_calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="clang version 15.0.0\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    info = module._compiler_info()

    assert which_calls == ["clang"], "expected only the first (executable) token to be resolved"
    assert len(run_calls) == 1
    cmd, kwargs = run_calls[0]
    assert cmd == ["/usr/bin/clang", "-arch", "arm64", "--version"], (
        "expected remaining tokens preserved and `--version` appended"
    )
    assert not kwargs.get("shell", False), "subprocess call must remain shell-free"
    assert info["command"] == "clang -arch arm64", "original CC string must remain recorded"
    assert info["resolved_path"] == "/usr/bin/clang"


def test_module_description_identifies_diagnostic_build_metadata_not_reproducible_record():
    module = _load_script_module("compiler_provenance_description")
    description = (module.__doc__ or "").lower()

    assert "diagnostic build metadata" in description, (
        "expected the module description to identify the output as diagnostic build metadata"
    )
    assert "reproducible build record" not in description, (
        "the module description must not call the output a reproducible build record"
    )
    assert "attestation" not in description, (
        "the module description must not call the output an attestation"
    )


def test_compiler_provenance_nonzero_returncode_yields_null_version_despite_output(monkeypatch):
    module = _load_script_module("compiler_provenance_nonzero_returncode")

    monkeypatch.setenv("CC", "cc")
    monkeypatch.setattr(module.shutil, "which", lambda command: "/usr/bin/cc")

    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 1, stdout="some stdout text\n", stderr="some stderr text\n"
        )

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    info = module._compiler_info()

    assert info["version"] is None, (
        "a nonzero returncode must yield a null version even when stdout/stderr is non-empty"
    )
