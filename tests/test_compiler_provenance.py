"""Tests for compiler, SDK, and package provenance capture."""

import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "compiler_provenance.py"

REQUIRED_SECTIONS = {"python", "platform", "compiler", "linker", "sdk", "packages"}
SDK_KEYS = {"SDKROOT", "CONDA_BUILD_SYSROOT", "selected_path", "version"}
LINKER_KEYS = {"command", "resolved_path", "version"}
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

    linker_info = data["linker"]
    assert set(linker_info.keys()) == LINKER_KEYS
    assert isinstance(linker_info["command"], str) and linker_info["command"]
    assert linker_info["resolved_path"] is None or isinstance(linker_info["resolved_path"], str)
    assert linker_info["version"] is None or isinstance(linker_info["version"], str)

    sdk_info = data["sdk"]
    assert set(sdk_info.keys()) == SDK_KEYS
    assert sdk_info["SDKROOT"] == os.environ.get("SDKROOT")
    assert sdk_info["CONDA_BUILD_SYSROOT"] == os.environ.get("CONDA_BUILD_SYSROOT")
    expected_selected_path = os.environ.get("CONDA_BUILD_SYSROOT") or os.environ.get("SDKROOT")
    assert sdk_info["selected_path"] == expected_selected_path
    assert sdk_info["version"] is None or isinstance(sdk_info["version"], str)

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
    assert isinstance(data["linker"]["command"], str) and data["linker"]["command"]
    assert data["linker"]["resolved_path"] is None
    assert data["linker"]["version"] is None
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


# Linker command contract: LD (default "ld") is parsed the same safe,
# shell-free way as CC, and version probing tries --version before
# falling back to -v, accepting output only from a zero return code.


def _patch_run(monkeypatch, module, fake_run):
    calls = []
    monkeypatch.setattr(
        module.subprocess, "run",
        lambda cmd, **kw: calls.append((cmd, kw)) or fake_run(cmd, **kw),
    )
    return calls


@pytest.mark.parametrize(
    "ld_env, which_target, resolved_path, stdout, expected_cmd",
    [
        pytest.param("ccache ld.lld", "ccache", "/usr/bin/ccache", "LLD 15.0.0\n",
                      ["/usr/bin/ccache", "ld.lld", "--version"], id="ccache-wrapped"),
        pytest.param("ld -arch arm64", "ld", "/usr/bin/ld", "@(#)PROGRAM:ld PROJECT:ld-1000\n",
                      ["/usr/bin/ld", "-arch", "arm64", "--version"], id="leading-flags"),
    ],
)
def test_linker_resolves_first_token_preserving_wrapper_or_flags(
    monkeypatch, ld_env, which_target, resolved_path, stdout, expected_cmd
):
    module = _load_script_module(f"compiler_provenance_linker_{which_target}")
    monkeypatch.setenv("LD", ld_env)

    which_calls = []
    monkeypatch.setattr(
        module.shutil, "which",
        lambda c: which_calls.append(c) or (resolved_path if c == which_target else None),
    )
    run_calls = _patch_run(
        monkeypatch, module,
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr=""),
    )

    info = module._linker_info()

    assert which_calls == [which_target], "expected only the first (executable) token to be resolved"
    assert len(run_calls) == 1, "must not fall back to -v when --version already succeeded"
    cmd, kwargs = run_calls[0]
    assert cmd == expected_cmd, "expected remaining tokens preserved and `--version` appended"
    assert not kwargs.get("shell", False), "subprocess call must remain shell-free"
    assert info == {"command": ld_env, "resolved_path": resolved_path, "version": stdout.strip()}


def test_linker_falls_back_to_dash_v_when_dash_dash_version_raises_oserror(monkeypatch):
    module = _load_script_module("compiler_provenance_linker_oserror_fallback")
    monkeypatch.setenv("LD", "ld")
    monkeypatch.setattr(module.shutil, "which", lambda c: "/usr/bin/ld")

    def _fake_run(cmd, **kw):
        if cmd[-1] == "--version":
            raise OSError("exec format error")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="@(#)PROGRAM:ld PROJECT:ld-1000\n")

    run_calls = _patch_run(monkeypatch, module, _fake_run)
    info = module._linker_info()

    assert [cmd[-1] for cmd, _ in run_calls] == ["--version", "-v"], (
        "expected -v to still be probed after --version raised OSError"
    )
    assert info["version"] == "@(#)PROGRAM:ld PROJECT:ld-1000"


def test_linker_version_tries_dash_dash_version_before_falling_back_to_dash_v(monkeypatch):
    module = _load_script_module("compiler_provenance_linker_fallback")
    monkeypatch.setenv("LD", "ld")
    monkeypatch.setattr(module.shutil, "which", lambda c: "/usr/bin/ld")

    def _fake_run(cmd, **kw):
        if cmd[-1] == "--version":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="unknown option\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="@(#)PROGRAM:ld PROJECT:ld-1000\n")

    run_calls = _patch_run(monkeypatch, module, _fake_run)
    info = module._linker_info()

    assert [cmd[-1] for cmd, _ in run_calls] == ["--version", "-v"], (
        "expected --version to be probed before falling back to -v"
    )
    assert info["version"] == "@(#)PROGRAM:ld PROJECT:ld-1000"


@pytest.mark.parametrize(
    "ld_env, which_result, run_side_effect, expected",
    [
        pytest.param(None, None, None,
                      {"command": "ld", "resolved_path": None, "version": None}, id="missing_executable"),
        pytest.param('ld "unterminated', None, None,
                      {"command": 'ld "unterminated', "resolved_path": None, "version": None},
                      id="malformed_ld_env_var"),
        pytest.param("ld", "/usr/bin/ld", "nonzero",
                      {"command": "ld", "resolved_path": "/usr/bin/ld", "version": None},
                      id="both_probes_nonzero"),
        pytest.param("ld", "/usr/bin/ld", "oserror",
                      {"command": "ld", "resolved_path": "/usr/bin/ld", "version": None},
                      id="subprocess_oserror"),
    ],
)
def test_linker_info_is_failure_safe(
    monkeypatch, request, ld_env, which_result, run_side_effect, expected
):
    module = _load_script_module(f"compiler_provenance_linker_{request.node.callspec.id}")

    if ld_env is None:
        monkeypatch.delenv("LD", raising=False)
    else:
        monkeypatch.setenv("LD", ld_env)
    monkeypatch.setattr(module.shutil, "which", lambda c: which_result)

    def _fake_run(cmd, **kw):
        if run_side_effect == "oserror":
            raise OSError("exec format error")
        return subprocess.CompletedProcess(cmd, 1, stdout="nope\n", stderr="nope\n")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    assert module._linker_info() == expected


# SDK contract: `sdk` exposes exactly SDKROOT, CONDA_BUILD_SYSROOT,
# selected_path, and version. selected_path prefers CONDA_BUILD_SYSROOT
# over SDKROOT, and version is read from the selected SDK's
# SDKSettings.json MinimalDisplayName, defaulting to null without failure.


def test_sdk_selected_path_prefers_conda_build_sysroot_and_reads_minimal_display_name(
    monkeypatch, tmp_path
):
    module = _load_script_module("compiler_provenance_sdk_precedence")

    conda_sysroot = tmp_path / "MacOSX26.5.sdk"
    conda_sysroot.mkdir()
    (conda_sysroot / "SDKSettings.json").write_text(json.dumps({"MinimalDisplayName": "26.5"}))
    sdkroot = tmp_path / "sdkroot"
    sdkroot.mkdir()
    monkeypatch.setenv("CONDA_BUILD_SYSROOT", str(conda_sysroot))
    monkeypatch.setenv("SDKROOT", str(sdkroot))

    assert module._sdk_info() == {
        "SDKROOT": str(sdkroot),
        "CONDA_BUILD_SYSROOT": str(conda_sysroot),
        "selected_path": str(conda_sysroot),
        "version": "26.5",
    }


def test_sdk_selected_path_falls_back_to_sdkroot_or_is_null_without_conda_build_sysroot(
    monkeypatch, tmp_path
):
    module = _load_script_module("compiler_provenance_sdk_fallback")
    monkeypatch.delenv("CONDA_BUILD_SYSROOT", raising=False)

    sdkroot = tmp_path / "sdkroot"
    sdkroot.mkdir()
    monkeypatch.setenv("SDKROOT", str(sdkroot))
    assert module._sdk_info()["selected_path"] == str(sdkroot)

    monkeypatch.delenv("SDKROOT", raising=False)
    info = module._sdk_info()
    assert info["selected_path"] is None
    assert info["version"] is None


@pytest.mark.parametrize(
    "write_settings",
    [
        pytest.param(None, id="missing-path"),
        pytest.param(lambda p: None, id="missing-settings-file"),
        pytest.param(lambda p: p.write_text("{not valid json"), id="malformed-json"),
        pytest.param(lambda p: p.write_text(json.dumps({"SomeOtherKey": "value"})),
                      id="absent-minimal-display-name"),
        pytest.param(lambda p: p.write_text(json.dumps([])), id="non-object-json"),
        pytest.param(lambda p: p.write_text(json.dumps({"MinimalDisplayName": 26.5})),
                      id="non-string-version"),
    ],
)
def test_sdk_version_is_null_for_invalid_metadata(monkeypatch, tmp_path, write_settings, request):
    module = _load_script_module(
        f"compiler_provenance_sdk_invalid_{request.node.callspec.id.replace('-', '_')}"
    )
    sdk_dir = tmp_path / "Test.sdk"
    monkeypatch.delenv("CONDA_BUILD_SYSROOT", raising=False)
    monkeypatch.setenv("SDKROOT", str(sdk_dir))

    if write_settings is not None:
        sdk_dir.mkdir()
        write_settings(sdk_dir / "SDKSettings.json")

    assert module._sdk_info()["version"] is None
