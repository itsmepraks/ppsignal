#!/usr/bin/env python3
"""Capture compiler/SDK/package versions as diagnostic build metadata."""

import argparse
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = 1
PACKAGE_NAMES = ("numpy", "postpyc", "postyp")

SCHEMA_SECTIONS = {
    "ci": frozenset({"repository", "commit_sha", "run_id", "run_attempt", "job"}),
    "python": frozenset({"implementation", "version", "executable"}),
    "platform": frozenset({"system", "release", "machine"}),
    "compiler": frozenset({"command", "resolved_path", "version"}),
    "linker": frozenset({"command", "resolved_path", "version"}),
    "sdk": frozenset({"SDKROOT", "CONDA_BUILD_SYSROOT", "selected_path", "version"}),
    "packages": frozenset({"numpy", "postpyc", "postyp"}),
}
SCHEMA_TOP_LEVEL_KEYS = frozenset({"schema_version", *SCHEMA_SECTIONS})


def _resolve_tool(command):
    """Split a shell-style command string and resolve its first token, shell-free."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = []
    if not tokens:
        return None, []
    resolved_path = shutil.which(tokens[0])
    if resolved_path is None:
        return None, []
    return resolved_path, tokens[1:]


def _probe_tool_version(resolved_path, extra_args, version_flags):
    for flag in version_flags:
        try:
            result = subprocess.run(
                [resolved_path, *extra_args, flag],
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )
        except (subprocess.SubprocessError, OSError):
            continue
        if result.returncode == 0:
            return result.stdout.strip() or result.stderr.strip() or None
    return None


def _tool_info(command, version_flags):
    resolved_path, extra_args = _resolve_tool(command)
    version = None
    if resolved_path is not None:
        version = _probe_tool_version(resolved_path, extra_args, version_flags)
    return {"command": command, "resolved_path": resolved_path, "version": version}


def _compiler_info():
    command = os.environ.get("CC") or "cc"
    return _tool_info(command, ("--version",))


def _linker_info():
    command = os.environ.get("LD") or "ld"
    return _tool_info(command, ("--version", "-v"))


def _sdk_version(selected_path):
    if not selected_path:
        return None
    settings_path = Path(selected_path) / "SDKSettings.json"
    try:
        data = json.loads(settings_path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    version = data.get("MinimalDisplayName")
    return version if isinstance(version, str) else None


def _sdk_info():
    sdkroot = os.environ.get("SDKROOT")
    conda_build_sysroot = os.environ.get("CONDA_BUILD_SYSROOT")
    selected_path = conda_build_sysroot or sdkroot
    return {
        "SDKROOT": sdkroot,
        "CONDA_BUILD_SYSROOT": conda_build_sysroot,
        "selected_path": selected_path,
        "version": _sdk_version(selected_path),
    }


def _ci_info():
    return {
        "repository": os.environ.get("GITHUB_REPOSITORY") or None,
        "commit_sha": os.environ.get("GITHUB_SHA") or None,
        "run_id": os.environ.get("GITHUB_RUN_ID") or None,
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT") or None,
        "job": os.environ.get("GITHUB_JOB") or None,
    }


def _package_versions():
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _build_provenance():
    return {
        "schema_version": SCHEMA_VERSION,
        "ci": _ci_info(),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "compiler": _compiler_info(),
        "linker": _linker_info(),
        "sdk": _sdk_info(),
        "packages": _package_versions(),
    }


def _validate_provenance_schema(data):
    if not isinstance(data, dict):
        raise ValueError("document is not a JSON object")
    if set(data.keys()) != SCHEMA_TOP_LEVEL_KEYS:
        unexpected = sorted(set(data.keys()) ^ SCHEMA_TOP_LEVEL_KEYS)
        raise ValueError(f"unexpected top-level keys: {unexpected}")
    schema_version = data.get("schema_version")
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {schema_version!r}")
    for section, expected_keys in SCHEMA_SECTIONS.items():
        value = data[section]
        if not isinstance(value, dict):
            raise ValueError(f"section {section!r} is not an object")
        if set(value.keys()) != expected_keys:
            raise ValueError(
                f"section {section!r} has unexpected keys: {sorted(set(value.keys()) ^ expected_keys)}"
            )
        for key, leaf in value.items():
            if leaf is not None and not isinstance(leaf, str):
                raise ValueError(f"{section}.{key} must be a string or null")


def _validate_command(path) -> int:
    try:
        with open(path) as f:
            data = json.load(f)
        _validate_provenance_schema(data)
    except (OSError, ValueError) as exc:
        print(f"FAILED: {path}: {exc}")
        return 1
    print(f"OK: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", help="Path to write the provenance JSON")
    group.add_argument(
        "--validate", help="Path to an existing provenance JSON file to validate"
    )
    args = parser.parse_args()

    if args.validate is not None:
        return _validate_command(args.validate)

    provenance = _build_provenance()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
