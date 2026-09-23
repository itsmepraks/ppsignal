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


def _compiler_info():
    command = os.environ.get("CC") or "cc"
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = []

    resolved_path = None
    version = None
    if tokens:
        resolved_path = shutil.which(tokens[0])
        if resolved_path is not None:
            try:
                result = subprocess.run(
                    [resolved_path, *tokens[1:], "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    shell=False,
                )
                if result.returncode == 0:
                    version = result.stdout.strip() or result.stderr.strip() or None
            except (OSError, subprocess.SubprocessError):
                version = None
    return {"command": command, "resolved_path": resolved_path, "version": version}


def _sdk_info():
    return {
        "SDKROOT": os.environ.get("SDKROOT"),
        "CONDA_BUILD_SYSROOT": os.environ.get("CONDA_BUILD_SYSROOT"),
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
        "sdk": _sdk_info(),
        "packages": _package_versions(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Path to write the provenance JSON")
    args = parser.parse_args()

    provenance = _build_provenance()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
