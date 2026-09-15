#!/usr/bin/env python3
"""Compile the compiler probe into a NumPy ufunc CPython extension module."""

import importlib.util
import os
import sys
import tempfile
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import numpy as np

from postpyc.build import BuildError, build_file

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "ppsignal" / "_compiler_probe.py"
MODULE_NAME = "ppsignal_probe_native"


def _build_dir() -> Path:
    env_dir = os.environ.get("PPSIGNAL_BUILD_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(tempfile.mkdtemp(prefix="ppsignal-ext-"))


def _report_failure(prefix: str, exc: Exception | None = None) -> int:
    print(f"FAILED: {prefix}" if prefix else "FAILED")
    if exc is not None:
        for line in [ln for ln in str(exc).splitlines() if ln.strip()][:2]:
            print(line)
    return 1


def _validate_identity(identity) -> str | None:
    """Return a concise failure reason, or None if identity behaves correctly."""
    for values in (
        np.array([-2.5, 0.0, 3.25, 10.0], dtype=np.float64),
        np.array([], dtype=np.float64),
    ):
        result = identity(values)
        if result.dtype != np.float64:
            return f"identity returned dtype {result.dtype}, expected float64"
        if not np.array_equal(result, values):
            return "identity did not return the exact input values"
        if np.shares_memory(result, values):
            return "identity result shares memory with its input"
    return None


def main() -> int:
    try:
        output_dir = _build_dir()
        output = output_dir / f"{MODULE_NAME}{EXTENSION_SUFFIXES[0]}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)
        build_file(
            SOURCE,
            output=output,
            ext_module=True,
            module_name=MODULE_NAME,
        )
    except BuildError as exc:
        return _report_failure("extension build error", exc)
    except OSError as exc:
        return _report_failure("", exc)

    if not output.is_file():
        return _report_failure("extension build did not produce the expected artifact")

    try:
        spec = importlib.util.spec_from_file_location(MODULE_NAME, output)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not isinstance(module.identity, np.ufunc):
            print("FAILED: identity is not a NumPy ufunc")
            return 1
        error = _validate_identity(module.identity)
        if error is not None:
            print(f"FAILED: {error}")
            return 1
    except Exception as exc:
        return _report_failure("extension load error", exc)

    print(f"OK: wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
