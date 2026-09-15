#!/usr/bin/env python3
"""Compile the compiler probe into a plain C shared library, header, and manifest."""

import os
import sys
import tempfile
from pathlib import Path

from postpyc.build import BuildError, build_file

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "ppsignal" / "_compiler_probe.py"


def _build_dir() -> Path:
    env_dir = os.environ.get("PPSIGNAL_BUILD_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(tempfile.mkdtemp(prefix="ppsignal-native-"))


def _report_failure(prefix: str, exc: Exception | None = None) -> int:
    print(f"FAILED: {prefix}" if prefix else "FAILED")
    if exc is not None:
        for line in [ln for ln in str(exc).splitlines() if ln.strip()][:2]:
            print(line)
    return 1


def main() -> int:
    try:
        output_dir = _build_dir()
        output = output_dir / "ppsignal_probe.so"
        header = output_dir / "ppsignal_probe.h"
        manifest = output_dir / "ppsignal_probe.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        for expected in (output, header, manifest):
            expected.unlink(missing_ok=True)
        build_file(
            SOURCE,
            output=output,
            emit_header=True,
            emit_manifest=True,
        )
    except BuildError as exc:
        return _report_failure("native build error", exc)
    except OSError as exc:
        return _report_failure("", exc)

    if not (output.is_file() and header.is_file() and manifest.is_file()):
        return _report_failure("native build did not produce all expected artifacts")

    print(f"OK: wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
