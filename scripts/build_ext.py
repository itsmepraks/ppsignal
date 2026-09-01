"""Build and smoke-test the window NumPy extension module."""

import importlib.util
import os
import sys
import tempfile
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import numpy as np
from postpyc.build import BuildError, build_file

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "ppsignal" / "_windows.py"
MODULE_NAME = "ppsignal_native"


def main() -> int:
    output_dir = Path(
        os.environ.get("PPSIGNAL_BUILD_DIR")
        or tempfile.mkdtemp(prefix="ppsignal-ext-")
    )
    target = output_dir / f"{MODULE_NAME}{EXTENSION_SUFFIXES[0]}"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        built = build_file(
            SOURCE,
            output=target,
            ext_module=True,
            module_name=MODULE_NAME,
        )
        spec = importlib.util.spec_from_file_location(MODULE_NAME, built)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (BuildError, OSError, ImportError) as error:
        print("extension build FAILED:")
        print("  " + "\n  ".join(str(error).splitlines()[:8]))
        return 1

    for name in ("hann", "boxcar", "bartlett"):
        if not isinstance(getattr(module, name, None), np.ufunc):
            print(f"extension build FAILED: {name} is not a NumPy ufunc")
            return 1
    print(f"built {built}")
    print("registered Hann, Boxcar, and Bartlett ufuncs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
