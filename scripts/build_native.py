"""Compile the Hann kernel as a plain shared library with C ABI sidecars."""

import os
import sys
import tempfile
from pathlib import Path

from postpyc.build import BuildError, build_file

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "ppsignal" / "_windows.py"


def main() -> int:
    output_dir = Path(
        os.environ.get("PPSIGNAL_BUILD_DIR")
        or tempfile.mkdtemp(prefix="ppsignal-native-")
    )
    output = output_dir / "ppsignal_windows.so"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        library = build_file(
            SOURCE,
            output=output,
            emit_header=True,
            emit_manifest=True,
        )
    except (BuildError, OSError) as error:
        print("native build FAILED:")
        print("  " + "\n  ".join(str(error).splitlines()[:8]))
        return 1

    print(f"built hann library: {library}")
    print(f"built C header: {library.with_suffix('.h')}")
    print(f"built ABI manifest: {library.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
