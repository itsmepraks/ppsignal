import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 backport
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]

NATIVE_LIBRARY_SUFFIXES = {".so", ".pyd", ".dll", ".dylib"}

EXPECTED_SDIST_MEMBERS = {
    "ppsignal/__init__.py",
    "ppsignal/_windows.py",
    "ppsignal/windows.py",
    "tests/test_package.py",
    "tests/test_windows.py",
    "tests/test_native_ext.py",
    "README.md",
    "pyproject.toml",
    "scripts/build_native.py",
    "scripts/build_ext.py",
    "tests/test_build_scripts.py",
}


def test_project_metadata_declares_source_only_package():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())

    project = data["project"]
    assert project["name"] == "ppsignal"
    assert project["version"] == "0.1.0"
    assert project["authors"] == [{"name": "itsmepraks"}]
    assert project["requires-python"] == ">=3.10"
    assert "numpy" in project["dependencies"]
    assert any(dep.startswith("postpyc>=0.3.0") for dep in project["dependencies"])
    assert any(dep.startswith("postyp>=0.3.0") for dep in project["dependencies"])

    wheel_targets = data["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel_targets["packages"] == ["ppsignal"]


@pytest.fixture(scope="module")
def distributions(tmp_path_factory):
    output = tmp_path_factory.mktemp("dist")
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(output)],
        cwd=ROOT,
        check=True,
    )
    return output


def test_build_creates_source_only_wheel_and_sdist(distributions):
    wheel = next(distributions.glob("*.whl"))
    sdist = next(distributions.glob("*.tar.gz"))

    assert wheel.name.endswith("-py3-none-any.whl")
    assert sdist.name == "ppsignal-0.1.0.tar.gz"

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()

    assert "ppsignal/__init__.py" in names
    assert "ppsignal/_windows.py" in names
    assert "ppsignal/windows.py" in names

    for name in names:
        suffix = Path(name).suffix
        assert suffix not in NATIVE_LIBRARY_SUFFIXES, f"native library in wheel: {name}"


def test_sdist_contains_expected_source_files(distributions):
    sdist = next(distributions.glob("*.tar.gz"))

    with tarfile.open(sdist) as archive:
        members = archive.getnames()

    prefix = "ppsignal-0.1.0/"
    relative_members = {
        name[len(prefix):] for name in members if name.startswith(prefix)
    }

    assert EXPECTED_SDIST_MEMBERS <= relative_members


def test_isolated_wheel_import_without_pip(distributions, tmp_path):
    wheel = next(distributions.glob("*.whl"))
    target = tmp_path / "extracted-wheel"
    target.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(target)

    script = (
        f"import sys; sys.path.insert(0, {str(target)!r}); "
        "from ppsignal.windows import hann; "
        "assert hann(3).tolist() == [0.0, 1.0, 0.0]"
    )
    subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        check=True,
    )
