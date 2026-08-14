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
    "ROADMAP.md",
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


def test_pixi_configuration_has_supported_platforms_and_tasks():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())

    pixi = data["tool"]["pixi"]

    assert pixi["workspace"]["channels"] == ["conda-forge"]
    assert pixi["workspace"]["platforms"] == [
        "osx-arm64",
        "osx-64",
        "linux-64",
        "linux-aarch64",
        "win-64",
    ]

    assert pixi["pypi-dependencies"]["ppsignal"] == {"path": ".", "editable": True}

    assert pixi["dependencies"]["python"] == ">=3.10"
    assert pixi["dependencies"]["c-compiler"] == "*"

    dev_dependencies = pixi["feature"]["dev"]["dependencies"]
    assert dev_dependencies["pytest"] == ">=7"
    assert dev_dependencies["numpy"] == "*"

    dev_pypi_dependencies = pixi["feature"]["dev"]["pypi-dependencies"]
    assert dev_pypi_dependencies["build"] == ">=1.2"

    assert pixi["environments"]["default"] == {"solve-group": "default"}
    assert pixi["environments"]["dev"] == {
        "features": ["dev"],
        "solve-group": "default",
    }

    tasks = pixi["tasks"]
    expected_commands = {
        "test": "pytest tests/",
        "build-dist": "python -m build",
        "build-native": "python scripts/build_native.py",
        "build-ext": "python scripts/build_ext.py",
        "build-prefix": (
            "postpyc build ppsignal/_windows.py --prefix dist/prefix "
            "--module-name ppsignal"
        ),
    }
    for task, cmd in expected_commands.items():
        assert tasks[task]["cmd"] == cmd


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


def test_roadmap_records_targets_and_compiler_gaps():
    roadmap = (ROOT / "ROADMAP.md").read_text()

    accuracy_section = roadmap[
        roadmap.index("## Accuracy target"):roadmap.index("## Completed")
    ]
    assert "1e-14" in accuracy_section

    completed_section = roadmap[
        roadmap.index("## Completed"):roadmap.index("## Next windows")
    ]
    assert "Hann" in completed_section

    next_section = roadmap[
        roadmap.index("## Next windows"):roadmap.index("## Deferred work")
    ]
    next_windows = ["boxcar", "hamming", "blackman", "bartlett", "triang"]
    positions = [next_section.index(name) for name in next_windows]
    assert positions == sorted(positions)

    deferred_section = roadmap[
        roadmap.index("## Deferred work"):roadmap.index("## Compiler gaps")
    ]
    assert "kaiser" in deferred_section
    assert "computed output" in deferred_section

    compiler_section = roadmap[roadmap.index("## Compiler gaps"):]
    assert "postpython issue" in compiler_section


def test_readme_has_current_package_commands():
    readme = (ROOT / "README.md").read_text()

    assert "Alpha" in readme
    assert "from ppsignal.windows import hann" in readme
    assert "hann(5)" in readme
    assert "sym=False" in readme

    install_section = readme[
        readme.index("## Install"):readme.index("## Use Hann")
    ]
    assert "do not compile native code" in install_section
    assert "ppsignal_native" in install_section
    assert "ppsignal` uses" in install_section
    assert "the interpreted Hann kernel" in install_section

    develop_section = readme[
        readme.index("## Develop"):readme.index("## Working rules (summary)")
    ]

    assert "pixi install -e dev" in develop_section
    assert "pixi run -e dev test" in develop_section
    assert "pixi run -e dev build-dist" in develop_section
    assert "pixi run build-native" in develop_section
    assert "pixi run build-ext" in develop_section
    assert "pixi run build-prefix" in develop_section

    assert "docs/spec.md" in readme
    assert "postscipy-roadmap.md" in readme
