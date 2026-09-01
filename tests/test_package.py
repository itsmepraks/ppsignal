import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

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

    dev_dependencies = project["optional-dependencies"]["dev"]
    assert any(dep.startswith("PyYAML>=6") for dep in dev_dependencies)

    wheel_targets = data["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel_targets["packages"] == ["ppsignal"]


def _distribution_output(tmp_path_factory):
    configured = os.environ.get("PPSIGNAL_DIST_DIR")
    if configured is not None:
        return (ROOT / configured).resolve()

    output = tmp_path_factory.mktemp("dist")
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(output)],
        cwd=ROOT,
        check=True,
    )
    return output


def _single_artifact(distributions, pattern):
    artifacts = list(distributions.glob(pattern))
    assert len(artifacts) == 1, f"expected one {pattern} artifact, found {artifacts}"
    return artifacts[0]


def test_distribution_output_uses_prebuilt_artifacts(
    monkeypatch, tmp_path, tmp_path_factory
):
    prebuilt = tmp_path / "prebuilt"
    prebuilt.mkdir()
    monkeypatch.setenv("PPSIGNAL_DIST_DIR", str(prebuilt))

    def unexpected_build(*args, **kwargs):
        pytest.fail("started a second distribution build")

    monkeypatch.setattr(subprocess, "run", unexpected_build)

    assert _distribution_output(tmp_path_factory) == prebuilt


@pytest.fixture(scope="module")
def distributions(tmp_path_factory):
    return _distribution_output(tmp_path_factory)


def test_build_creates_source_only_wheel_and_sdist(distributions):
    wheel = _single_artifact(distributions, "*.whl")
    sdist = _single_artifact(distributions, "*.tar.gz")

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
    sdist = _single_artifact(distributions, "*.tar.gz")

    with tarfile.open(sdist) as archive:
        members = archive.getnames()

    prefix = "ppsignal-0.1.0/"
    relative_members = {
        name[len(prefix):] for name in members if name.startswith(prefix)
    }

    assert EXPECTED_SDIST_MEMBERS <= relative_members

    for name in members:
        suffix = Path(name).suffix
        assert suffix not in NATIVE_LIBRARY_SUFFIXES, f"native library in sdist: {name}"


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
    assert dev_pypi_dependencies["pyyaml"] == ">=6"

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

    build_native_description = tasks["build-native"]["description"]
    assert "shared library" in build_native_description
    assert "Hann" in build_native_description
    assert "Boxcar" in build_native_description
    assert "Bartlett" in build_native_description

    build_ext_description = tasks["build-ext"]["description"]
    assert "NumPy extension" in build_ext_description
    assert "Hann" in build_ext_description
    assert "Boxcar" in build_ext_description
    assert "Bartlett" in build_ext_description


def test_ci_runs_interpreted_native_and_distribution_checks():
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    )

    assert workflow["on"] == {
        "push": {"branches": ["main"]},
        "pull_request": None,
    }
    assert workflow["permissions"] == {"contents": "read"}

    jobs = workflow["jobs"]
    interpreted = jobs["interpreted"]
    assert interpreted["strategy"]["matrix"]["python-version"] == [
        "3.10",
        "3.12",
    ]
    interpreted_commands = [
        step["run"] for step in interpreted["steps"] if "run" in step
    ]
    assert any(
        'python -m pip install -e ".[dev]"' in cmd for cmd in interpreted_commands
    )
    assert "python -m pytest tests/" in interpreted_commands

    native = jobs["native"]
    native_commands = [step["run"] for step in native["steps"] if "run" in step]
    assert native["name"] == "Native build and tests"
    assert native["steps"][1]["with"]["python-version"] == "3.12"
    assert any(
        'python -m pip install -e ".[dev]"' in cmd for cmd in native_commands
    )
    assert native_commands[-3:] == [
        "python scripts/build_native.py",
        "python scripts/build_ext.py",
        "python -m pytest tests/test_build_scripts.py tests/test_native_ext.py",
    ]

    distribution = jobs["distribution"]
    distribution_commands = [
        step["run"] for step in distribution["steps"] if "run" in step
    ]
    assert distribution["name"] == "Distribution build and tests"
    assert distribution["steps"][1]["with"]["python-version"] == "3.12"
    assert distribution["env"] == {"PPSIGNAL_DIST_DIR": "dist"}
    assert any(
        'python -m pip install -e ".[dev]"' in cmd for cmd in distribution_commands
    )
    assert distribution_commands[-2:] == [
        "python -m build",
        "python -m pytest tests/test_package.py",
    ]


def test_isolated_wheel_import_without_pip(distributions, tmp_path):
    wheel = _single_artifact(distributions, "*.whl")
    target = tmp_path / "extracted-wheel"
    target.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(target)

    script = (
        "import sys, builtins, math\n"
        f"sys.path.insert(0, {str(target)!r})\n"
        "_real_import = builtins.__import__\n"
        "def _blocking_import(name, *args, **kwargs):\n"
        "    if name == 'ppsignal_native':\n"
        "        raise ModuleNotFoundError(\n"
        "            'No module named ppsignal_native', name='ppsignal_native'\n"
        "        )\n"
        "    return _real_import(name, *args, **kwargs)\n"
        "builtins.__import__ = _blocking_import\n"
        "from ppsignal.windows import bartlett, boxcar, hann\n"
        "assert hann(3).tolist() == [0.0, 1.0, 0.0]\n"
        "assert boxcar(3).tolist() == [1.0, 1.0, 1.0]\n"
        "assert bartlett(3).tolist() == [0.0, 1.0, 0.0]\n"
        "bartlett4 = bartlett(4).tolist()\n"
        "bartlett4_expected = [0.0, 2.0 / 3.0, 2.0 / 3.0, 0.0]\n"
        "assert len(bartlett4) == len(bartlett4_expected)\n"
        "assert all(\n"
        "    math.isclose(actual, expected, rel_tol=1e-15, abs_tol=1e-15)\n"
        "    for actual, expected in zip(bartlett4, bartlett4_expected)\n"
        "), bartlett4\n"
    )
    subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        check=True,
    )


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
    assert "the interpreted window kernels." in install_section

    boxcar_section = readme[
        readme.index("## Use Boxcar"):readme.index("## Use Bartlett")
    ]
    assert "from ppsignal.windows import boxcar" in boxcar_section
    assert "boxcar(5)" in boxcar_section
    assert "all ones" in boxcar_section
    assert "sym" in boxcar_section
    assert "does not change" in boxcar_section
    assert "exact numpy.float64 ones" in boxcar_section
    assert "scipy.signal.windows.boxcar" in boxcar_section

    bartlett_section = readme[
        readme.index("## Use Bartlett"):readme.index("## Develop")
    ]
    assert "from ppsignal.windows import bartlett" in bartlett_section
    assert "bartlett(5)" in bartlett_section
    assert "symmetric" in bartlett_section
    assert "periodic" in bartlett_section
    assert "1e-15" in bartlett_section
    assert "scipy.signal.windows.bartlett" in bartlett_section

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
