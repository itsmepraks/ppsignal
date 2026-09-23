import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent


def _load_pyproject():
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def test_pyproject_metadata():
    data = _load_pyproject()
    project = data["project"]

    assert project["name"] == "ppsignal"
    assert project["version"] == "0.1.0"
    assert project["authors"] == [{"name": "itsmepraks"}]
    assert project["requires-python"] == ">=3.10"
    assert project["dependencies"] == [
        "numpy",
        "postpyc>=0.3.0",
        "postyp>=0.3.0",
    ]

    wheel_packages = data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert wheel_packages == ["ppsignal"]


def test_build_system_requires_hatchling_version():
    data = _load_pyproject()
    assert data["build-system"]["requires"] == ["hatchling>=1.18"]


def test_keywords():
    data = _load_pyproject()
    assert data["project"]["keywords"] == [
        "signal-processing",
        "scipy",
        "gufunc",
        "postpyc",
        "postyp",
    ]


def test_classifiers_include_required_entries():
    data = _load_pyproject()
    classifiers = data["project"]["classifiers"]
    for expected in [
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
    ]:
        assert expected in classifiers


def test_urls_has_standard_spec_link():
    data = _load_pyproject()
    assert (
        data["project"]["urls"]["Standard"]
        == "https://github.com/openteams-ai/postpython/blob/main/docs/spec.md"
    )


def test_windows_module_does_not_exist():
    assert not (ROOT / "ppsignal" / "windows.py").exists()


def test_pixi_workspace_channels_and_platforms():
    workspace = _load_pyproject()["tool"]["pixi"]["workspace"]
    assert workspace["channels"] == ["conda-forge"]
    assert workspace["platforms"] == [
        "osx-arm64",
        "osx-64",
        "linux-64",
        "linux-aarch64",
        "win-64",
    ]


def test_pixi_has_editable_local_ppsignal_dependency():
    pypi_deps = _load_pyproject()["tool"]["pixi"]["pypi-dependencies"]
    ppsignal_dep = pypi_deps["ppsignal"]
    assert ppsignal_dep["path"] == "."
    assert ppsignal_dep["editable"] is True


def test_pixi_dependencies_require_python_and_c_compiler():
    deps = _load_pyproject()["tool"]["pixi"]["dependencies"]
    assert deps["python"] == ">=3.10"
    assert "c-compiler" in deps


def test_pixi_dev_environment_has_test_tooling():
    pixi = _load_pyproject()["tool"]["pixi"]
    dev_feature_deps = pixi["feature"]["dev"]["dependencies"]
    for pkg in ("pytest", "numpy"):
        assert pkg in dev_feature_deps

    dev_feature_pypi_deps = pixi["feature"]["dev"]["pypi-dependencies"]
    assert dev_feature_pypi_deps["build"] == ">=1.2"
    assert dev_feature_pypi_deps["hatchling"] == ">=1.18"

    environments = pixi["environments"]
    assert "dev" in environments
    dev_env = environments["dev"]
    features = dev_env["features"] if isinstance(dev_env, dict) else dev_env
    assert "dev" in features


def test_pixi_default_environment_includes_dev_feature():
    environments = _load_pyproject()["tool"]["pixi"]["environments"]
    default_env = environments["default"]
    features = (
        default_env.get("features", []) if isinstance(default_env, dict) else default_env
    )
    assert "dev" in features, (
        "default environment must include the dev feature, otherwise global "
        "tasks like 'pixi run test' and 'pixi run build-dist' lack "
        "pytest/build/hatchling"
    )


def test_dev_optional_dependencies_include_hatchling_for_no_isolation_builds():
    dev_extras = _load_pyproject()["project"]["optional-dependencies"]["dev"]
    assert "hatchling>=1.18" in dev_extras


def test_pixi_tasks_cover_test_and_build_workflows():
    tasks = _load_pyproject()["tool"]["pixi"]["tasks"]
    expected_cmds = {
        "test": "pytest tests/",
        "build-dist": "python -m build --no-isolation",
        "build-native": "python scripts/build_native.py",
        "build-ext": "python scripts/build_ext.py",
        "build-prefix": (
            "postpyc build ppsignal/_compiler_probe.py "
            "--prefix dist/prefix --module-name ppsignal"
        ),
        "provenance": "python scripts/compiler_provenance.py --output dist/provenance.json",
    }
    for name, cmd in expected_cmds.items():
        task = tasks[name]
        assert isinstance(task, dict)
        assert task["cmd"] == cmd
        assert isinstance(task["description"], str) and task["description"]


def test_built_wheel_and_sdist_contain_expected_files_and_no_native_binaries(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    wheels = list(tmp_path.glob("*.whl"))
    sdists = list(tmp_path.glob("*.tar.gz"))
    assert len(wheels) == 1, wheels
    assert len(sdists) == 1, sdists

    wheel_path = wheels[0]
    sdist_path = sdists[0]

    assert wheel_path.name.endswith("-py3-none-any.whl"), wheel_path.name

    with zipfile.ZipFile(wheel_path) as zf:
        wheel_names = set(zf.namelist())
    with tarfile.open(sdist_path, "r:gz") as tf:
        sdist_names = set(tf.getnames())

    forbidden_suffixes = (".so", ".pyd", ".dll", ".dylib")
    for name in wheel_names:
        assert not name.endswith(forbidden_suffixes), name
    for name in sdist_names:
        assert not name.endswith(forbidden_suffixes), name

    for expected in ("ppsignal/__init__.py", "ppsignal/_compiler_probe.py"):
        assert any(n.endswith(expected) for n in wheel_names), expected
        assert any(n.endswith(expected) for n in sdist_names), expected

    for script in ("scripts/build_native.py", "scripts/build_ext.py"):
        assert any(n.endswith(script) for n in sdist_names), script

    test_files = sorted(p.name for p in (ROOT / "tests").glob("*.py"))
    assert test_files, "expected current test files to exist"
    for test_file in test_files:
        assert any(n.endswith(f"tests/{test_file}") for n in sdist_names), test_file

    assert not any(n.endswith("ppsignal/windows.py") for n in wheel_names)
    assert not any(n.endswith("ppsignal/windows.py") for n in sdist_names)
