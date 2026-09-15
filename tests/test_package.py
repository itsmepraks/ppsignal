import sys
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
