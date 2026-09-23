"""Contract tests for .github/workflows/ci.yml.

These tests read the workflow file as text and assert the observable
security/reproducibility policy without parsing YAML. They are ongoing
contract tests: any change to the workflow's triggers, permissions, job
structure, action pinning, or provenance handling must keep these
assertions passing.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "ci.yml"


def _read_workflow():
    if not WORKFLOW_PATH.exists():
        pytest.fail(f"missing workflow file: {WORKFLOW_PATH}")
    return WORKFLOW_PATH.read_text()


def test_workflow_file_exists():
    assert WORKFLOW_PATH.exists()


def test_triggers_on_pull_request_and_push_to_main():
    text = _read_workflow()
    assert re.search(r"^on:", text, re.MULTILINE)
    assert re.search(r"pull_request:", text)
    assert re.search(r"push:", text)
    assert re.search(r"branches:\s*\[?.*\bmain\b", text)
    assert not re.search(r"pull_request_target", text), (
        "expected the workflow to never trigger on the unsafe `pull_request_target` event"
    )


def _top_level_permissions_block(text):
    match = re.search(r"^permissions:\s*\n", text, re.MULTILINE)
    assert match, "expected a top-level `permissions:` key"
    rest = text[match.end() :]
    next_top_level = re.search(r"^\S", rest, re.MULTILINE)
    return rest[: next_top_level.start()] if next_top_level else rest


def test_top_level_permissions_are_read_only_contents():
    text = _read_workflow()
    block = _top_level_permissions_block(text)
    entries = [
        line.strip()
        for line in block.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert entries == ["contents: read"], (
        "expected the top-level `permissions:` block to contain exactly one entry, "
        f"`contents: read`, but found {entries!r}"
    )


def test_interpreted_job_has_python_matrix_and_editable_install():
    text = _read_workflow()
    assert re.search(r"python-version:.*(\[.*3\.10.*3\.12.*\]|\n(\s*-\s*[\"']?3\.10.*\n)+.*3\.12)", text)
    assert re.search(r"pip install .*(-e|--editable)\s+\S*\[dev\]", text)


def test_locked_job_uses_pinned_ubuntu_checkout_and_pixi_setup():
    text = _read_workflow()
    assert "ubuntu-latest" in text
    assert "actions/checkout" in text
    assert "prefix-dev/setup-pixi" in text
    assert re.search(r"pixi-version:\s*v0\.81\.0", text)
    assert re.search(r"frozen:\s*true", text)
    assert re.search(r"cache:\s*true", text)


# CI/workflow-hardening contract: the tests below enforce that actions are
# pinned by immutable commit SHA (with a version tag comment for
# reviewability) rather than by mutable tag, that both checkout steps set
# `persist-credentials: false`, that every job declares `timeout-minutes`,
# that the provenance artifact upload sets `retention-days: 14`, and that
# the provenance artifact is uploaded right after provenance generation
# but before the build-dist step runs.

CHECKOUT_ACTION = "actions/checkout"
CHECKOUT_SHA = "3d3c42e5aac5ba805825da76410c181273ba90b1"
CHECKOUT_COMMENT = "v7.0.1"

SETUP_PYTHON_ACTION = "actions/setup-python"
SETUP_PYTHON_SHA = "5fda3b95a4ea91299a34e894583c3862153e4b97"
SETUP_PYTHON_COMMENT = "v7.0.0"

SETUP_PIXI_ACTION = "prefix-dev/setup-pixi"
SETUP_PIXI_SHA = "d3f436a425481402e6a95a1d1fc10331c708cd9e"
SETUP_PIXI_COMMENT = "v0.10.2"

UPLOAD_ARTIFACT_ACTION = "actions/upload-artifact"
UPLOAD_ARTIFACT_SHA = "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"
UPLOAD_ARTIFACT_COMMENT = "v7.0.1"


def _action_pin_pattern(action, sha, comment):
    return re.compile(
        r"uses:\s*" + re.escape(action) + "@" + re.escape(sha) + r"\s*#\s*" + re.escape(comment)
    )


def _job_blocks(text):
    jobs_match = re.search(r"^jobs:\s*\n", text, re.MULTILINE)
    assert jobs_match, "expected a top-level `jobs:` section"
    jobs_text = text[jobs_match.end() :]

    next_top_level = re.search(r"^\S", jobs_text, re.MULTILINE)
    if next_top_level:
        jobs_text = jobs_text[: next_top_level.start()]

    job_starts = [m.start() for m in re.finditer(r"^  [a-zA-Z_][\w-]*:\s*\n", jobs_text, re.MULTILINE)]
    job_starts.append(len(jobs_text))
    return [jobs_text[job_starts[i] : job_starts[i + 1]] for i in range(len(job_starts) - 1)]


def test_checkout_action_is_pinned_to_reviewed_sha_with_version_comment():
    text = _read_workflow()
    pattern = _action_pin_pattern(CHECKOUT_ACTION, CHECKOUT_SHA, CHECKOUT_COMMENT)
    matches = pattern.findall(text)
    assert len(matches) == 2, "expected both jobs to pin actions/checkout to the reviewed SHA"


def test_setup_python_action_is_pinned_to_reviewed_sha_with_version_comment():
    text = _read_workflow()
    pattern = _action_pin_pattern(SETUP_PYTHON_ACTION, SETUP_PYTHON_SHA, SETUP_PYTHON_COMMENT)
    assert pattern.search(text), "expected actions/setup-python pinned to the reviewed SHA"


def test_setup_pixi_action_is_pinned_to_reviewed_sha_with_version_comment():
    text = _read_workflow()
    pattern = _action_pin_pattern(SETUP_PIXI_ACTION, SETUP_PIXI_SHA, SETUP_PIXI_COMMENT)
    assert pattern.search(text), "expected prefix-dev/setup-pixi pinned to the reviewed SHA"


def test_upload_artifact_action_is_pinned_to_reviewed_sha_with_version_comment():
    text = _read_workflow()
    pattern = _action_pin_pattern(UPLOAD_ARTIFACT_ACTION, UPLOAD_ARTIFACT_SHA, UPLOAD_ARTIFACT_COMMENT)
    assert pattern.search(text), "expected actions/upload-artifact pinned to the reviewed SHA"


def test_both_checkout_steps_disable_persist_credentials():
    text = _read_workflow()
    pattern = _action_pin_pattern(CHECKOUT_ACTION, CHECKOUT_SHA, CHECKOUT_COMMENT)
    matches = list(pattern.finditer(text))
    assert len(matches) == 2, "expected exactly two pinned checkout steps"
    for match in matches:
        following = text[match.end() : match.end() + 200]
        assert re.search(r"persist-credentials:\s*false", following), (
            "expected persist-credentials: false immediately following each checkout step"
        )


def test_both_jobs_declare_timeout_minutes():
    text = _read_workflow()
    blocks = _job_blocks(text)
    assert len(blocks) == 2, "expected exactly two jobs (interpreted, locked)"
    for block in blocks:
        assert re.search(r"timeout-minutes:\s*\d+", block), "expected timeout-minutes on every job"


def _named_job_block(text, name):
    """Return the block of text for the given top-level job name."""
    blocks = _job_blocks(text)
    block = next((b for b in blocks if b.startswith(f"  {name}:")), None)
    assert block, f"expected a `{name}:` job"
    return block


def _job_steps(block):
    """Split a job block into its six-space-indented step blocks."""
    step_starts = [m.start() for m in re.finditer(r"^      - ", block, re.MULTILINE)]
    step_starts.append(len(block))
    return [block[step_starts[i] : step_starts[i + 1]] for i in range(len(step_starts) - 1)]


def test_locked_job_provenance_lifecycle_order_and_upload_contract():
    """Five consecutive steps must run, in order: generate provenance under
    the locked pixi env, validate the resulting JSON under the same locked
    env (not the runner's system Python), upload it only after successful
    validation (never `if: always()`, so invalid/partial provenance can
    never be published), then build the distribution, then run the test
    suite under the same locked pixi env."""
    text = _read_workflow()
    steps = _job_steps(_named_job_block(text, "locked"))

    provenance_idx = next(
        (i for i, step in enumerate(steps) if step.lstrip().splitlines()[0] == "- run: pixi run --locked provenance"),
        None,
    )
    assert provenance_idx is not None, "expected `- run: pixi run --locked provenance` as a step's first line"
    assert provenance_idx + 4 < len(steps), (
        "expected at least four steps immediately after `pixi run --locked provenance`"
    )

    validate_step, upload_step, build_step, test_step = steps[provenance_idx + 1 : provenance_idx + 5]

    assert (
        validate_step.lstrip().splitlines()[0]
        == "- run: pixi run --locked python -m json.tool dist/provenance.json"
    ), (
        "expected the step immediately after provenance generation to validate the JSON "
        "via `pixi run --locked python -m json.tool dist/provenance.json`, not the runner's "
        "system Python"
    )

    upload_uses_line = f"- uses: {UPLOAD_ARTIFACT_ACTION}@{UPLOAD_ARTIFACT_SHA} # {UPLOAD_ARTIFACT_COMMENT}"
    assert upload_step.lstrip().splitlines()[0] == upload_uses_line, (
        "expected the step immediately after JSON validation to be the pinned upload-artifact action"
    )

    assert build_step.lstrip().splitlines()[0] == "- run: pixi run --locked build-dist", (
        "expected the step immediately after the upload to be `pixi run --locked build-dist`"
    )

    assert test_step.lstrip().splitlines()[0] == "- run: pixi run --locked test", (
        "expected the step immediately after `pixi run --locked build-dist` to be `pixi run --locked test`"
    )

    upload_lines = [line.strip() for line in upload_step.splitlines() if line.strip()]
    assert upload_lines == [
        upload_uses_line,
        "with:",
        "name: compiler-provenance",
        "path: dist/provenance.json",
        "if-no-files-found: error",
        "retention-days: 14",
    ], (
        "expected the upload step to contain exactly the pinned `uses:` line, `with:`, "
        "`name: compiler-provenance`, `path: dist/provenance.json`, `if-no-files-found: error`, "
        f"and `retention-days: 14` (and nothing else, e.g. no `if: always()`), got {upload_lines!r}"
    )
