"""Tests for ``msmarco_genqa.util.manifest``.

The module is small (filesystem + git introspection) and the goal here is
to lock down the schema shape and the privacy-relevant behaviour: no
absolute-home paths, no usernames, no host-specific bits.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from msmarco_genqa.util import manifest as manifest_mod
from msmarco_genqa.util.manifest import (
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
    DirtyTreeError,
    RequiredFieldMissingError,
    _validate_required,
    build_manifest,
    write_manifest,
    write_run_manifest,
)


# Reusable v2-compliant extras dict for tests that need to satisfy the
# required-field contract but aren't testing it themselves. Mirrors the
# shape runners will populate by the end of commits 1-3.
def _full_extras() -> dict:
    return {
        "seed": 42,
        "resolved_config_hash": "0" * 64,
        "data_fingerprint": "1" * 64,
        "env_fingerprint": "2" * 64,
    }


def test_build_manifest_schema(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("seed: 42\n")
    out = tmp_path / "metrics.json"
    out.write_text('{"mrr@10": 0.5}')
    reqs = tmp_path / "requirements.txt"
    reqs.write_text("numpy==1.0\n")

    manifest = build_manifest(
        project_root=tmp_path,
        command=["python", "experiments/run_retrieval.py"],
        config_paths=[cfg],
        dependency_paths=[reqs],
        output_paths=[out],
        extra={"top_k": 1000},
    )

    assert manifest["schema"] == "msmarco-genqa.manifest.v2"
    assert manifest["command"] == ["python", "experiments/run_retrieval.py"]
    assert manifest["extra"] == {"top_k": 1000}
    assert "timestamp_utc" in manifest
    # Each file record has the truncated digest.
    assert manifest["config"][0]["sha256_16"]
    assert manifest["outputs"][0]["sha256_16"]
    assert manifest["dependencies"][0]["sha256_16"]
    # Paths are repo-relative — no absolute home leakage.
    for rec in (*manifest["config"], *manifest["outputs"], *manifest["dependencies"]):
        assert not rec["path"].startswith("/")


def test_build_manifest_handles_missing_file(tmp_path: Path):
    manifest = build_manifest(
        project_root=tmp_path,
        output_paths=[tmp_path / "does_not_exist.tsv"],
    )
    rec = manifest["outputs"][0]
    assert rec.get("exists") is False
    assert "sha256_16" not in rec


def test_write_manifest_roundtrip(tmp_path: Path):
    manifest = build_manifest(
        project_root=tmp_path,
        command=["dummy"],
    )
    p = tmp_path / "subdir" / "manifest.json"
    write_manifest(manifest, p)
    assert p.exists()
    loaded = json.loads(p.read_text())
    assert loaded["command"] == ["dummy"]
    assert loaded["schema"] == "msmarco-genqa.manifest.v2"


def test_python_section_excludes_full_executable_path(tmp_path: Path):
    """Privacy: full sys.executable would leak the user's home dir."""
    manifest = build_manifest(project_root=tmp_path)
    exe = manifest["python"]["executable"]
    assert "/" not in exe
    assert "\\" not in exe


# --------------------------------------------------------------------------- #
# write_run_manifest — what the runners actually call
# --------------------------------------------------------------------------- #


def _fake_runner_outputs(project_root: Path, week: str) -> tuple[Path, Path, Path, Path]:
    """Lay out a fake repo + runner output dir.

    Returns ``(output_dir, config_path, metrics_path, extra_output_path)``.
    """
    # Dependency files at the repo root that the helper auto-discovers.
    (project_root / "requirements.txt").write_text("numpy==1.26.4\n")
    (project_root / "requirements-lock.txt").write_text("numpy==1.26.4\nfaiss-cpu==1.13.0\n")
    (project_root / "pyproject.toml").write_text("[project]\nname='fake'\n")

    # A config file under the canonical configs/ dir.
    configs_dir = project_root / "configs"
    configs_dir.mkdir()
    config_path = configs_dir / "baseline.yaml"
    config_path.write_text("seed: 42\n")

    # Runner output dir.
    output_dir = project_root / "outputs" / week
    output_dir.mkdir(parents=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text('{"task": "fake", "metrics": {"mrr@10": 0.5}}\n')
    extra_path = output_dir / "run.tsv"
    extra_path.write_text("q1\tQ0\td1\t1\t1.0\tfake\n")

    return output_dir, config_path, metrics_path, extra_path


def test_write_run_manifest_basic(tmp_path: Path):
    output_dir, config_path, metrics_path, extra_path = _fake_runner_outputs(
        tmp_path, "week_test"
    )

    manifest_path = write_run_manifest(
        project_root=tmp_path,
        output_dir=output_dir,
        command=["python", "experiments/run_fake.py"],
        config_path=config_path,
        extra_outputs=[extra_path],
        extra={"task": "fake", "n_eval_queries": 7},
        allow_incomplete=True,  # this test predates the v2 required-field
                                # contract; the contract is exercised in
                                # the dedicated tests below.
    )

    # 1. The manifest file exists at the standard location.
    assert manifest_path == output_dir / "manifest.json"
    assert manifest_path.exists()

    data = json.loads(manifest_path.read_text())

    # 2. metrics.json is auto-included as an output, with hash.
    out_paths = [rec["path"] for rec in data["outputs"]]
    assert any(p.endswith("metrics.json") for p in out_paths)
    assert any(p.endswith("run.tsv") for p in out_paths)
    metrics_rec = next(r for r in data["outputs"] if r["path"].endswith("metrics.json"))
    assert "sha256_16" in metrics_rec

    # 3. Config + auto-discovered deps are captured.
    assert data["config"][0]["path"].endswith("configs/baseline.yaml")
    dep_paths = [rec["path"] for rec in data["dependencies"]]
    # We auto-discover all three: lockfile first (priority), then requirements,
    # then pyproject. Each gets a hash.
    assert any(p.endswith("requirements-lock.txt") for p in dep_paths)
    assert any(p.endswith("requirements.txt") for p in dep_paths)
    assert any(p.endswith("pyproject.toml") for p in dep_paths)
    for rec in data["dependencies"]:
        assert rec.get("sha256_16")

    # 4. Extra metadata round-trips.
    assert data["extra"] == {"task": "fake", "n_eval_queries": 7}
    assert data["command"] == ["python", "experiments/run_fake.py"]


def test_write_run_manifest_paths_are_repo_relative(tmp_path: Path):
    """Privacy: no absolute path under the user's $HOME ever lands in the
    manifest. All recorded paths must be relative to the project root."""
    output_dir, config_path, _, extra_path = _fake_runner_outputs(tmp_path, "week_priv")
    manifest_path = write_run_manifest(
        project_root=tmp_path,
        output_dir=output_dir,
        command=["python", "experiments/run_fake.py"],
        config_path=config_path,
        extra_outputs=[extra_path],
        allow_incomplete=True,  # privacy is orthogonal to the v2 contract.
    )
    data = json.loads(manifest_path.read_text())

    # Walk every recorded file record and confirm: no absolute paths, no
    # leakage of the user's $HOME, no leakage of the tmp_path (which is an
    # absolute path under /private/var/... on macOS or /tmp/... on Linux).
    home = os.path.expanduser("~")
    tmp_str = str(tmp_path)
    for section in ("config", "dependencies", "outputs"):
        for rec in data[section]:
            p = rec["path"]
            assert not p.startswith("/"), f"absolute path leak: {p}"
            assert not p.startswith("\\"), f"absolute path leak: {p}"
            assert home not in p, f"home path leak: {p}"
            assert tmp_str not in p, f"tmp path leak: {p}"


def test_write_run_manifest_missing_deps_omitted(tmp_path: Path):
    """When the repo doesn't have a lockfile / pyproject, only the deps that
    *do* exist should show up in the manifest (no synthetic placeholders)."""
    output_dir = tmp_path / "outputs" / "week_min"
    output_dir.mkdir(parents=True)
    (output_dir / "metrics.json").write_text("{}")

    manifest_path = write_run_manifest(
        project_root=tmp_path,
        output_dir=output_dir,
        command=["python"],
        allow_incomplete=True,  # this test checks dep-discovery, not the
                                # required-field contract.
    )
    data = json.loads(manifest_path.read_text())
    assert data["dependencies"] == []
    assert data["config"] == []


def test_write_run_manifest_records_git_commit(tmp_path: Path):
    """The git section is best-effort; in tmp_path there's no .git, so the
    commit field is ``None`` but the section is still present."""
    output_dir = tmp_path / "outputs" / "week_git"
    output_dir.mkdir(parents=True)
    (output_dir / "metrics.json").write_text("{}")

    manifest_path = write_run_manifest(
        project_root=tmp_path,
        output_dir=output_dir,
        allow_incomplete=True,  # tmp_path has no .git, so git.commit is
                                # legitimately None here; the strict
                                # contract is tested elsewhere.
    )
    data = json.loads(manifest_path.read_text())
    assert "git" in data
    assert "commit" in data["git"]
    assert "dirty" in data["git"]


# --------------------------------------------------------------------------- #
# Dirty-tree handling
# --------------------------------------------------------------------------- #


def _make_minimal_output_dir(tmp_path: Path, name: str) -> Path:
    output_dir = tmp_path / "outputs" / name
    output_dir.mkdir(parents=True)
    (output_dir / "metrics.json").write_text("{}")
    return output_dir


def test_write_run_manifest_dirty_tree_emits_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """A dirty working tree should produce a clear warning so that downstream
    readers know the recorded commit alone is not enough to reproduce."""
    monkeypatch.setattr(
        manifest_mod,
        "_git_info",
        lambda: {"commit": "deadbeefface", "dirty": True},
    )
    output_dir = _make_minimal_output_dir(tmp_path, "dirty_warn")

    with caplog.at_level(logging.WARNING, logger="msmarco_genqa.util.manifest"):
        manifest_path = write_run_manifest(
            project_root=tmp_path,
            output_dir=output_dir,
            allow_incomplete=True,  # dirty-tree warning is orthogonal to
                                    # the required-field contract.
        )

    # The manifest still got written (default behaviour is warn-not-fail).
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data["git"]["dirty"] is True

    # The warning fired and explicitly named the dirty state.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected a WARNING-level log on dirty tree"
    assert any("dirty" in r.message.lower() for r in warnings)


def test_write_run_manifest_require_clean_tree_raises_on_dirty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """``require_clean_tree=True`` must refuse to write the manifest when the
    tree is dirty — including not leaving a partial file behind."""
    monkeypatch.setattr(
        manifest_mod,
        "_git_info",
        lambda: {"commit": "deadbeefface", "dirty": True},
    )
    output_dir = _make_minimal_output_dir(tmp_path, "dirty_refuse")

    with pytest.raises(DirtyTreeError, match="dirty"):
        write_run_manifest(
            project_root=tmp_path,
            output_dir=output_dir,
            require_clean_tree=True,
        )

    # No partial manifest written on refusal.
    assert not (output_dir / "manifest.json").exists()


def test_write_run_manifest_require_clean_tree_ok_when_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """When the tree is clean, ``require_clean_tree=True`` should be a no-op:
    manifest written, no warning fired."""
    monkeypatch.setattr(
        manifest_mod,
        "_git_info",
        lambda: {"commit": "deadbeefface", "dirty": False},
    )
    output_dir = _make_minimal_output_dir(tmp_path, "clean_ok")

    with caplog.at_level(logging.WARNING, logger="msmarco_genqa.util.manifest"):
        manifest_path = write_run_manifest(
            project_root=tmp_path,
            output_dir=output_dir,
            require_clean_tree=True,
            allow_incomplete=True,  # require_clean_tree is orthogonal to
                                    # the required-field contract.
        )

    assert manifest_path.exists()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("dirty" in r.message.lower() for r in warnings)


# --------------------------------------------------------------------------- #
# Schema v2 — required-fields contract
# --------------------------------------------------------------------------- #


def test_schema_version_constant():
    """The module-level constant and the field on built manifests agree."""
    assert SCHEMA_VERSION == "msmarco-genqa.manifest.v2"
    manifest = build_manifest(project_root=Path("/tmp"))
    assert manifest["schema"] == SCHEMA_VERSION


def test_required_fields_set():
    """The contract enumerates exactly the six expected fields. If this set
    grows, downstream callers must be updated in lockstep, so the test
    pins it explicitly."""
    assert REQUIRED_FIELDS == (
        "git.commit",
        "git.dirty",
        "extra.seed",
        "extra.resolved_config_hash",
        "extra.data_fingerprint",
        "extra.env_fingerprint",
    )


def _v2_compliant_manifest() -> dict:
    """Build a minimal manifest dict that satisfies every REQUIRED_FIELDS
    entry. Used as the positive-case fixture for the validator tests."""
    return {
        "schema": SCHEMA_VERSION,
        "git": {"commit": "deadbeef1234", "dirty": False},
        "extra": _full_extras(),
    }


def test_validate_required_passes_on_compliant_manifest():
    """A manifest with every required field populated must not raise."""
    _validate_required(_v2_compliant_manifest())  # no exception expected


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_validate_required_raises_when_any_field_is_missing(field: str):
    """Removing any single required field must trigger
    RequiredFieldMissingError with that field named in the message."""
    manifest = _v2_compliant_manifest()
    head, _, tail = field.partition(".")
    if tail:
        del manifest[head][tail]
    else:
        del manifest[head]

    with pytest.raises(RequiredFieldMissingError, match=field):
        _validate_required(manifest)


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_validate_required_raises_when_any_field_is_none(field: str):
    """A required field set explicitly to ``None`` is just as invalid as
    a missing one. ``False`` and ``0`` remain valid (the validator only
    rejects ``None`` and absent)."""
    manifest = _v2_compliant_manifest()
    head, _, tail = field.partition(".")
    if tail:
        manifest[head][tail] = None
    else:
        manifest[head] = None

    with pytest.raises(RequiredFieldMissingError, match=field):
        _validate_required(manifest)


def test_validate_required_accepts_falsy_but_non_none_values():
    """``False`` (e.g. git.dirty=False) and ``0`` (e.g. extra.seed=0) are
    legitimate values for required fields and must NOT trigger refusal."""
    manifest = _v2_compliant_manifest()
    manifest["git"]["dirty"] = False
    manifest["extra"]["seed"] = 0
    _validate_required(manifest)  # no exception expected


def test_validate_required_enumerates_all_missing_in_one_error():
    """When multiple fields are missing, the error names every one — so
    the user can fix them all in a single pass."""
    manifest = _v2_compliant_manifest()
    del manifest["extra"]["seed"]
    del manifest["extra"]["data_fingerprint"]

    with pytest.raises(RequiredFieldMissingError) as excinfo:
        _validate_required(manifest)
    msg = str(excinfo.value)
    assert "extra.seed" in msg
    assert "extra.data_fingerprint" in msg


def test_write_run_manifest_default_strict_refuses_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """By default (``allow_incomplete=False``), write_run_manifest must
    refuse to write a manifest that's missing required fields, and must
    not leave a partial file behind."""
    monkeypatch.setattr(
        manifest_mod,
        "_git_info",
        lambda: {"commit": "deadbeefface", "dirty": False},
    )
    output_dir = _make_minimal_output_dir(tmp_path, "v2_strict_refuses")

    with pytest.raises(RequiredFieldMissingError):
        write_run_manifest(
            project_root=tmp_path,
            output_dir=output_dir,
            extra={"seed": 42},  # missing the three fingerprint fields
        )

    assert not (output_dir / "manifest.json").exists()


def test_write_run_manifest_strict_accepts_full_extras(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Strict mode with all required extras populated succeeds."""
    monkeypatch.setattr(
        manifest_mod,
        "_git_info",
        lambda: {"commit": "deadbeefface", "dirty": False},
    )
    output_dir = _make_minimal_output_dir(tmp_path, "v2_strict_ok")

    manifest_path = write_run_manifest(
        project_root=tmp_path,
        output_dir=output_dir,
        extra=_full_extras(),
    )
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data["schema"] == SCHEMA_VERSION
    assert data["extra"]["seed"] == 42


def test_write_run_manifest_allow_incomplete_bypasses_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """``allow_incomplete=True`` must let the write proceed even with the
    extras dict empty — the development-time bypass."""
    monkeypatch.setattr(
        manifest_mod,
        "_git_info",
        lambda: {"commit": "deadbeefface", "dirty": False},
    )
    output_dir = _make_minimal_output_dir(tmp_path, "v2_bypass")

    manifest_path = write_run_manifest(
        project_root=tmp_path,
        output_dir=output_dir,
        allow_incomplete=True,
    )
    assert manifest_path.exists()


def test_write_run_manifest_strict_catches_missing_git_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """If git probing fails (e.g. tmp dir with no .git), strict mode must
    refuse the write even when ``extra`` is fully populated."""
    monkeypatch.setattr(
        manifest_mod,
        "_git_info",
        lambda: {"commit": None, "dirty": None},
    )
    output_dir = _make_minimal_output_dir(tmp_path, "v2_no_git")

    with pytest.raises(RequiredFieldMissingError, match="git.commit"):
        write_run_manifest(
            project_root=tmp_path,
            output_dir=output_dir,
            extra=_full_extras(),
        )

    assert not (output_dir / "manifest.json").exists()
