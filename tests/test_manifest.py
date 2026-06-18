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
    CANONICAL_SAMPLED_CAVEAT,
    PROFILE_REQUIRED_FIELDS,
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
    DirtyTreeError,
    RequiredFieldMissingError,
    _validate_required,
    build_manifest,
    compute_data_fingerprint,
    compute_env_fingerprint,
    compute_resolved_config_hash,
    compute_sampling_block,
    write_manifest,
    write_resolved_config,
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


def _fake_runner_outputs(project_root: Path, stage: str) -> tuple[Path, Path, Path, Path]:
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
    output_dir = project_root / "outputs" / stage
    output_dir.mkdir(parents=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text('{"task": "fake", "metrics": {"mrr@10": 0.5}}\n')
    extra_path = output_dir / "run.tsv"
    extra_path.write_text("q1\tQ0\td1\t1\t1.0\tfake\n")

    return output_dir, config_path, metrics_path, extra_path


def test_write_run_manifest_basic(tmp_path: Path):
    output_dir, config_path, metrics_path, extra_path = _fake_runner_outputs(
        tmp_path, "stage_test"
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
    assert Path(data["config"][0]["path"]).as_posix().endswith("configs/baseline.yaml")
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
    output_dir, config_path, _, extra_path = _fake_runner_outputs(tmp_path, "stage_priv")
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
    output_dir = tmp_path / "outputs" / "stage_min"
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
    output_dir = tmp_path / "outputs" / "stage_git"
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


# --------------------------------------------------------------------------- #
# Per-task profile contract (nli_grounding) — R5 metric-robustness
# --------------------------------------------------------------------------- #


_NLI_PROFILE_FIELDS: tuple[str, ...] = PROFILE_REQUIRED_FIELDS["nli_grounding"]


def _nli_extras() -> dict:
    """Minimal ``extra.nli`` sub-dict that satisfies the nli_grounding
    profile. Mirrors the shape an R5 NLI runner will populate."""
    return {
        "backbone": "cross-encoder/nli-deberta-v3-small",
        "revision": "fa2804872c3b4bd748f38c0185cc85775361e735",
        "score_formula": "p_entail",
        "threshold": {"type": "raw_scores"},
        "premise_hypothesis_direction": "passage_as_premise",
        "label_index_mapping": {"entailment": 2, "neutral": 1, "contradiction": 0},
        "aggregation": "whole_passage",
    }


def _v2_nli_compliant_manifest() -> dict:
    """Base v2-compliant manifest with the nli_grounding profile fields
    layered in. Positive-case fixture for profile validator tests."""
    manifest = _v2_compliant_manifest()
    manifest["extra"]["nli"] = _nli_extras()
    return manifest


def test_profile_required_fields_set():
    """The nli_grounding profile enumerates exactly the seven fields locked
    by the R5 design (backbone, revision, score formula, threshold,
    premise/hypothesis direction, label-index mapping, aggregation). Pinned
    explicitly so any change forces a deliberate update."""
    assert PROFILE_REQUIRED_FIELDS["nli_grounding"] == (
        "extra.nli.backbone",
        "extra.nli.revision",
        "extra.nli.score_formula",
        "extra.nli.threshold",
        "extra.nli.premise_hypothesis_direction",
        "extra.nli.label_index_mapping",
        "extra.nli.aggregation",
    )


def test_validate_required_passes_with_nli_profile_when_full():
    """A manifest with both base and NLI fields populated passes under
    profile='nli_grounding'."""
    _validate_required(_v2_nli_compliant_manifest(), profile="nli_grounding")


@pytest.mark.parametrize("field", _NLI_PROFILE_FIELDS)
def test_validate_required_raises_when_any_nli_field_is_missing(field: str):
    """Removing any single NLI profile field must trigger
    RequiredFieldMissingError with that field named in the message."""
    manifest = _v2_nli_compliant_manifest()
    # All NLI fields live under "extra.nli.<leaf>".
    _, _, leaf = field.rpartition(".")
    del manifest["extra"]["nli"][leaf]

    with pytest.raises(RequiredFieldMissingError, match=field):
        _validate_required(manifest, profile="nli_grounding")


@pytest.mark.parametrize("field", _NLI_PROFILE_FIELDS)
def test_validate_required_raises_when_any_nli_field_is_none(field: str):
    """An NLI profile field set explicitly to ``None`` is just as invalid
    as missing."""
    manifest = _v2_nli_compliant_manifest()
    _, _, leaf = field.rpartition(".")
    manifest["extra"]["nli"][leaf] = None

    with pytest.raises(RequiredFieldMissingError, match=field):
        _validate_required(manifest, profile="nli_grounding")


def test_validate_required_without_profile_ignores_missing_nli_fields():
    """Backward compatibility: with profile=None (default), a manifest that
    lacks the entire extra.nli sub-dict is still valid as long as the base
    REQUIRED_FIELDS are populated. Existing non-NLI runners must keep
    working unchanged."""
    manifest = _v2_compliant_manifest()  # no extra.nli at all
    _validate_required(manifest)  # no exception expected
    _validate_required(manifest, profile=None)  # explicit form also fine


def test_validate_required_unknown_profile_raises_valueerror():
    """Typo'd profile names must fail loudly up-front rather than silently
    degrading to base-only validation."""
    manifest = _v2_compliant_manifest()
    with pytest.raises(ValueError, match="unknown manifest profile"):
        _validate_required(manifest, profile="nli-grounding")  # wrong sep


def test_validate_required_combines_base_and_profile_missing_in_one_error():
    """When both a base required field and a profile field are missing,
    the error names both — single-pass diagnosis."""
    manifest = _v2_nli_compliant_manifest()
    del manifest["extra"]["seed"]
    del manifest["extra"]["nli"]["score_formula"]

    with pytest.raises(RequiredFieldMissingError) as excinfo:
        _validate_required(manifest, profile="nli_grounding")
    msg = str(excinfo.value)
    assert "extra.seed" in msg
    assert "extra.nli.score_formula" in msg


def test_validate_required_error_message_names_profile_when_set():
    """The error message identifies the profile in play so the user knows
    which contract refused the write."""
    manifest = _v2_nli_compliant_manifest()
    del manifest["extra"]["nli"]["backbone"]

    with pytest.raises(RequiredFieldMissingError, match="nli_grounding"):
        _validate_required(manifest, profile="nli_grounding")


def test_write_run_manifest_with_nli_profile_strict_refuses_when_nli_extras_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """``write_run_manifest(profile='nli_grounding')`` must enforce the NLI
    profile even when base fields are populated."""
    monkeypatch.setattr(
        manifest_mod,
        "_git_info",
        lambda: {"commit": "deadbeefface", "dirty": False},
    )
    output_dir = _make_minimal_output_dir(tmp_path, "nli_strict_refuses")

    with pytest.raises(RequiredFieldMissingError, match="extra.nli"):
        write_run_manifest(
            project_root=tmp_path,
            output_dir=output_dir,
            extra=_full_extras(),  # base required ok; nli sub-dict absent
            profile="nli_grounding",
        )

    assert not (output_dir / "manifest.json").exists()


def test_write_run_manifest_with_nli_profile_strict_accepts_when_nli_extras_populated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Strict mode + profile='nli_grounding' + all NLI fields populated
    must write successfully and record the NLI block on disk."""
    monkeypatch.setattr(
        manifest_mod,
        "_git_info",
        lambda: {"commit": "deadbeefface", "dirty": False},
    )
    output_dir = _make_minimal_output_dir(tmp_path, "nli_strict_ok")

    extras = {**_full_extras(), "nli": _nli_extras()}
    manifest_path = write_run_manifest(
        project_root=tmp_path,
        output_dir=output_dir,
        extra=extras,
        profile="nli_grounding",
    )
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data["extra"]["nli"]["backbone"] == "cross-encoder/nli-deberta-v3-small"
    assert data["extra"]["nli"]["aggregation"] == "whole_passage"


# --------------------------------------------------------------------------- #
# Resolved config — content hash + adjacent YAML artefact
# --------------------------------------------------------------------------- #


def _sample_resolved_cfg() -> dict:
    """A representative cfg dict with nesting + heterogeneous values, so
    hashing tests can detect order / type / depth sensitivity bugs."""
    return {
        "seed": 42,
        "retrieval": {"backend": "bm25s", "k1": 1.5, "b": 0.75, "top_k": 1000},
        "dense": {"model_name": "all-MiniLM-L6-v2", "sample_size": 50000},
        "eval_retrieval": {"output_dir": "outputs/W2_bm25"},
    }


def test_compute_resolved_config_hash_is_64_hex():
    """Hash is the full sha256 hex (not the truncated 16-char form)."""
    h = compute_resolved_config_hash(_sample_resolved_cfg())
    assert len(h) == 64
    int(h, 16)  # raises if not valid hex


def test_compute_resolved_config_hash_deterministic():
    """Same dict in, same hash out — stable across calls."""
    cfg = _sample_resolved_cfg()
    assert compute_resolved_config_hash(cfg) == compute_resolved_config_hash(cfg)


def test_compute_resolved_config_hash_insensitive_to_key_order():
    """Two dicts with identical content but different insertion order must
    hash to the same value, otherwise the hash is unstable across Python
    versions / dict re-iterations and the manifest contract is meaningless."""
    a = {"seed": 42, "retrieval": {"backend": "bm25s", "k1": 1.5}}
    b = {"retrieval": {"k1": 1.5, "backend": "bm25s"}, "seed": 42}
    assert compute_resolved_config_hash(a) == compute_resolved_config_hash(b)


def test_compute_resolved_config_hash_sensitive_to_value_change():
    """Changing any leaf value must flip the hash; otherwise CLI-override
    capture is meaningless."""
    base = _sample_resolved_cfg()
    h_base = compute_resolved_config_hash(base)

    perturbed = _sample_resolved_cfg()
    perturbed["dense"]["sample_size"] = 30000  # the W4-A density sweep case
    assert compute_resolved_config_hash(perturbed) != h_base


def test_compute_resolved_config_hash_handles_non_json_native(tmp_path: Path):
    """``Path`` objects appear in cfg dicts via CLI ``type=Path`` argparse
    flags. The hasher must coerce them deterministically rather than
    raise ``TypeError`` (handled via ``json.dumps(default=str)``)."""
    cfg = {"output_dir": tmp_path / "outputs" / "x"}
    h = compute_resolved_config_hash(cfg)
    assert len(h) == 64


def test_write_resolved_config_roundtrip(tmp_path: Path):
    """The yaml on disk must reload to the same dict (modulo yaml's
    canonical types)."""
    import yaml

    cfg = _sample_resolved_cfg()
    path = write_resolved_config(cfg, tmp_path / "outputs" / "run1")
    assert path == tmp_path / "outputs" / "run1" / "resolved_config.yaml"
    assert path.exists()
    loaded = yaml.safe_load(path.read_text())
    assert loaded == cfg


def test_write_resolved_config_creates_output_dir(tmp_path: Path):
    """``output_dir`` not existing yet must NOT raise — the helper mkdirs
    p=True so it can be the first thing a runner writes."""
    target = tmp_path / "deep" / "nested" / "outputs"
    assert not target.exists()
    path = write_resolved_config({"seed": 7}, target)
    assert path.exists()
    assert target.is_dir()


def test_write_resolved_config_keys_are_sorted(tmp_path: Path):
    """The on-disk yaml has keys sorted, so diff'ing two resolved configs
    across runs is monotone (no order-of-write noise)."""
    cfg = {"z_last": 3, "a_first": 1, "m_middle": 2}
    path = write_resolved_config(cfg, tmp_path)
    content = path.read_text()
    # Each key should appear in alphabetical order in the file.
    a_pos = content.index("a_first")
    m_pos = content.index("m_middle")
    z_pos = content.index("z_last")
    assert a_pos < m_pos < z_pos


# --------------------------------------------------------------------------- #
# Data fingerprint — lean
# --------------------------------------------------------------------------- #


def test_compute_data_fingerprint_is_64_hex(tmp_path: Path):
    h = compute_data_fingerprint(cache_dir=tmp_path / "cache", corpus_limit=None)
    assert len(h) == 64
    int(h, 16)


def test_compute_data_fingerprint_deterministic(tmp_path: Path):
    """Two calls with identical inputs return identical fingerprints."""
    a = compute_data_fingerprint(cache_dir=tmp_path / "cache", corpus_limit=8800000)
    b = compute_data_fingerprint(cache_dir=tmp_path / "cache", corpus_limit=8800000)
    assert a == b


def test_compute_data_fingerprint_sensitive_to_cache_dir(tmp_path: Path):
    """Different cache_dir → different fingerprint (different ir_datasets cache
    anchors different corpus identities)."""
    a = compute_data_fingerprint(cache_dir=tmp_path / "cache_a", corpus_limit=None)
    b = compute_data_fingerprint(cache_dir=tmp_path / "cache_b", corpus_limit=None)
    assert a != b


def test_compute_data_fingerprint_sensitive_to_corpus_limit(tmp_path: Path):
    """corpus_limit=None (full corpus) vs corpus_limit=200000 (smoke) must
    produce distinct fingerprints — different runs."""
    a = compute_data_fingerprint(cache_dir=tmp_path / "cache", corpus_limit=None)
    b = compute_data_fingerprint(cache_dir=tmp_path / "cache", corpus_limit=200000)
    assert a != b


def test_compute_data_fingerprint_extra_files_change_hash(tmp_path: Path):
    """Changing the content of an extra file changes the fingerprint —
    so sample_doc_ids.json drift or input_run.tsv drift is caught."""
    extra = tmp_path / "sample_doc_ids.json"
    extra.write_text('["d1","d2"]')
    h1 = compute_data_fingerprint(
        cache_dir=tmp_path / "cache",
        extra_files={"sample_doc_ids": extra},
    )

    extra.write_text('["d1","d2","d3"]')
    h2 = compute_data_fingerprint(
        cache_dir=tmp_path / "cache",
        extra_files={"sample_doc_ids": extra},
    )
    assert h1 != h2


def test_compute_data_fingerprint_missing_extra_file_does_not_raise(tmp_path: Path):
    """A None / non-existent extra-file path must NOT raise — it folds in
    as null so the fingerprint still serialises. Avoids fail-on-disk
    during commit-3 transition before all runners have wired the path."""
    h = compute_data_fingerprint(
        cache_dir=tmp_path / "cache",
        extra_files={"sample_doc_ids": tmp_path / "absent.json", "other": None},
    )
    assert len(h) == 64


def test_compute_data_fingerprint_extra_files_key_order_stable(tmp_path: Path):
    """Insertion order of extra_files dict must NOT change the hash —
    sorted internally."""
    f1 = tmp_path / "a.json"
    f1.write_text("[]")
    f2 = tmp_path / "b.json"
    f2.write_text("[]")
    h_ab = compute_data_fingerprint(
        cache_dir=tmp_path,
        extra_files={"alpha": f1, "beta": f2},
    )
    h_ba = compute_data_fingerprint(
        cache_dir=tmp_path,
        extra_files={"beta": f2, "alpha": f1},
    )
    assert h_ab == h_ba


# --------------------------------------------------------------------------- #
# Env fingerprint — stable hash of capture_environment()
# --------------------------------------------------------------------------- #


def _sample_env_dict() -> dict:
    """A representative env dict matching capture_environment() shape."""
    return {
        "python": "3.10.13",
        "platform": "darwin",
        "git_commit": "deadbeef1234",
        "cpu": {"brand": "Apple M2 Pro", "logical_count": 10},
        "mem_gb": 16.0,
        "packages": {"bm25s": "0.2.4", "torch": "2.3.1", "numpy": "1.26.4"},
    }


def test_compute_env_fingerprint_is_64_hex():
    h = compute_env_fingerprint(_sample_env_dict())
    assert len(h) == 64
    int(h, 16)


def test_compute_env_fingerprint_deterministic():
    a = compute_env_fingerprint(_sample_env_dict())
    b = compute_env_fingerprint(_sample_env_dict())
    assert a == b


def test_compute_env_fingerprint_insensitive_to_key_order():
    base = _sample_env_dict()
    reordered = {
        "packages": dict(reversed(list(base["packages"].items()))),
        "mem_gb": base["mem_gb"],
        "cpu": {"logical_count": base["cpu"]["logical_count"], "brand": base["cpu"]["brand"]},
        "git_commit": base["git_commit"],
        "platform": base["platform"],
        "python": base["python"],
    }
    assert compute_env_fingerprint(base) == compute_env_fingerprint(reordered)


def test_compute_env_fingerprint_sensitive_to_package_change():
    """A package version bump must change the env fingerprint — otherwise
    reproducibility audits silently miss the env drift."""
    base = _sample_env_dict()
    h_base = compute_env_fingerprint(base)

    bumped = _sample_env_dict()
    bumped["packages"]["torch"] = "2.4.0"
    assert compute_env_fingerprint(bumped) != h_base


def test_compute_env_fingerprint_sensitive_to_python_change():
    base = _sample_env_dict()
    h_base = compute_env_fingerprint(base)
    bumped = _sample_env_dict()
    bumped["python"] = "3.11.0"
    assert compute_env_fingerprint(bumped) != h_base


def test_compute_env_fingerprint_handles_none_fields():
    """capture_environment() returns ``None`` for unknown cpu brand / git
    commit / mem_gb. The fingerprint must accept those without raising."""
    env = {
        "python": "3.10.0",
        "platform": "linux",
        "git_commit": None,
        "cpu": {"brand": None, "logical_count": 4},
        "mem_gb": None,
        "packages": {},
    }
    h = compute_env_fingerprint(env)
    assert len(h) == 64


# --------------------------------------------------------------------------- #
# Sampling caveat block
# --------------------------------------------------------------------------- #


def test_compute_sampling_block_full_corpus_is_minimal():
    """is_sampled=False produces a minimal 1-key dict; no caveat / method /
    sample_size pollution. Full-corpus runs must NOT carry a sampling
    warning, otherwise it dilutes the warning's signal when it fires."""
    assert compute_sampling_block(is_sampled=False) == {"is_sampled": False}


def test_compute_sampling_block_sampled_default_uses_canonical_caveat():
    """is_sampled=True with no overrides uses the canonical caveat string
    and the default 'qrels-anchored' method label."""
    block = compute_sampling_block(is_sampled=True, sample_size=50000)
    assert block["is_sampled"] is True
    assert block["method"] == "qrels-anchored"
    assert block["sample_size"] == 50000
    assert block["caveat"] == CANONICAL_SAMPLED_CAVEAT


def test_compute_sampling_block_canonical_caveat_mentions_critical_terms():
    """The canonical caveat must contain the load-bearing honest phrases:
    'qrels-anchored', upper-bound, and 'not comparable to full-corpus'.
    Drift in the wording silently weakens the published warning."""
    text = CANONICAL_SAMPLED_CAVEAT.lower()
    assert "qrels-anchored" in text
    assert "upper-bound" in text
    assert "not comparable to full-corpus" in text


def test_compute_sampling_block_custom_method_label():
    """method= overrides the 'qrels-anchored' default — used by the BM25
    runner's --corpus-limit smoke path (which is first-N-truncated, not
    qrels-anchored)."""
    block = compute_sampling_block(
        is_sampled=True, method="first-N-truncated", sample_size=200000
    )
    assert block["method"] == "first-N-truncated"
    assert block["sample_size"] == 200000


def test_compute_sampling_block_custom_caveat_overrides():
    """caveat= lets a runner emit a non-default warning when the sampling
    scheme is sufficiently unusual that the canonical wording would be
    misleading. The override is wholesale, not appended."""
    custom = "Custom warning text for an exotic sampling scheme."
    block = compute_sampling_block(
        is_sampled=True, sample_size=1234, caveat=custom
    )
    assert block["caveat"] == custom


def test_compute_sampling_block_sample_size_optional():
    """sample_size=None is valid — used by the generation runner which
    inherits sampling context from upstream retrieval but has no
    sample_size of its own."""
    block = compute_sampling_block(is_sampled=True, sample_size=None)
    assert block["sample_size"] is None
    assert block["is_sampled"] is True


def test_compute_sampling_block_keys_are_stable_set():
    """The 4-key shape on is_sampled=True is the locked contract;
    downstream report tooling will key off these names. Pinning the set
    catches accidental renames."""
    block = compute_sampling_block(is_sampled=True, sample_size=50000)
    assert set(block.keys()) == {"is_sampled", "method", "sample_size", "caveat"}
