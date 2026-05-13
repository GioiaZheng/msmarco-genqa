"""Tests for ``src.util.manifest``.

The module is small (filesystem + git introspection) and the goal here is
to lock down the schema shape and the privacy-relevant behaviour: no
absolute-home paths, no usernames, no host-specific bits.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.util.manifest import build_manifest, write_manifest, write_run_manifest


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

    assert manifest["schema"] == "msmarco-genqa.manifest.v1"
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
    assert loaded["schema"] == "msmarco-genqa.manifest.v1"


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
    )
    data = json.loads(manifest_path.read_text())
    assert "git" in data
    assert "commit" in data["git"]
    assert "dirty" in data["git"]
