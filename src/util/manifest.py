"""Build a lightweight per-run artifact manifest.

A *manifest* is a small JSON blob written alongside experiment outputs that
captures enough provenance to re-identify a run six months from now:

- Git commit (short SHA + dirty flag).
- Command line that produced the artifacts.
- Config files used, and a content hash of each.
- Dependency files (requirements / lockfile) with content hash.
- Output paths and per-file sizes.
- A wall-clock timestamp.

All four ``experiments/run_*.py`` runners write a manifest via
``write_run_manifest`` (the convenience wrapper at the bottom of this
file) after their ``metrics.json``. ``build_manifest`` is the lower-level
constructor; ``write_run_manifest`` is the standard entry point.

Manifests deliberately exclude:

- Absolute paths under the developer's home (CLAUDE.md privacy rule).
- API tokens or cache prefixes.
- The contents of large output files (only their size/hash is stored).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Git
# --------------------------------------------------------------------------- #


def _git_info() -> dict[str, Any]:
    """Return git commit short SHA + dirty flag, or empties on failure.

    Never raises — manifests should still be writable when the repo is in
    an unusual state (detached HEAD, no `.git`, etc.).
    """
    info: dict[str, Any] = {"commit": None, "dirty": None}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode().strip() or None
    except Exception:  # noqa: BLE001
        pass
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode().strip()
        info["dirty"] = bool(status)
    except Exception:  # noqa: BLE001
        pass
    return info


# --------------------------------------------------------------------------- #
# File hashing
# --------------------------------------------------------------------------- #


def _file_hash(path: Path, algo: str = "sha256", chunk_size: int = 1 << 20) -> str | None:
    """Return the hex digest of ``path``, or ``None`` if unreadable."""
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.new(algo)
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def _file_record(path: Path, project_root: Path) -> dict[str, Any]:
    """Summarise a single file: relative path, size, sha256 (truncated)."""
    rel = path.relative_to(project_root) if path.is_relative_to(project_root) else path
    record: dict[str, Any] = {"path": str(rel)}
    if path.exists() and path.is_file():
        record["size_bytes"] = path.stat().st_size
        digest = _file_hash(path)
        if digest:
            # Truncated 16-char hash is enough to spot accidental changes; the
            # full digest balloons the manifest without practical benefit.
            record["sha256_16"] = digest[:16]
    else:
        record["exists"] = False
    return record


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def build_manifest(
    *,
    project_root: Path,
    command: Iterable[str] | None = None,
    config_paths: Iterable[Path] = (),
    dependency_paths: Iterable[Path] = (),
    output_paths: Iterable[Path] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the manifest dict.

    Parameters
    ----------
    project_root :
        Used to normalise all paths to repo-relative form (privacy rule).
    command :
        The argv-style command that produced the run. Defaults to ``sys.argv``.
    config_paths :
        Config files consumed by the run (e.g. ``configs/baseline.yaml``).
    dependency_paths :
        Dep declarations whose hash should be captured
        (e.g. ``requirements.txt``, ``requirements-lock.txt``).
    output_paths :
        Artifacts produced by the run (``run.tsv``, ``metrics.json``, ...).
    extra :
        Free-form fields the caller wants to attach (e.g. dataset name,
        rerank depth, eval-query count).
    """
    cmd = list(command) if command is not None else list(sys.argv)
    return {
        "schema": "msmarco-genqa.manifest.v1",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git": _git_info(),
        "command": cmd,
        "config": [_file_record(Path(p), project_root) for p in config_paths],
        "dependencies": [_file_record(Path(p), project_root) for p in dependency_paths],
        "outputs": [_file_record(Path(p), project_root) for p in output_paths],
        "python": {
            "version": sys.version.split()[0],
            "executable": Path(sys.executable).name,  # name only — full path is host-specific
            "platform": sys.platform,
        },
        "extra": dict(extra) if extra else {},
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> Path:
    """Write the manifest as pretty-printed JSON. Returns ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=False, default=str)
        f.write("\n")
    logger.info("Wrote manifest to %s", path)
    return path


# --------------------------------------------------------------------------- #
# Convenience: standard runner entry point
# --------------------------------------------------------------------------- #


def write_run_manifest(
    *,
    project_root: Path,
    output_dir: Path,
    command: Iterable[str] | None = None,
    config_path: Path | None = None,
    extra_outputs: Iterable[Path] = (),
    extra: dict[str, Any] | None = None,
    manifest_name: str = "manifest.json",
) -> Path:
    """Standardised manifest writer for the experiment runners.

    All four ``experiments/run_*.py`` scripts call this exactly once, after
    they've written ``metrics.json``. It captures the inputs (config +
    dependency files) and the outputs (``metrics.json`` plus whatever the
    caller passes via ``extra_outputs``) into ``output_dir / manifest.json``.

    Conventions baked in:

    - The manifest path is ``output_dir / manifest.json`` unless
      ``manifest_name`` is overridden.
    - ``output_dir / 'metrics.json'`` is always included as a captured output.
    - ``requirements.txt``, ``requirements-lock.txt``, and ``pyproject.toml``
      at the repo root are auto-included as dependency paths if present —
      callers don't need to enumerate them.
    - All paths land in the manifest as repo-relative (privacy rule:
      we never paste absolute home paths into committed JSON).
    """
    metrics_path = output_dir / "metrics.json"

    # Auto-discover dependency declarations at the repo root. Order matters
    # for readability: the lockfile is what reproduces, requirements.txt is
    # the loose dev install, pyproject.toml is the package metadata.
    dep_candidates = [
        project_root / "requirements-lock.txt",
        project_root / "requirements.txt",
        project_root / "pyproject.toml",
    ]
    deps = [p for p in dep_candidates if p.exists()]

    config_paths = [config_path] if config_path is not None else []
    outputs = [metrics_path, *extra_outputs]

    manifest = build_manifest(
        project_root=project_root,
        command=command,
        config_paths=config_paths,
        dependency_paths=deps,
        output_paths=outputs,
        extra=extra,
    )
    return write_manifest(manifest, output_dir / manifest_name)
