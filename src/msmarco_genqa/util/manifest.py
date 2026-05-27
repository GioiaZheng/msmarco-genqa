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

Schema v2 — runtime contract
----------------------------

Schema bumped to ``msmarco-genqa.manifest.v2``. The bump is a hard break
of v1: writes always emit v2; v1 manifests on disk remain readable as
plain JSON for archaeological purposes. The behaviour change is the
*runtime contract*: a manifest write under default (strict) mode fails
with ``RequiredFieldMissingError`` if any of ``REQUIRED_FIELDS`` is
missing or ``None``. Runners expose ``--allow-incomplete-manifest`` as
the dev-time bypass, symmetric to ``--require-clean-tree`` from v1.
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


SCHEMA_VERSION = "msmarco-genqa.manifest.v2"

# Required-field contract enforced by ``write_run_manifest`` under default
# (strict) mode. Dotted paths into the manifest dict. A value of ``None`` or
# a missing key both count as a violation; ``False`` and ``0`` are valid.
#
# Field meanings:
# - git.commit       : short SHA of HEAD at write time (non-None).
# - git.dirty        : bool (True iff uncommitted changes; never None).
# - extra.seed       : the seed passed to ``set_global_seed``.
# - extra.resolved_config_hash : content hash of the resolved config dict
#                                (after CLI overrides). Populated by
#                                commit 2 of research/reproducibility-protocol.
# - extra.data_fingerprint     : lean hash of corpus cache + qrels.
#                                Populated by commit 3.
# - extra.env_fingerprint      : stable hash of capture_environment().
#                                Populated by commit 3.
REQUIRED_FIELDS: tuple[str, ...] = (
    "git.commit",
    "git.dirty",
    "extra.seed",
    "extra.resolved_config_hash",
    "extra.data_fingerprint",
    "extra.env_fingerprint",
)


class DirtyTreeError(RuntimeError):
    """Raised by ``write_run_manifest`` when ``require_clean_tree=True`` and
    the git working tree has uncommitted changes.

    The recorded commit SHA alone is not sufficient to reproduce a run made
    from a dirty tree, so canonical / headline runs may opt in to this check.
    """


class RequiredFieldMissingError(RuntimeError):
    """Raised by ``write_run_manifest`` under default (strict) mode when one
    or more ``REQUIRED_FIELDS`` is missing or ``None``.

    The runtime contract for v2: the recorded fields must be sufficient to
    re-identify and reproduce the run. Missing fields silently turn the
    manifest into archaeology, so the default behaviour refuses to write.
    Pass ``allow_incomplete=True`` (from runners:
    ``--allow-incomplete-manifest``) to bypass during development.
    """


_MISSING = object()


def _get_dotted(d: dict, dotted: str):
    """Return the value at ``dotted`` path inside ``d``, or ``_MISSING``.

    A returned ``_MISSING`` means "key absent at some level"; a returned
    ``None`` means "present but null". The validator treats both as
    violations.
    """
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def _validate_required(manifest: dict[str, Any]) -> None:
    """Raise ``RequiredFieldMissingError`` if any ``REQUIRED_FIELDS`` entry
    is missing or ``None`` in ``manifest``. Returns ``None`` on success.

    The error message enumerates every violating field so the user can fix
    them in one pass rather than discovering them one at a time.
    """
    missing: list[str] = []
    for field in REQUIRED_FIELDS:
        value = _get_dotted(manifest, field)
        if value is _MISSING or value is None:
            missing.append(field)
    if missing:
        raise RequiredFieldMissingError(
            f"manifest is missing required field(s) {missing}. The write "
            f"was refused under the {SCHEMA_VERSION} runtime contract. "
            "Populate the missing field(s) via the runner's extra dict, "
            "or pass --allow-incomplete-manifest at the CLI to bypass "
            "(development only)."
        )


# --------------------------------------------------------------------------- #
# Resolved config — content hash + adjacent YAML artifact
# --------------------------------------------------------------------------- #
#
# The "resolved config" is the cfg dict AFTER all CLI overrides have been
# applied and is the actual config that drove the run. The file-level
# sha256 of configs/baseline.yaml that already lives in manifest["config"][0]
# is NOT sufficient: it misses --sample-size, --model-name, and similar
# runner CLI overrides that meaningfully change the run. The contract is
# that the runner hashes the resolved dict, writes the dict to
# output_dir/resolved_config.yaml, and passes the hash via extra so the
# manifest's required-fields validator sees it.


def compute_resolved_config_hash(cfg: dict[str, Any]) -> str:
    """Return a stable sha256 hex digest of the resolved config dict.

    Stability properties:
    - Insensitive to key insertion order (uses json.dumps sort_keys=True).
    - Sensitive to any value change at any depth.
    - Pure function: same dict in, same hash out, no side effects.

    The 64-char full digest is returned (not the truncated 16-char form
    used in manifest["config"][...] file records), because this hash is
    the canonical re-identifier for the run's logical config: shorter
    digests have non-trivial collision risk across the lifetime of the
    project's run registry.
    """
    serialised = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def write_resolved_config(cfg: dict[str, Any], output_dir: Path) -> Path:
    """Write the resolved config dict to ``output_dir/resolved_config.yaml``.

    Returns the written path. ``output_dir`` is created if needed. The
    YAML is written with ``default_flow_style=False`` and ``sort_keys=True``
    so the on-disk form matches the hash input (key-order-stable). YAML
    is preferred over JSON for the on-disk form because configs/baseline.yaml
    is itself YAML — keeping the resolved-config artifact in the same
    surface dialect makes diffing trivial.
    """
    import yaml  # imported lazily; yaml is a runtime dep but kept out of the manifest module's hot path

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "resolved_config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=True)
    return path


# --------------------------------------------------------------------------- #
# Data fingerprint — lean (per 2026-05-27 design lock)
# --------------------------------------------------------------------------- #
#
# Lean per locked design: cache_dir str + corpus_limit + per-run extra
# input files (sample_doc_ids JSON for dense, input run.tsv for
# reranker/generation) content-hashed. We do NOT hash the 8.8M-passage
# corpus body — its identity is anchored by the cache_dir path + the
# ir_datasets dataset name + corpus_limit; rehashing the body on every
# run is wasteful and download-order-dependent.


def compute_data_fingerprint(
    *,
    cache_dir: Path,
    corpus_limit: int | None = None,
    extra_files: dict[str, Path] | None = None,
) -> str:
    """Lean sha256 hex digest identifying the data inputs of a run.

    Components:
    - ``cache_dir`` as string. Anchors which ir_datasets cache served
      the corpus/queries/qrels for this run.
    - ``corpus_limit``: scalar, ``None`` for the full corpus.
    - ``extra_files``: optional ``{label: Path}`` mapping for run-specific
      inputs that should be content-hashed — e.g. ``sample_doc_ids.json``
      for dense, ``input run.tsv`` for reranker/generation. Each
      file's truncated 16-char sha256 (matching the manifest file-record
      convention) is folded in; ``None`` is recorded for missing files
      so the fingerprint distinguishes "ran without this input" from
      "ran with a present but possibly empty input".

    Returns the 64-char full sha256 hex of the canonical JSON encoding
    (sort_keys=True), matching the rigor of ``compute_resolved_config_hash``.
    """
    parts: dict[str, Any] = {
        "cache_dir": str(cache_dir),
        "corpus_limit": corpus_limit,
    }
    if extra_files:
        for label, raw_path in sorted(extra_files.items()):
            if raw_path is None:
                parts[label] = None
                continue
            path = Path(raw_path)
            if path.exists() and path.is_file():
                # Match manifest file-record truncated-digest convention
                # (16-char is enough to spot accidental changes; the
                # fingerprint as a whole is still 64-char).
                full_digest = _file_hash(path)
                parts[label] = full_digest[:16] if full_digest else None
            else:
                parts[label] = None
    serialised = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


# --------------------------------------------------------------------------- #
# Env fingerprint — stable hash of capture_environment()
# --------------------------------------------------------------------------- #


def compute_env_fingerprint(env_dict: dict[str, Any]) -> str:
    """Return a 64-char sha256 hex of a captured-environment dict.

    Input is the dict returned by
    ``msmarco_genqa.util.environment.capture_environment()``. The hash
    is stable across calls with identical input and across Python
    dict-iteration orders (json.dumps sort_keys=True).

    Sensitive to package version changes, python version changes, cpu
    brand, mem_gb — anything that ``capture_environment`` records.
    Insensitive to call-time noise (the environment dict has no
    timestamps or wall-clock fields).
    """
    serialised = json.dumps(env_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


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
        "schema": SCHEMA_VERSION,
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
    require_clean_tree: bool = False,
    allow_incomplete: bool = False,
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

    Dirty-tree handling: if the git working tree has uncommitted changes,
    the recorded commit SHA alone is not sufficient to reproduce the run.
    By default this only emits a ``logger.warning``. Pass
    ``require_clean_tree=True`` (from runners: ``--require-clean-tree``)
    to refuse to write the manifest in that case — useful for canonical /
    headline runs where the recorded provenance must be tight.

    Required-field contract (schema v2): under default ``allow_incomplete=False``,
    the manifest is validated against ``REQUIRED_FIELDS`` before write and
    a ``RequiredFieldMissingError`` is raised if any field is missing or
    ``None``. Pass ``allow_incomplete=True`` (from runners:
    ``--allow-incomplete-manifest``) to bypass during development — the
    bypass is symmetric to ``require_clean_tree`` in role: development
    convenience that must be deliberately opted in.
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

    is_dirty = manifest["git"].get("dirty") is True
    manifest_path = output_dir / manifest_name

    if is_dirty and require_clean_tree:
        raise DirtyTreeError(
            f"refusing to write manifest at {manifest_path}: git working "
            "tree has uncommitted changes and require_clean_tree=True. "
            "Commit your changes and rerun, or omit --require-clean-tree."
        )

    if not allow_incomplete:
        _validate_required(manifest)

    path = write_manifest(manifest, manifest_path)

    if is_dirty:
        logger.warning(
            "Wrote %s from a DIRTY git tree (commit %s + uncommitted "
            "changes). The recorded commit alone is NOT sufficient to "
            "reproduce this run.",
            path,
            manifest["git"].get("commit"),
        )

    return path
