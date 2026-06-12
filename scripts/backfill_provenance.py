#!/usr/bin/env python3
"""Backfill ``provenance.backfill.json`` for canonical baseline runs.

Three canonical baseline runs — W2 BM25 full-corpus, W4 dense baseline
(50k sample), W5 reranker over the full BM25 run — predate the manifest
plumbing added later in this project. The output directories on disk
therefore lack a runtime ``manifest.json``. This script writes a
*backfilled* provenance file next to each of those three output
directories.

The output of this script is deliberately NOT a runtime manifest:

* The filename is ``provenance.backfill.json`` (not ``manifest.json``).
* The schema string is ``msmarco-genqa.backfilled-provenance.v1``
  (not ``msmarco-genqa.manifest.v1``).
* An ``unknown`` block enumerates, key-by-key, which dimensions of
  runtime provenance are unrecoverable for these specific runs.

Visually distinguishing the backfill file from a real manifest is a
load-bearing requirement of the design: a reviewer who confuses the two
would conclude that the run is more reproducible than it actually is.

Usage::

    python3 scripts/backfill_provenance.py
    python3 scripts/backfill_provenance.py --anchor-tag v1.0-first-report

The script is idempotent: re-running it overwrites the JSON in place
and never lists the previously-written ``provenance.backfill.json`` as
one of the outputs it captures.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger("backfill_provenance")

SCHEMA = "msmarco-genqa.backfilled-provenance.v1"
PRODUCED_BY = "scripts/backfill_provenance.py"

DEFAULT_TARGETS = [
    "outputs/W2_bm25",
    "outputs/W4_dense",
    "outputs/W5_reranker",
]
CONFIG_PATH = "configs/baseline.yaml"

# The seeding patch referenced by ``unknown.production_random_seed_effectiveness``.
# Hardcoded here on purpose: this footnote is the file's load-bearing piece, so
# it should not depend on a runtime git lookup that could fail silently.
SEEDING_PATCH_COMMIT_SHORT = "4534f31"
SEEDING_PATCH_BRANCH = "infra/reproducibility-round1"


def _resolve_short_sha(commit_ref: str, *, project_root: Path) -> str:
    """Resolve ``commit_ref`` to its 12-char short SHA via ``git rev-parse``."""
    out = subprocess.check_output(
        ["git", "rev-parse", "--short=12", commit_ref],
        cwd=project_root,
        text=True,
        timeout=5,
    ).strip()
    return out


def _sha256_hex_16(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _file_sha256_hex_16(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _config_at_commit_hash(
    *, commit: str, config_path: str, project_root: Path
) -> str:
    """Hash ``config_path`` at ``commit`` via ``git show`` (no checkout)."""
    blob = subprocess.check_output(
        ["git", "show", f"{commit}:{config_path}"],
        cwd=project_root,
        timeout=10,
    )
    return _sha256_hex_16(blob)


def _outputs_on_disk(output_dir: Path, project_root: Path) -> list[dict[str, Any]]:
    """Hash each file in ``output_dir`` (one level deep). Skip ourselves."""
    records: list[dict[str, Any]] = []
    for child in sorted(output_dir.iterdir()):
        if child.is_dir():
            continue
        # Idempotence: do NOT list the file we are about to (re)write.
        if child.name == "provenance.backfill.json":
            continue
        rel = (
            child.relative_to(project_root)
            if child.is_relative_to(project_root)
            else child
        )
        records.append(
            {
                "path": str(rel),
                "sha256_16": _file_sha256_hex_16(child),
                "size_bytes": child.stat().st_size,
            }
        )
    return records


def _unknown_block(anchor_commit: str) -> dict[str, str]:
    """The eight per-key explanations for unrecoverable runtime provenance.

    The set of keys here is a public part of the schema; tests assert
    that every key is present.
    """
    return {
        "production_commit": (
            "May be earlier than the anchor commit; the history reaching "
            "the tag is lossy. Outputs could have been generated on any "
            f"of several commits leading up to {anchor_commit}."
        ),
        "production_command_line": (
            "No record of the CLI argv used at runtime (e.g. whether "
            "--num-eval-queries was set explicitly, whether --resume was "
            "used)."
        ),
        "production_timestamp": (
            "Wall-clock time of the original run not recorded."
        ),
        "git_dirty_at_production": (
            "We do not know whether the working tree had uncommitted "
            "changes when the run executed."
        ),
        "python_version_at_production": (
            "Not captured at runtime."
        ),
        "package_versions_at_production": (
            "requirements-lock.txt at the anchor commit is in git, but "
            "its presence does not prove the actual installed environment "
            "matched it."
        ),
        "production_input_files": (
            "The ir_datasets cache, BM25 index, and dense FAISS index are "
            "gitignored and could have been rebuilt since this run. We "
            "cannot verify byte-identity with whatever the production "
            "runner consumed as input."
        ),
        "production_random_seed_effectiveness": (
            "configs/baseline.yaml sets seed: 42 and the runner calls "
            "random.seed(seed). However, at the anchor commit "
            f"({anchor_commit}) the runner did NOT call np.random.seed, "
            "torch.manual_seed, transformers.set_seed, or set "
            "torch.backends.cudnn.deterministic. The unified seeding "
            "helper (src/msmarco_genqa/util/seeding.py) added in "
            f"{SEEDING_PATCH_BRANCH} commit {SEEDING_PATCH_COMMIT_SHORT} "
            "is the patch that retroactively makes these knowable for "
            "future runs; for this backfilled run, the effective seeding "
            "coverage at runtime is unknown beyond Python's stdlib "
            "`random` module."
        ),
    }


def build_backfilled_provenance(
    *,
    output_dir: Path,
    anchor_tag: str,
    anchor_commit: str,
    config_path: str,
    config_sha_16: str,
    project_root: Path,
    now_utc: dt.datetime | None = None,
) -> dict[str, Any]:
    """Build the backfilled-provenance dict (pure: no git, no I/O writes).

    Parameters
    ----------
    output_dir :
        Directory whose files are hashed and listed under
        ``outputs_on_disk_now``.
    anchor_tag, anchor_commit :
        The git anchor (typically ``v1.0-first-report``) and its
        resolved short SHA.
    config_path, config_sha_16 :
        Path-at-anchor of the config file and its truncated sha256.
    project_root :
        Used to normalise output paths to repo-relative form.
    now_utc :
        Optional override for the ``backfill_created_at`` timestamp,
        useful for deterministic tests.
    """
    ts = now_utc if now_utc is not None else dt.datetime.now(dt.timezone.utc)
    return {
        "schema": SCHEMA,
        "note": (
            "BACKFILLED retroactively, not a runtime manifest. Generated "
            f"by {PRODUCED_BY}."
        ),
        "backfill_created_at": ts.isoformat(timespec="seconds"),
        "produced_by": PRODUCED_BY,
        "anchor": {
            "tag": anchor_tag,
            "commit": anchor_commit,
            "note": (
                "This output directory existed at this commit. The actual "
                "production commit may be EARLIER in the history reaching "
                "this tag; we have no way to determine which one."
            ),
        },
        "config_at_anchor": {
            "path": config_path,
            "sha256_16": config_sha_16,
            "comment": (
                f"Hash of {config_path} at the anchor commit. We cannot "
                "verify the runner actually used this exact config; we "
                "only know it was in the tree."
            ),
        },
        "outputs_on_disk_now": _outputs_on_disk(output_dir, project_root),
        "unknown": _unknown_block(anchor_commit),
    }


def write_backfill_for_dir(
    *,
    output_dir: Path,
    anchor_tag: str,
    anchor_commit: str,
    config_path: str,
    config_sha_16: str,
    project_root: Path,
) -> Path:
    """Build + write the backfill JSON for one output directory.

    Returns the path written.
    """
    doc = build_backfilled_provenance(
        output_dir=output_dir,
        anchor_tag=anchor_tag,
        anchor_commit=anchor_commit,
        config_path=config_path,
        config_sha_16=config_sha_16,
        project_root=project_root,
    )
    out = output_dir / "provenance.backfill.json"
    with open(out, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=False)
        f.write("\n")
    logger.info("Wrote %s", out)
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchor-tag",
        default="v1.0-first-report",
        help=(
            "Git tag treated as the latest-possible commit at which the "
            "targeted outputs existed (default: v1.0-first-report)."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repo root. Defaults to the parent of scripts/.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        help=(
            "Override the target output dirs (repo-relative). Repeatable. "
            "Defaults to the three canonical baseline dirs."
        ),
    )
    args = parser.parse_args()

    anchor_commit = _resolve_short_sha(
        args.anchor_tag, project_root=args.project_root
    )
    config_sha_16 = _config_at_commit_hash(
        commit=anchor_commit,
        config_path=CONFIG_PATH,
        project_root=args.project_root,
    )

    logger.info("anchor tag=%s commit=%s", args.anchor_tag, anchor_commit)
    logger.info(
        "config %s @ %s sha256_16=%s", CONFIG_PATH, anchor_commit, config_sha_16
    )

    targets = args.target if args.target is not None else DEFAULT_TARGETS
    for rel in targets:
        d = args.project_root / rel
        if not d.is_dir():
            logger.warning("Skipping %s (directory does not exist)", d)
            continue
        write_backfill_for_dir(
            output_dir=d,
            anchor_tag=args.anchor_tag,
            anchor_commit=anchor_commit,
            config_path=CONFIG_PATH,
            config_sha_16=config_sha_16,
            project_root=args.project_root,
        )


if __name__ == "__main__":
    main()
