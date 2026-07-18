"""Validate the canonical artifact registry and all tracked references."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any


REGISTRY_SCHEMA = "msmarco-genqa.artifact-registry.v1"
TABLE_ARTIFACT_SCHEMA = "msmarco-genqa.table-artifact.v1"
POINTER_SCHEMA = "msmarco-genqa.external-artifact-pointer.v1"
HEX_40 = re.compile(r"^[0-9a-f]{40}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
ENTRY_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
MANIFEST_AVAILABILITY = {"tracked", "local_only", "not_preserved"}
PROVENANCE_STATUS = {"exact", "historical_partial"}
CONFIG_RUNTIME_USE = {"unknown", "exact_local_artifact"}
LOCKFILE_STATUS = {"not_present", "repository_snapshot"}


def normalized_bytes_sha256(data: bytes) -> str:
    """Hash text bytes after normalizing CRLF/CR to LF."""
    normalized = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def normalized_text_sha256(path: Path) -> str:
    """Hash text bytes after normalizing CRLF/CR to LF."""
    return normalized_bytes_sha256(path.read_bytes())


class GitHistory:
    """Read immutable evidence from a local Git object database."""

    def __init__(self, project_root: Path, executable: str) -> None:
        self.project_root = project_root
        self.executable = executable
        self._commit_cache: dict[str, bool] = {}
        self._file_cache: dict[tuple[str, str], bytes | None] = {}

    def _run(self, *args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [self.executable, "-C", str(self.project_root), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def commit_exists(self, commit: str) -> bool:
        if commit not in self._commit_cache:
            result = self._run("cat-file", "-e", f"{commit}^{{commit}}")
            self._commit_cache[commit] = result.returncode == 0
        return self._commit_cache[commit]

    def resolve_commit(self, ref: str) -> str | None:
        result = self._run("rev-parse", "--verify", f"{ref}^{{commit}}")
        if result.returncode != 0:
            return None
        return result.stdout.decode("ascii").strip()

    def file_bytes(self, commit: str, path: str) -> bytes | None:
        key = (commit, path)
        if key not in self._file_cache:
            result = self._run("show", f"{commit}:{path}")
            self._file_cache[key] = result.stdout if result.returncode == 0 else None
        return self._file_cache[key]


def _load_json(path: Path, *, label: str, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label}: cannot read JSON: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label}: expected a JSON object")
        return None
    return value


def _resolve_path(
    project_root: Path,
    value: Any,
    *,
    label: str,
    errors: list[str],
    must_exist: bool,
) -> Path | None:
    if not isinstance(value, str) or not value or "\\" in value:
        errors.append(f"{label}: path must be a non-empty repository-relative POSIX path")
        return None
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        errors.append(f"{label}: unsafe repository-relative path {value!r}")
        return None
    path = project_root.joinpath(*pure.parts)
    try:
        path.resolve().relative_to(project_root)
    except ValueError:
        errors.append(f"{label}: path escapes the repository: {value!r}")
        return None
    if must_exist and not path.is_file():
        errors.append(f"{label}: referenced file does not exist: {value}")
    return path


def _require_hex(value: Any, pattern: re.Pattern[str], *, label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        errors.append(f"{label}: expected lowercase {pattern.pattern[1:-1]}")


def _check_hashed_json(
    record: Any,
    *,
    project_root: Path,
    label: str,
    expected_schema: str,
    errors: list[str],
) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        errors.append(f"{label}: expected an object")
        return None
    path = _resolve_path(
        project_root,
        record.get("path"),
        label=f"{label}.path",
        errors=errors,
        must_exist=True,
    )
    expected_hash = record.get("sha256_lf")
    _require_hex(expected_hash, HEX_64, label=f"{label}.sha256_lf", errors=errors)
    if path is None or not path.is_file():
        return None
    actual_hash = normalized_text_sha256(path)
    if isinstance(expected_hash, str) and actual_hash != expected_hash:
        errors.append(
            f"{label}: sha256_lf mismatch for {record.get('path')}; "
            f"expected {expected_hash}, got {actual_hash}"
        )
    payload = _load_json(path, label=label, errors=errors)
    if payload is None:
        return None
    if record.get("schema") != expected_schema:
        errors.append(f"{label}.schema: expected {expected_schema!r}")
    if payload.get("schema") != expected_schema:
        errors.append(f"{label}: referenced JSON does not use {expected_schema!r}")
    expected_id = record.get("artifact_id")
    if not isinstance(expected_id, str) or payload.get("artifact_id") != expected_id:
        errors.append(f"{label}: artifact_id does not match the referenced JSON")
    return payload


def _check_commit(
    value: Any,
    *,
    label: str,
    errors: list[str],
    git_history: GitHistory | None = None,
) -> None:
    _require_hex(value, HEX_40, label=label, errors=errors)
    if (
        git_history is not None
        and isinstance(value, str)
        and HEX_40.fullmatch(value)
        and not git_history.commit_exists(value)
    ):
        errors.append(f"{label}: commit is not available in Git history: {value}")


def _check_provenance(
    provenance: Any,
    *,
    entry_label: str,
    project_root: Path,
    artifact_payload: dict[str, Any] | None,
    errors: list[str],
    git_history: GitHistory | None,
) -> None:
    label = f"{entry_label}.provenance"
    if not isinstance(provenance, dict):
        errors.append(f"{label}: expected an object")
        return
    status = provenance.get("status")
    if status not in PROVENANCE_STATUS:
        errors.append(f"{label}.status: expected one of {sorted(PROVENANCE_STATUS)}")
    production_commit = provenance.get("production_commit")
    if status == "exact":
        _check_commit(
            production_commit,
            label=f"{label}.production_commit",
            errors=errors,
            git_history=git_history,
        )
        if artifact_payload is not None:
            recorded_commit = (artifact_payload.get("experiment") or {}).get("git_commit")
            if recorded_commit != production_commit:
                errors.append(
                    f"{label}: exact production commit does not match artifact experiment.git_commit"
                )
    elif status == "historical_partial" and production_commit is not None:
        errors.append(f"{label}: historical_partial production_commit must be null")

    evidence_commits = provenance.get("evidence_commits")
    if not isinstance(evidence_commits, list) or not evidence_commits:
        errors.append(f"{label}.evidence_commits: expected a non-empty list")
    else:
        for index, commit in enumerate(evidence_commits):
            _check_commit(
                commit,
                label=f"{label}.evidence_commits[{index}]",
                errors=errors,
                git_history=git_history,
            )

    if status == "historical_partial":
        anchor = provenance.get("report_anchor")
        if not isinstance(anchor, dict) or not isinstance(anchor.get("tag"), str):
            errors.append(f"{label}.report_anchor: historical entries require tag and commit")
        else:
            anchor_commit = anchor.get("commit")
            _check_commit(
                anchor_commit,
                label=f"{label}.report_anchor.commit",
                errors=errors,
                git_history=git_history,
            )
            if git_history is not None:
                resolved = git_history.resolve_commit(anchor["tag"])
                if resolved is None:
                    errors.append(
                        f"{label}.report_anchor.tag: tag is not available in Git history: "
                        f"{anchor['tag']}"
                    )
                elif resolved != anchor_commit:
                    errors.append(
                        f"{label}.report_anchor: tag {anchor['tag']!r} resolves to "
                        f"{resolved}, not {anchor_commit}"
                    )

    manifests = provenance.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        errors.append(f"{label}.manifests: expected a non-empty list")
        return
    for index, manifest in enumerate(manifests):
        item_label = f"{label}.manifests[{index}]"
        if not isinstance(manifest, dict):
            errors.append(f"{item_label}: expected an object")
            continue
        availability = manifest.get("availability")
        if availability not in MANIFEST_AVAILABILITY:
            errors.append(
                f"{item_label}.availability: expected one of {sorted(MANIFEST_AVAILABILITY)}"
            )
            continue
        path = _resolve_path(
            project_root,
            manifest.get("path"),
            label=f"{item_label}.path",
            errors=errors,
            must_exist=availability == "tracked",
        )
        if availability != "tracked" and isinstance(manifest.get("path"), str):
            if not manifest["path"].startswith("outputs/"):
                errors.append(f"{item_label}: untracked run records must live under outputs/")
        if not isinstance(manifest.get("kind"), str) or not manifest["kind"]:
            errors.append(f"{item_label}.kind: expected a non-empty string")
        if path is None:
            continue


def _check_config_snapshots(
    snapshots: Any,
    *,
    entry_label: str,
    project_root: Path,
    errors: list[str],
    git_history: GitHistory | None,
) -> None:
    label = f"{entry_label}.config_snapshots"
    if not isinstance(snapshots, list) or not snapshots:
        errors.append(f"{label}: expected a non-empty list")
        return
    for index, snapshot in enumerate(snapshots):
        item_label = f"{label}[{index}]"
        if not isinstance(snapshot, dict):
            errors.append(f"{item_label}: expected an object")
            continue
        runtime_use = snapshot.get("runtime_use")
        if runtime_use not in CONFIG_RUNTIME_USE:
            errors.append(f"{item_label}.runtime_use: invalid value")
        snapshot_path = snapshot.get("path")
        _resolve_path(
            project_root,
            snapshot_path,
            label=f"{item_label}.path",
            errors=errors,
            must_exist=runtime_use == "unknown" and str(snapshot.get("path", "")).startswith("configs/"),
        )
        commit = snapshot.get("commit")
        digest = snapshot.get("sha256_lf")
        _check_commit(
            commit,
            label=f"{item_label}.commit",
            errors=errors,
            git_history=git_history,
        )
        _require_hex(digest, HEX_64, label=f"{item_label}.sha256_lf", errors=errors)
        if (
            git_history is not None
            and runtime_use == "unknown"
            and isinstance(snapshot_path, str)
            and isinstance(commit, str)
            and HEX_40.fullmatch(commit)
        ):
            historical_bytes = git_history.file_bytes(commit, snapshot_path)
            if historical_bytes is None:
                errors.append(
                    f"{item_label}: {snapshot_path} is not available at commit {commit}"
                )
            elif isinstance(digest, str) and normalized_bytes_sha256(historical_bytes) != digest:
                errors.append(f"{item_label}: sha256_lf does not match the Git snapshot")


def _check_lockfile_snapshots(
    snapshots: Any,
    *,
    entry_label: str,
    errors: list[str],
    git_history: GitHistory | None,
) -> None:
    label = f"{entry_label}.lockfile_snapshots"
    if not isinstance(snapshots, list) or not snapshots:
        errors.append(f"{label}: expected a non-empty list")
        return
    for index, snapshot in enumerate(snapshots):
        item_label = f"{label}[{index}]"
        if not isinstance(snapshot, dict):
            errors.append(f"{item_label}: expected an object")
            continue
        status = snapshot.get("status")
        if status not in LOCKFILE_STATUS:
            errors.append(f"{item_label}.status: invalid value")
        if snapshot.get("path") != "requirements-lock.txt":
            errors.append(f"{item_label}.path: expected 'requirements-lock.txt'")
        commit = snapshot.get("commit")
        _check_commit(
            commit,
            label=f"{item_label}.commit",
            errors=errors,
            git_history=git_history,
        )
        digest = snapshot.get("sha256_lf")
        if status == "not_present":
            if digest is not None:
                errors.append(f"{item_label}.sha256_lf: not_present snapshots require null")
        else:
            _require_hex(digest, HEX_64, label=f"{item_label}.sha256_lf", errors=errors)
        if (
            git_history is not None
            and isinstance(commit, str)
            and HEX_40.fullmatch(commit)
        ):
            historical_bytes = git_history.file_bytes(commit, "requirements-lock.txt")
            if status == "not_present" and historical_bytes is not None:
                errors.append(f"{item_label}: lockfile exists at a not_present snapshot")
            elif status == "repository_snapshot":
                if historical_bytes is None:
                    errors.append(f"{item_label}: lockfile is absent at commit {commit}")
                elif isinstance(digest, str) and normalized_bytes_sha256(historical_bytes) != digest:
                    errors.append(f"{item_label}: sha256_lf does not match the Git snapshot")


def _check_metric_summary(value: Any, *, label: str, errors: list[str]) -> None:
    if not isinstance(value, dict) or not value:
        errors.append(f"{label}: expected a non-empty object")
        return
    for name, metric in value.items():
        if not isinstance(name, str) or not name:
            errors.append(f"{label}: metric names must be non-empty strings")
        if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(metric):
            errors.append(f"{label}.{name}: metric must be a finite number")


def validate_registry(
    registry_path: Path,
    *,
    project_root: Path,
    git_history: GitHistory | None = None,
) -> list[str]:
    """Return every schema, path, and hash inconsistency in a registry."""
    root = project_root.resolve()
    errors: list[str] = []
    registry = _load_json(registry_path, label="registry", errors=errors)
    if registry is None:
        return errors
    if registry.get("schema") != REGISTRY_SCHEMA:
        errors.append(f"registry.schema: expected {REGISTRY_SCHEMA!r}")
    if registry.get("hash_convention") != (
        "sha256_lf normalizes CRLF and CR text newlines to LF before hashing"
    ):
        errors.append("registry.hash_convention: unsupported convention")

    lockfile = registry.get("current_lockfile")
    if not isinstance(lockfile, dict):
        errors.append("registry.current_lockfile: expected an object")
    else:
        lock_path = _resolve_path(
            root,
            lockfile.get("path"),
            label="registry.current_lockfile.path",
            errors=errors,
            must_exist=True,
        )
        expected_hash = lockfile.get("sha256_lf")
        _require_hex(expected_hash, HEX_64, label="registry.current_lockfile.sha256_lf", errors=errors)
        if lock_path is not None and lock_path.is_file() and isinstance(expected_hash, str):
            actual = normalized_text_sha256(lock_path)
            if actual != expected_hash:
                errors.append(
                    "registry.current_lockfile: requirements-lock.txt changed without "
                    "updating its registry hash and reproduction note"
                )
        last_dependency_commit = lockfile.get("last_dependency_change_commit")
        _check_commit(
            last_dependency_commit,
            label="registry.current_lockfile.last_dependency_change_commit",
            errors=errors,
            git_history=git_history,
        )
        if (
            git_history is not None
            and isinstance(last_dependency_commit, str)
            and HEX_40.fullmatch(last_dependency_commit)
            and isinstance(expected_hash, str)
        ):
            historical_bytes = git_history.file_bytes(
                last_dependency_commit, "requirements-lock.txt"
            )
            if historical_bytes is None:
                errors.append(
                    "registry.current_lockfile: lockfile is absent at the last dependency "
                    "change commit"
                )
            elif normalized_bytes_sha256(historical_bytes) != expected_hash:
                errors.append(
                    "registry.current_lockfile: hash differs from the last dependency "
                    "change commit"
                )
        _resolve_path(
            root,
            lockfile.get("reproduction_note"),
            label="registry.current_lockfile.reproduction_note",
            errors=errors,
            must_exist=True,
        )
        validation = lockfile.get("validation")
        if not isinstance(validation, list) or not validation or not all(
            isinstance(command, str) and command for command in validation
        ):
            errors.append("registry.current_lockfile.validation: expected commands")

    entries = registry.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append("registry.entries: expected a non-empty list")
        return errors
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        label = f"registry.entries[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{label}: expected an object")
            continue
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or ENTRY_ID.fullmatch(entry_id) is None:
            errors.append(f"{label}.id: expected a lowercase kebab-case id")
        elif entry_id in seen:
            errors.append(f"{label}.id: duplicate registry id {entry_id!r}")
        else:
            seen.add(entry_id)

        artifact_payload = _check_hashed_json(
            entry.get("artifact_record"),
            project_root=root,
            label=f"{label}.artifact_record",
            expected_schema=TABLE_ARTIFACT_SCHEMA,
            errors=errors,
        )
        if "external_pointer" in entry:
            _check_hashed_json(
                entry.get("external_pointer"),
                project_root=root,
                label=f"{label}.external_pointer",
                expected_schema=POINTER_SCHEMA,
                errors=errors,
            )
        _check_provenance(
            entry.get("provenance"),
            entry_label=label,
            project_root=root,
            artifact_payload=artifact_payload,
            errors=errors,
            git_history=git_history,
        )
        _check_config_snapshots(
            entry.get("config_snapshots"),
            entry_label=label,
            project_root=root,
            errors=errors,
            git_history=git_history,
        )
        _check_lockfile_snapshots(
            entry.get("lockfile_snapshots"),
            entry_label=label,
            errors=errors,
            git_history=git_history,
        )
        _check_metric_summary(
            entry.get("metric_summary"),
            label=f"{label}.metric_summary",
            errors=errors,
        )
        notes = entry.get("notes")
        if not isinstance(notes, list) or not notes or not all(
            isinstance(note, str) and note for note in notes
        ):
            errors.append(f"{label}.notes: expected non-empty note strings")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("artifacts/registry.json"),
        help="Registry JSON path.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to resolve references.",
    )
    parser.add_argument(
        "--git-executable",
        default=None,
        help="Git executable used to verify historical commits and snapshots.",
    )
    args = parser.parse_args(argv)
    git_executable = (
        args.git_executable or os.environ.get("GIT_EXECUTABLE") or shutil.which("git")
    )
    git_history = (
        GitHistory(args.project_root.resolve(), git_executable) if git_executable else None
    )
    errors = validate_registry(
        args.registry,
        project_root=args.project_root,
        git_history=git_history,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    if git_history is None:
        print("Artifact registry checks passed (Git history checks skipped: git not found)")
    else:
        print("Artifact registry checks passed, including Git history")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
