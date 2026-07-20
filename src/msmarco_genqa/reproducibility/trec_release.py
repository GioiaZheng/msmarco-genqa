"""Build, recover, and re-evaluate the public TREC-DL run bundle.

The release bundle deliberately contains only ranked document identifiers,
scores, compact metadata, and checksums. It excludes passage/query text,
dataset mirrors, model caches, and machine-local run manifests.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from msmarco_genqa.data.trec_dl import DEFAULT_REL_THRESHOLD, load_trec_dl
from msmarco_genqa.evaluation.trec import evaluate_internal_trec_scope, read_qrels
from msmarco_genqa.reranking.io import read_run_tsv


BUNDLE_SCHEMA = "msmarco-genqa.trec-dl-release-bundle.v1"
POINTER_SCHEMA = "msmarco-genqa.external-artifact-pointer.v1"
REPRODUCTION_SCHEMA = "msmarco-genqa.trec-dl-reproduction.v1"
SOURCE_RECORD_SCHEMA = "msmarco-genqa.table-artifact.v1"
MANIFEST_NAME = "bundle_manifest.json"
README_NAME = "README.md"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
DEFAULT_TOLERANCE = 1e-12
MAX_MANIFEST_BYTES = 1_000_000
MAX_MEMBER_BYTES = 50_000_000
MAX_TOTAL_MEMBER_BYTES = 100_000_000
WINDOWS_DEVICE_NAMES = {
    "AUX",
    "CON",
    "NUL",
    "PRN",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class ReleaseArtifactError(RuntimeError):
    """Raised when a release artifact fails validation or reproduction."""


@dataclass(frozen=True)
class RunSpec:
    year: int
    system: str
    source_key: str
    result_key: str
    archive_path: str


RUN_SPECS = (
    RunSpec(2019, "bm25", "bm25_run", "bm25", "runs/trec_dl_2019_bm25.tsv"),
    RunSpec(
        2019,
        "bm25_ce",
        "rerank_run",
        "bm25_ce",
        "runs/trec_dl_2019_bm25_ce.tsv",
    ),
    RunSpec(2020, "bm25", "bm25_run", "bm25", "runs/trec_dl_2020_bm25.tsv"),
    RunSpec(
        2020,
        "bm25_ce",
        "rerank_run",
        "bm25_ce",
        "runs/trec_dl_2020_bm25_ce.tsv",
    ),
)


def _load_json_object(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseArtifactError(f"cannot read JSON object {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseArtifactError(f"{source}: expected a JSON object")
    return value


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest for *data*."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Return the lowercase SHA-256 digest for a file without loading it all."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_archive_name(name: str) -> str:
    if not name or "\\" in name or "\x00" in name:
        raise ReleaseArtifactError(f"unsafe archive member name: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ReleaseArtifactError(f"unsafe archive member name: {name!r}")
    for part in path.parts:
        stem = part.split(".", 1)[0].upper()
        if ":" in part or part.rstrip(" .") != part or stem in WINDOWS_DEVICE_NAMES:
            raise ReleaseArtifactError(f"unsafe archive member name: {name!r}")
    return path.as_posix()


def _member_record(path: str, data: bytes, **metadata: Any) -> dict[str, Any]:
    return {
        "path": _safe_archive_name(path),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        **metadata,
    }


def _run_statistics(path: Path) -> tuple[int, int, int]:
    run = read_run_tsv(path)
    if not run:
        raise ReleaseArtifactError(f"{path}: run contains no records")
    row_count = sum(len(rows) for rows in run.values())
    max_depth = max(len(rows) for rows in run.values())
    return len(run), row_count, max_depth


def _expected_metrics(result: dict[str, Any]) -> dict[str, float]:
    names = ("mrr@10", "ndcg@10", "recall@100", "recall@1000")
    metrics = {
        name: float(result[name])
        for name in names
        if name in result
    }
    if not metrics:
        raise ReleaseArtifactError("source record contains no headline metrics")
    return metrics


def _bundle_readme(artifact_id: str, experiment_commit: str) -> bytes:
    text = f"""# TREC-DL baseline run bundle

Artifact: `{artifact_id}`

Experiment commit: `{experiment_commit}`

This archive contains the BM25 and BM25-plus-cross-encoder ranked run files
for the judged TREC-DL 2019 and 2020 passage tracks. It contains document
identifiers and scores, but no passage text, query text, qrels mirror, model
weights, or machine-local paths.

From a clone of `GioiaZheng/msmarco-genqa`, run:

```bash
make reproduce-trec-eval
```

The command verifies the archive and member SHA-256 digests, obtains the
public judgments through `ir_datasets`, recomputes the headline metrics, and
checks them against `bundle_manifest.json`. This validates the published
retrieval evidence without rebuilding the 8.8M-passage index or rerunning the
cross-encoder.
"""
    return text.encode("utf-8")


def _write_deterministic_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name in sorted(members):
                safe_name = _safe_archive_name(name)
                info = zipfile.ZipInfo(safe_name, date_time=ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                info.create_system = 3
                archive.writestr(info, members[name], compresslevel=9)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def build_release_bundle(
    *,
    source_record_path: Path | str,
    project_root: Path | str,
    output_archive: Path | str,
    artifact_id: str = "trec-dl-bm25-ce-2019-2020-v1",
) -> dict[str, Any]:
    """Build a deterministic, text-only release archive from canonical runs."""
    root = Path(project_root).resolve()
    source_record_path = Path(source_record_path).resolve()
    try:
        source_record_relative = source_record_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ReleaseArtifactError(
            f"source record must be inside the project root: {source_record_path}"
        ) from exc
    source_record = _load_json_object(source_record_path)
    if source_record.get("schema") != SOURCE_RECORD_SCHEMA:
        raise ReleaseArtifactError(
            f"{source_record_path}: expected schema {SOURCE_RECORD_SCHEMA!r}"
        )

    experiment = source_record.get("experiment")
    if not isinstance(experiment, dict) or not experiment.get("git_commit"):
        raise ReleaseArtifactError("source record is missing experiment.git_commit")
    experiment_commit = str(experiment["git_commit"])

    members: dict[str, bytes] = {}
    run_records: list[dict[str, Any]] = []
    results = source_record.get("results")
    if not isinstance(results, dict):
        raise ReleaseArtifactError("source record is missing results")

    for spec in RUN_SPECS:
        year_result = results.get(str(spec.year))
        if not isinstance(year_result, dict):
            raise ReleaseArtifactError(f"source record is missing results.{spec.year}")
        artifact_paths = year_result.get("artifact_paths")
        artifact_hashes = year_result.get("artifact_sha256")
        system_result = year_result.get(spec.result_key)
        if not all(isinstance(value, dict) for value in (artifact_paths, artifact_hashes, system_result)):
            raise ReleaseArtifactError(
                f"source record has incomplete {spec.year} {spec.system} metadata"
            )

        relative_source = artifact_paths.get(spec.source_key)
        expected_source_hash = artifact_hashes.get(spec.source_key)
        if not isinstance(relative_source, str) or not isinstance(expected_source_hash, str):
            raise ReleaseArtifactError(
                f"source record has no path/hash for {spec.year} {spec.system}"
            )
        source_path = (root / relative_source).resolve()
        try:
            source_path.relative_to(root)
        except ValueError as exc:
            raise ReleaseArtifactError(
                f"source path escapes project root: {relative_source}"
            ) from exc
        if not source_path.is_file():
            raise ReleaseArtifactError(f"missing canonical run: {relative_source}")
        data = source_path.read_bytes()
        actual_source_hash = sha256_bytes(data)
        if actual_source_hash != expected_source_hash.lower():
            raise ReleaseArtifactError(
                f"{relative_source}: SHA-256 mismatch; expected "
                f"{expected_source_hash}, got {actual_source_hash}"
            )
        query_count, row_count, max_depth = _run_statistics(source_path)
        judged_topics = int(year_result.get("judged_topics", 0))
        if query_count != judged_topics:
            raise ReleaseArtifactError(
                f"{relative_source}: found {query_count} queries; "
                f"source record declares {judged_topics}"
            )

        members[spec.archive_path] = data
        run_records.append(
            _member_record(
                spec.archive_path,
                data,
                kind="trec_run",
                year=spec.year,
                system=spec.system,
                query_count=query_count,
                row_count=row_count,
                max_depth=max_depth,
                expected_metrics=_expected_metrics(system_result),
            )
        )

    readme = _bundle_readme(artifact_id, experiment_commit)
    members[README_NAME] = readme
    readme_record = _member_record(README_NAME, readme, kind="documentation")
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "artifact_id": artifact_id,
        "experiment_commit": experiment_commit,
        "source_record": {
            "repository_path": source_record_relative,
            "sha256": sha256_file(source_record_path),
        },
        "scope": {
            "datasets": [
                "msmarco-passage/trec-dl-2019/judged",
                "msmarco-passage/trec-dl-2020/judged",
            ],
            "systems": ["bm25", "bm25_ce"],
            "binary_relevance_threshold": DEFAULT_REL_THRESHOLD,
            "metric_tolerance": DEFAULT_TOLERANCE,
        },
        "redistribution": {
            "included": "ranked document identifiers, ranks, scores, and checksums",
            "excluded": [
                "MS MARCO passage text",
                "query text",
                "qrels mirrors",
                "model weights and caches",
                "machine-local paths",
            ],
            "qrels_recovery": "ir_datasets judged TREC-DL subsets",
        },
        "members": [readme_record, *run_records],
    }
    members[MANIFEST_NAME] = _canonical_json_bytes(manifest)

    archive_path = Path(output_archive).resolve()
    _write_deterministic_zip(archive_path, members)
    return {
        "archive": str(archive_path),
        "bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
        "manifest": manifest,
    }


def _read_archive_manifest(
    archive: zipfile.ZipFile,
    *,
    expected_schema: str = BUNDLE_SCHEMA,
) -> dict[str, Any]:
    try:
        if archive.getinfo(MANIFEST_NAME).file_size > MAX_MANIFEST_BYTES:
            raise ReleaseArtifactError(f"{MANIFEST_NAME} exceeds the size limit")
        raw = archive.read(MANIFEST_NAME)
        manifest = json.loads(raw.decode("utf-8"))
    except KeyError as exc:
        raise ReleaseArtifactError(f"archive is missing {MANIFEST_NAME}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseArtifactError(f"invalid {MANIFEST_NAME}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != expected_schema:
        raise ReleaseArtifactError(f"archive does not use {expected_schema}")
    return manifest


def _manifest_members(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_members = manifest.get("members")
    if not isinstance(raw_members, list) or not raw_members:
        raise ReleaseArtifactError("bundle manifest has no members")
    members: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for record in raw_members:
        if not isinstance(record, dict):
            raise ReleaseArtifactError("bundle manifest member is not an object")
        name = record.get("path")
        if not isinstance(name, str):
            raise ReleaseArtifactError("bundle manifest member has no path")
        safe_name = _safe_archive_name(name)
        if safe_name in members:
            raise ReleaseArtifactError(f"duplicate bundle manifest member: {safe_name}")
        if not isinstance(record.get("sha256"), str) or not isinstance(record.get("bytes"), int):
            raise ReleaseArtifactError(f"incomplete member record: {safe_name}")
        member_bytes = int(record["bytes"])
        if member_bytes < 0 or member_bytes > MAX_MEMBER_BYTES:
            raise ReleaseArtifactError(f"invalid member size: {safe_name}")
        total_bytes += member_bytes
        if total_bytes > MAX_TOTAL_MEMBER_BYTES:
            raise ReleaseArtifactError("bundle members exceed the total size limit")
        members[safe_name] = record
    return members


def verify_release_archive(
    archive_path: Path | str,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    expected_schema: str = BUNDLE_SCHEMA,
) -> dict[str, Any]:
    """Verify outer archive identity, safe paths, and every member checksum."""
    source = Path(archive_path)
    if not source.is_file():
        raise ReleaseArtifactError(f"release archive not found: {source}")
    actual_sha256 = sha256_file(source)
    actual_bytes = source.stat().st_size
    if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
        raise ReleaseArtifactError(
            f"{source}: archive SHA-256 mismatch; expected {expected_sha256}, got {actual_sha256}"
        )
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise ReleaseArtifactError(
            f"{source}: archive size mismatch; expected {expected_bytes}, got {actual_bytes}"
        )

    try:
        with zipfile.ZipFile(source) as archive:
            infos = archive.infolist()
            names = [_safe_archive_name(info.filename) for info in infos]
            if len(names) != len(set(names)):
                raise ReleaseArtifactError("archive contains duplicate member names")
            manifest = _read_archive_manifest(archive, expected_schema=expected_schema)
            member_records = _manifest_members(manifest)
            expected_names = {MANIFEST_NAME, *member_records}
            if set(names) != expected_names:
                missing = sorted(expected_names - set(names))
                extra = sorted(set(names) - expected_names)
                raise ReleaseArtifactError(
                    f"archive member set mismatch; missing={missing}, extra={extra}"
                )
            for name, record in member_records.items():
                data = archive.read(name)
                if len(data) != record["bytes"]:
                    raise ReleaseArtifactError(f"{name}: member size mismatch")
                actual_member_hash = sha256_bytes(data)
                if actual_member_hash != str(record["sha256"]).lower():
                    raise ReleaseArtifactError(f"{name}: member SHA-256 mismatch")
    except zipfile.BadZipFile as exc:
        raise ReleaseArtifactError(f"invalid ZIP archive {source}: {exc}") from exc

    return {
        "archive": str(source.resolve()),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "manifest": manifest,
    }


def load_release_pointer(pointer_path: Path | str) -> dict[str, Any]:
    """Load and validate a Git-tracked external-artifact pointer."""
    pointer = _load_json_object(pointer_path)
    if pointer.get("schema") != POINTER_SCHEMA:
        raise ReleaseArtifactError(f"pointer does not use {POINTER_SCHEMA}")
    if not isinstance(pointer.get("artifact_id"), str) or not pointer["artifact_id"]:
        raise ReleaseArtifactError("pointer is missing artifact_id")
    release = pointer.get("release")
    download = pointer.get("download")
    if not isinstance(release, dict) or not isinstance(download, dict):
        raise ReleaseArtifactError("pointer is missing release/download metadata")
    required_release = ("repository", "tag", "asset")
    required_download = ("url", "sha256", "bytes")
    if any(not isinstance(release.get(key), str) or not release[key] for key in required_release):
        raise ReleaseArtifactError("pointer has incomplete release metadata")
    if any(download.get(key) is None or download.get(key) == "" for key in required_download):
        raise ReleaseArtifactError("pointer has incomplete download metadata")
    if not isinstance(download["url"], str) or not download["url"].startswith(
        ("https://", "file://")
    ):
        raise ReleaseArtifactError(
            "pointer download.url must use HTTPS (file:// is for local validation)"
        )
    if (
        isinstance(download["bytes"], bool)
        or not isinstance(download["bytes"], int)
        or download["bytes"] < 1
    ):
        raise ReleaseArtifactError("pointer download.bytes must be a positive integer")
    if not isinstance(download["sha256"], str):
        raise ReleaseArtifactError("pointer download.sha256 is not a SHA-256 digest")
    digest = download["sha256"].lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ReleaseArtifactError("pointer download.sha256 is not a SHA-256 digest")
    return pointer


def _download(
    url: str,
    destination: Path,
    *,
    expected_bytes: int,
    timeout: float,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "msmarco-genqa/trec-release"})
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as out:
            copied = 0
            while True:
                block = response.read(min(1024 * 1024, expected_bytes - copied + 1))
                if not block:
                    break
                copied += len(block)
                if copied > expected_bytes:
                    raise ReleaseArtifactError(
                        f"download exceeds the pinned size of {expected_bytes} bytes"
                    )
                out.write(block)
            if copied != expected_bytes:
                raise ReleaseArtifactError(
                    f"download size mismatch; expected {expected_bytes}, got {copied}"
                )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _verify_extracted_directory(bundle_dir: Path, manifest: dict[str, Any]) -> None:
    for name, record in _manifest_members(manifest).items():
        target = _safe_extraction_target(bundle_dir, name)
        if not target.is_file():
            raise ReleaseArtifactError(f"extracted bundle is missing {name}")
        if target.stat().st_size != record["bytes"] or sha256_file(target) != str(
            record["sha256"]
        ).lower():
            raise ReleaseArtifactError(f"extracted bundle member failed validation: {name}")


def _safe_extraction_target(bundle_dir: Path, name: str) -> Path:
    safe_name = _safe_archive_name(name)
    root = bundle_dir.resolve()
    target = bundle_dir.joinpath(*PurePosixPath(safe_name).parts)
    for parent in target.parents:
        if parent == bundle_dir.parent:
            break
        if parent.exists() and parent.is_symlink():
            raise ReleaseArtifactError(f"refusing to extract through a symlink: {parent}")
    if target.exists() and target.is_symlink():
        raise ReleaseArtifactError(f"refusing to overwrite a symlink: {target}")
    try:
        target.parent.resolve().relative_to(root)
    except ValueError as exc:
        raise ReleaseArtifactError(f"archive member escapes extraction root: {name}") from exc
    return target


def _extract_verified_archive(
    archive_path: Path,
    bundle_dir: Path,
    *,
    expected_schema: str = BUNDLE_SCHEMA,
) -> dict[str, Any]:
    verified = verify_release_archive(archive_path, expected_schema=expected_schema)
    manifest = verified["manifest"]
    bundle_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for name in sorted({MANIFEST_NAME, *_manifest_members(manifest)}):
            target = _safe_extraction_target(bundle_dir, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(name)
            if target.exists():
                if not target.is_file() or target.read_bytes() != data:
                    raise ReleaseArtifactError(
                        f"refusing to overwrite different existing bundle member: {target}"
                    )
                continue
            with tempfile.NamedTemporaryFile(
                prefix=f".{target.name}.", suffix=".part", dir=target.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(data)
            temporary.replace(target)
    _verify_extracted_directory(bundle_dir, manifest)
    return verified


def fetch_release_bundle(
    *,
    pointer_path: Path | str,
    output_dir: Path | str,
    timeout: float = 60.0,
    expected_schema: str = BUNDLE_SCHEMA,
) -> dict[str, Any]:
    """Download, verify, and safely extract the release selected by a pointer."""
    pointer = load_release_pointer(pointer_path)
    release = pointer["release"]
    download = pointer["download"]
    destination = Path(output_dir).resolve()
    asset_name = _safe_archive_name(str(release["asset"]))
    if "/" in asset_name:
        raise ReleaseArtifactError("release asset must be a file name, not a path")
    archive_path = destination / asset_name
    expected_hash = str(download["sha256"]).lower()
    expected_bytes = int(download["bytes"])
    if archive_path.exists():
        verify_release_archive(
            archive_path,
            expected_sha256=expected_hash,
            expected_bytes=expected_bytes,
            expected_schema=expected_schema,
        )
    else:
        try:
            _download(
                str(download["url"]),
                archive_path,
                expected_bytes=expected_bytes,
                timeout=timeout,
            )
        except (OSError, urllib.error.URLError) as exc:
            raise ReleaseArtifactError(
                f"cannot download {download['url']}: {exc}"
            ) from exc
        verify_release_archive(
            archive_path,
            expected_sha256=expected_hash,
            expected_bytes=expected_bytes,
            expected_schema=expected_schema,
        )
    bundle_dir = destination / "bundle"
    verified = _extract_verified_archive(
        archive_path,
        bundle_dir,
        expected_schema=expected_schema,
    )
    if verified["manifest"].get("artifact_id") != pointer["artifact_id"]:
        raise ReleaseArtifactError(
            "release bundle artifact_id does not match the Git-tracked pointer"
        )
    return {
        **verified,
        "bundle_dir": str(bundle_dir),
        "release": release,
    }


def _load_qrels_for_year(
    year: int,
    *,
    qrels_paths: dict[int, Path] | None,
    cache_dir: Path | None,
) -> tuple[dict[str, dict[str, int]], str]:
    if qrels_paths and year in qrels_paths:
        path = qrels_paths[year]
        return read_qrels(path, qrels_format="trec"), str(path)
    dataset = load_trec_dl(year, cache_dir=cache_dir, rel_threshold=DEFAULT_REL_THRESHOLD)
    return dataset.graded_qrels, dataset.dataset_name


def _write_reproduction_table(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# Reproduced TREC-DL metrics",
        "",
        "| Track | System | MRR@10 | nDCG@10 | Recall@100 | Recall@1000 | Max abs. delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(results):
        record = results[key]
        metrics = record["metrics"]
        lines.append(
            "| {year} | {system} | {mrr:.4f} | {ndcg:.4f} | {r100:.4f} | {r1000} | {delta:.2e} |".format(
                year=record["year"],
                system=record["system"],
                mrr=metrics["mrr@10"],
                ndcg=metrics["ndcg@10"],
                r100=metrics["recall@100"],
                r1000=f"{metrics['recall@1000']:.4f}" if "recall@1000" in metrics else "n/a",
                delta=record["max_abs_delta"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def evaluate_release_bundle(
    *,
    bundle_dir: Path | str,
    output_dir: Path | str,
    qrels_paths: dict[int, Path] | None = None,
    cache_dir: Path | str | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Recompute every published metric and fail on any out-of-tolerance delta."""
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ReleaseArtifactError("tolerance must be finite and non-negative")
    source = Path(bundle_dir).resolve()
    manifest = _load_json_object(source / MANIFEST_NAME)
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ReleaseArtifactError(f"bundle directory does not use {BUNDLE_SCHEMA}")
    _verify_extracted_directory(source, manifest)

    qrels_by_year: dict[int, dict[str, dict[str, int]]] = {}
    qrels_sources: dict[int, str] = {}
    cache_path = Path(cache_dir).resolve() if cache_dir is not None else None
    run_members = [
        record
        for record in _manifest_members(manifest).values()
        if record.get("kind") == "trec_run"
    ]
    results: dict[str, Any] = {}
    global_max_delta = 0.0
    for record in run_members:
        year = int(record["year"])
        if year not in qrels_by_year:
            qrels_by_year[year], qrels_sources[year] = _load_qrels_for_year(
                year,
                qrels_paths=qrels_paths,
                cache_dir=cache_path,
            )
        run = read_run_tsv(source / str(record["path"]))
        metrics = evaluate_internal_trec_scope(
            run,
            qrels_by_year[year],
            rel_threshold=DEFAULT_REL_THRESHOLD,
        )
        expected_metrics = record.get("expected_metrics")
        if not isinstance(expected_metrics, dict) or not expected_metrics:
            raise ReleaseArtifactError(f"{record['path']}: expected metrics are missing")
        deltas = {
            name: abs(float(metrics[name]) - float(expected))
            for name, expected in expected_metrics.items()
        }
        max_delta = max(deltas.values(), default=0.0)
        if max_delta > tolerance:
            raise ReleaseArtifactError(
                f"{record['path']}: reproduced metrics differ from the release record; "
                f"max delta {max_delta:.3g} exceeds {tolerance:g}"
            )
        global_max_delta = max(global_max_delta, max_delta)
        key = f"{year}_{record['system']}"
        reproduced_metrics = {
            name: float(metrics[name])
            for name in expected_metrics
        }
        reproduced_metrics["n_queries"] = int(metrics["n_queries"])
        results[key] = {
            "year": year,
            "system": record["system"],
            "run": record["path"],
            "qrels_source": qrels_sources[year],
            "metrics": reproduced_metrics,
            "expected_metrics": expected_metrics,
            "absolute_deltas": deltas,
            "max_abs_delta": max_delta,
        }

    report = {
        "schema": REPRODUCTION_SCHEMA,
        "artifact_id": manifest.get("artifact_id"),
        "experiment_commit": manifest.get("experiment_commit"),
        "verified": True,
        "tolerance": tolerance,
        "max_abs_delta": global_max_delta,
        "results": results,
    }
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_bytes(_canonical_json_bytes(report))
    _write_reproduction_table(destination / "metrics.md", results)
    return report
