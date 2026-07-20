"""Build, recover, and re-evaluate the public BEIR benchmark run bundle."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from msmarco_genqa.data.benchmark import load_benchmark_queries
from msmarco_genqa.evaluation.trec import evaluate_internal_trec_scope, read_qrels
from msmarco_genqa.reproducibility.trec_release import (
    DEFAULT_TOLERANCE,
    MANIFEST_NAME,
    README_NAME,
    SOURCE_RECORD_SCHEMA,
    ReleaseArtifactError,
    _canonical_json_bytes,
    _expected_metrics,
    _load_json_object,
    _manifest_members,
    _member_record,
    _run_statistics,
    _verify_extracted_directory,
    _write_deterministic_zip,
    fetch_release_bundle as _fetch_release_bundle,
    sha256_bytes,
    sha256_file,
    verify_release_archive as _verify_release_archive,
)
from msmarco_genqa.reranking.io import read_run_tsv


BUNDLE_SCHEMA = "msmarco-genqa.beir-release-bundle.v1"
REPRODUCTION_SCHEMA = "msmarco-genqa.beir-reproduction.v1"
DEFAULT_ARTIFACT_ID = "beir-nfcorpus-scifact-bm25-ce-v1"
DEFAULT_REL_THRESHOLD = 1

RUN_SPECS = (
    {
        "dataset_key": "nfcorpus",
        "source_key": "bm25_run",
        "result_key": "bm25",
        "system": "bm25",
        "archive_path": "runs/beir_nfcorpus_bm25.tsv",
    },
    {
        "dataset_key": "nfcorpus",
        "source_key": "rerank_run",
        "result_key": "bm25_ce",
        "system": "bm25_ce",
        "archive_path": "runs/beir_nfcorpus_bm25_ce.tsv",
    },
    {
        "dataset_key": "scifact",
        "source_key": "bm25_run",
        "result_key": "bm25",
        "system": "bm25",
        "archive_path": "runs/beir_scifact_bm25.tsv",
    },
    {
        "dataset_key": "scifact",
        "source_key": "rerank_run",
        "result_key": "bm25_ce",
        "system": "bm25_ce",
        "archive_path": "runs/beir_scifact_bm25_ce.tsv",
    },
)


def _bundle_readme(
    artifact_id: str,
    experiment_commits: dict[str, str],
) -> bytes:
    commits = ", ".join(
        f"`{key}`: `{value}`" for key, value in sorted(experiment_commits.items())
    )
    text = f"""# BEIR cross-domain baseline run bundle

Artifact: `{artifact_id}`

Experiment commits: {commits}

This archive contains the BM25 and BM25-plus-cross-encoder ranked run files
for the NFCorpus and SciFact BEIR test sets. It contains document identifiers,
ranks, and scores, but no document text, query text, qrels mirror, model
weights, caches, or machine-local manifests.

From a clone of `GioiaZheng/msmarco-genqa`, run:

```bash
make reproduce-beir-eval
```

The command verifies the archive and every member SHA-256 digest, obtains the
public judgments through `ir_datasets`, recomputes the headline metrics, and
checks them against `bundle_manifest.json`. This validates the published
retrieval evidence without rebuilding either corpus index or rerunning the
cross-encoder.
"""
    return text.encode("utf-8")


def _experiment_commits(source_record: dict[str, Any]) -> dict[str, str]:
    experiment = source_record.get("experiment")
    commits = experiment.get("git_commits") if isinstance(experiment, dict) else None
    if not isinstance(commits, dict) or not commits:
        raise ReleaseArtifactError("source record is missing experiment.git_commits")
    normalized = {
        str(key): str(value)
        for key, value in commits.items()
        if isinstance(key, str) and isinstance(value, str) and value
    }
    if len(normalized) != len(commits):
        raise ReleaseArtifactError("source record contains an invalid experiment commit")
    return normalized


def build_release_bundle(
    *,
    source_record_path: Path | str,
    project_root: Path | str,
    output_archive: Path | str,
    artifact_id: str = DEFAULT_ARTIFACT_ID,
) -> dict[str, Any]:
    """Build a deterministic, text-only release archive from canonical runs."""
    root = Path(project_root).resolve()
    source_path = Path(source_record_path).resolve()
    try:
        source_relative = source_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ReleaseArtifactError(
            f"source record must be inside the project root: {source_path}"
        ) from exc

    source_record = _load_json_object(source_path)
    if source_record.get("schema") != SOURCE_RECORD_SCHEMA:
        raise ReleaseArtifactError(
            f"{source_path}: expected schema {SOURCE_RECORD_SCHEMA!r}"
        )
    commits = _experiment_commits(source_record)
    results = source_record.get("results")
    if not isinstance(results, dict):
        raise ReleaseArtifactError("source record is missing results")

    members: dict[str, bytes] = {}
    run_records: list[dict[str, Any]] = []
    dataset_ids: list[str] = []
    for spec in RUN_SPECS:
        dataset_key = str(spec["dataset_key"])
        dataset_result = results.get(dataset_key)
        if not isinstance(dataset_result, dict):
            raise ReleaseArtifactError(f"source record is missing results.{dataset_key}")
        dataset_id = dataset_result.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id.startswith("beir/"):
            raise ReleaseArtifactError(f"results.{dataset_key} has no BEIR dataset_id")
        if dataset_id not in dataset_ids:
            dataset_ids.append(dataset_id)

        artifact_paths = dataset_result.get("artifact_paths")
        artifact_hashes = dataset_result.get("artifact_sha256")
        system_result = dataset_result.get(str(spec["result_key"]))
        if not all(
            isinstance(value, dict)
            for value in (artifact_paths, artifact_hashes, system_result)
        ):
            raise ReleaseArtifactError(
                f"source record has incomplete {dataset_key} {spec['system']} metadata"
            )

        source_key = str(spec["source_key"])
        relative_run = artifact_paths.get(source_key)
        expected_hash = artifact_hashes.get(source_key)
        if not isinstance(relative_run, str) or not isinstance(expected_hash, str):
            raise ReleaseArtifactError(
                f"source record has no path/hash for {dataset_key} {spec['system']}"
            )
        run_path = (root / relative_run).resolve()
        try:
            run_path.relative_to(root)
        except ValueError as exc:
            raise ReleaseArtifactError(f"source path escapes project root: {relative_run}") from exc
        if not run_path.is_file():
            raise ReleaseArtifactError(f"missing canonical run: {relative_run}")
        data = run_path.read_bytes()
        actual_hash = sha256_bytes(data)
        if actual_hash != expected_hash.lower():
            raise ReleaseArtifactError(
                f"{relative_run}: SHA-256 mismatch; expected {expected_hash}, got {actual_hash}"
            )

        query_count, row_count, max_depth = _run_statistics(run_path)
        judged_topics = int(dataset_result.get("judged_topics", 0))
        if query_count != judged_topics:
            raise ReleaseArtifactError(
                f"{relative_run}: found {query_count} queries; "
                f"source record declares {judged_topics}"
            )
        archive_path = str(spec["archive_path"])
        members[archive_path] = data
        run_records.append(
            _member_record(
                archive_path,
                data,
                kind="beir_run",
                dataset_key=dataset_key,
                dataset_id=dataset_id,
                system=spec["system"],
                query_count=query_count,
                row_count=row_count,
                max_depth=max_depth,
                expected_metrics=_expected_metrics(system_result),
            )
        )

    readme = _bundle_readme(artifact_id, commits)
    members[README_NAME] = readme
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "artifact_id": artifact_id,
        "experiment_commits": commits,
        "source_record": {
            "repository_path": source_relative,
            "sha256": sha256_file(source_path),
        },
        "scope": {
            "datasets": dataset_ids,
            "systems": ["bm25", "bm25_ce"],
            "binary_relevance_threshold": DEFAULT_REL_THRESHOLD,
            "metric_tolerance": DEFAULT_TOLERANCE,
        },
        "redistribution": {
            "included": "ranked document identifiers, ranks, scores, and checksums",
            "excluded": [
                "document text",
                "query text",
                "qrels mirrors",
                "model weights and caches",
                "machine-local paths and manifests",
            ],
            "qrels_recovery": "ir_datasets BEIR test datasets",
        },
        "members": [
            _member_record(README_NAME, readme, kind="documentation"),
            *run_records,
        ],
    }
    members[MANIFEST_NAME] = _canonical_json_bytes(manifest)

    archive = Path(output_archive).resolve()
    _write_deterministic_zip(archive, members)
    return {
        "archive": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
        "manifest": manifest,
    }


def verify_release_archive(
    archive_path: Path | str,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    """Verify a BEIR release archive and every declared member."""
    return _verify_release_archive(
        archive_path,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
        expected_schema=BUNDLE_SCHEMA,
    )


def fetch_release_bundle(
    *,
    pointer_path: Path | str,
    output_dir: Path | str,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Download, verify, and safely extract the BEIR release selected by a pointer."""
    return _fetch_release_bundle(
        pointer_path=pointer_path,
        output_dir=output_dir,
        timeout=timeout,
        expected_schema=BUNDLE_SCHEMA,
    )


def _load_qrels(
    dataset_key: str,
    dataset_id: str,
    *,
    qrels_paths: dict[str, Path] | None,
    cache_dir: Path | None,
) -> tuple[dict[str, dict[str, int]], str]:
    if qrels_paths and dataset_key in qrels_paths:
        path = qrels_paths[dataset_key]
        return read_qrels(path, qrels_format="trec"), str(path)
    dataset = load_benchmark_queries(dataset_id, cache_dir=cache_dir)
    return dataset.graded_qrels, dataset.spec.dataset_id


def _write_reproduction_table(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# Reproduced BEIR cross-domain metrics",
        "",
        "| Dataset | System | MRR@10 | nDCG@10 | Recall@100 | Recall@1000 | Max abs. delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(results):
        record = results[key]
        metrics = record["metrics"]
        lines.append(
            "| {dataset} | {system} | {mrr:.4f} | {ndcg:.4f} | {r100:.4f} | "
            "{r1000} | {delta:.2e} |".format(
                dataset=record["dataset_key"],
                system=record["system"],
                mrr=metrics["mrr@10"],
                ndcg=metrics["ndcg@10"],
                r100=metrics["recall@100"],
                r1000=(
                    f"{metrics['recall@1000']:.4f}"
                    if "recall@1000" in metrics
                    else "n/a"
                ),
                delta=record["max_abs_delta"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def evaluate_release_bundle(
    *,
    bundle_dir: Path | str,
    output_dir: Path | str,
    qrels_paths: dict[str, Path] | None = None,
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

    cache_path = Path(cache_dir).resolve() if cache_dir is not None else None
    qrels_by_dataset: dict[str, dict[str, dict[str, int]]] = {}
    qrels_sources: dict[str, str] = {}
    results: dict[str, Any] = {}
    global_max_delta = 0.0
    run_members = [
        record
        for record in _manifest_members(manifest).values()
        if record.get("kind") == "beir_run"
    ]
    for record in run_members:
        dataset_key = str(record["dataset_key"])
        dataset_id = str(record["dataset_id"])
        if dataset_key not in qrels_by_dataset:
            qrels_by_dataset[dataset_key], qrels_sources[dataset_key] = _load_qrels(
                dataset_key,
                dataset_id,
                qrels_paths=qrels_paths,
                cache_dir=cache_path,
            )
        run = read_run_tsv(source / str(record["path"]))
        metrics = evaluate_internal_trec_scope(
            run,
            qrels_by_dataset[dataset_key],
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
        key = f"{dataset_key}_{record['system']}"
        reproduced_metrics = {name: float(metrics[name]) for name in expected_metrics}
        reproduced_metrics["n_queries"] = int(metrics["n_queries"])
        results[key] = {
            "dataset_key": dataset_key,
            "dataset_id": dataset_id,
            "system": record["system"],
            "run": record["path"],
            "qrels_source": qrels_sources[dataset_key],
            "metrics": reproduced_metrics,
            "expected_metrics": expected_metrics,
            "absolute_deltas": deltas,
            "max_abs_delta": max_delta,
        }

    report = {
        "schema": REPRODUCTION_SCHEMA,
        "artifact_id": manifest.get("artifact_id"),
        "experiment_commits": manifest.get("experiment_commits"),
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
