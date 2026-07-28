"""Build, recover, and re-evaluate the NFCorpus video-query ablation bundle."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from msmarco_genqa.data.benchmark import load_benchmark_queries
from msmarco_genqa.evaluation.bootstrap import paired_bootstrap_diff
from msmarco_genqa.evaluation.retrieval import (
    first_relevant_rank,
    recall_at_k,
    reciprocal_rank,
)
from msmarco_genqa.evaluation.trec import (
    evaluate_internal_trec_scope,
    graded_ndcg_at_k,
    read_qrels,
)
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
    _verify_extracted_directory,
    _write_deterministic_zip,
    fetch_release_bundle as _fetch_release_bundle,
    sha256_bytes,
    sha256_file,
    verify_release_archive as _verify_release_archive,
)
from msmarco_genqa.reranking.io import read_run_tsv


BUNDLE_SCHEMA = "msmarco-genqa.nfcorpus-video-release-bundle.v1"
REPRODUCTION_SCHEMA = "msmarco-genqa.nfcorpus-video-reproduction.v1"
DEFAULT_ARTIFACT_ID = "nfcorpus-video-query-representation-bm25-ce-v1"
DEFAULT_DATASET_ID = "beir/nfcorpus/test"
DEFAULT_REL_THRESHOLD = 1
REPRESENTATIONS = ("title", "description", "title_plus_description")
SYSTEM_METRICS = {
    "bm25": ("mrr@10", "ndcg@10", "recall@100", "recall@1000"),
    "bm25_ce": ("mrr@10", "ndcg@10", "recall@100"),
}


@dataclass(frozen=True)
class RunSpec:
    representation: str
    system: str
    source_key: str
    result_key: str
    archive_path: str


RUN_SPECS = tuple(
    RunSpec(
        representation=representation,
        system=system,
        source_key=source_key,
        result_key=result_key,
        archive_path=(
            f"runs/nfcorpus_video_{representation}_"
            f"{'bm25_ce' if system == 'bm25_ce' else 'bm25'}.tsv"
        ),
    )
    for representation in REPRESENTATIONS
    for system, source_key, result_key in (
        ("bm25", "bm25_run", "bm25"),
        ("bm25_ce", "rerank_run", "bm25_ce"),
    )
)


def _qid_sha256(qids: set[str]) -> str:
    payload = "".join(f"{qid}\n" for qid in sorted(qids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _experiment_commits(source_record: Mapping[str, Any]) -> dict[str, str]:
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


def _bootstrap_record(value: Any, *, label: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ReleaseArtifactError(f"source record is missing {label}")
    names = ("mean_delta", "ci_low", "ci_high", "p_two_sided")
    try:
        record = {name: float(value[name]) for name in names}
    except (KeyError, TypeError, ValueError) as exc:
        raise ReleaseArtifactError(f"source record has invalid {label}") from exc
    if not all(math.isfinite(number) for number in record.values()):
        raise ReleaseArtifactError(f"source record has non-finite {label}")
    return record


def _expected_paired_comparisons(results: Mapping[str, Any]) -> dict[str, Any]:
    description = results.get("description")
    combined = results.get("title_plus_description")
    if not isinstance(description, dict) or not isinstance(combined, dict):
        raise ReleaseArtifactError("source record is missing treatment results")
    description_paired = description.get("paired_vs_title")
    combined_paired = combined.get("paired_vs_title")
    if not isinstance(description_paired, dict) or not isinstance(combined_paired, dict):
        raise ReleaseArtifactError("source record is missing paired comparisons")

    return {
        "bm25": {
            "description_vs_title": {
                "recall@100": _bootstrap_record(
                    description_paired.get("recall@100"),
                    label="results.description.paired_vs_title.recall@100",
                )
            },
            "title_plus_description_vs_title": {
                "mrr@10": _bootstrap_record(
                    combined_paired.get("bm25_mrr@10"),
                    label=(
                        "results.title_plus_description.paired_vs_title."
                        "bm25_mrr@10"
                    ),
                ),
                "ndcg@10": _bootstrap_record(
                    combined_paired.get("bm25_ndcg@10"),
                    label=(
                        "results.title_plus_description.paired_vs_title."
                        "bm25_ndcg@10"
                    ),
                ),
                "recall@100": _bootstrap_record(
                    combined_paired.get("recall@100"),
                    label=(
                        "results.title_plus_description.paired_vs_title."
                        "recall@100"
                    ),
                ),
            },
        },
        "bm25_ce": {
            "title_plus_description_vs_title": {
                "mrr@10": _bootstrap_record(
                    combined_paired.get("bm25_ce_mrr@10"),
                    label=(
                        "results.title_plus_description.paired_vs_title."
                        "bm25_ce_mrr@10"
                    ),
                ),
                "ndcg@10": _bootstrap_record(
                    combined_paired.get("bm25_ce_ndcg@10"),
                    label=(
                        "results.title_plus_description.paired_vs_title."
                        "bm25_ce_ndcg@10"
                    ),
                ),
            }
        },
    }


def _bundle_readme(artifact_id: str, experiment_commits: Mapping[str, str]) -> bytes:
    commits = ", ".join(
        f"`{key}`: `{value}`" for key, value in sorted(experiment_commits.items())
    )
    text = f"""# NFCorpus video query-representation run bundle

Artifact: `{artifact_id}`

Experiment commits: {commits}

This archive contains six exact ranked outputs for the official 102-query
NFCorpus test/video subset: BM25 top-1,000 and fixed top-100 cross-encoder
reranking for title, description, and title-plus-description queries.

The payload contains query/document identifiers, ranks, scores, checksums, and
compact metric expectations. It contains no query text, document text, qrels
mirror, model weights, caches, or machine-local manifests.

From a clone of `GioiaZheng/msmarco-genqa`, run:

```bash
make reproduce-nfcorpus-video-eval
```

The command downloads this archive, verifies its pinned byte size and SHA-256,
verifies every member digest, obtains the public NFCorpus judgments through
`ir_datasets`, and recomputes the six aggregate rows plus the paired-bootstrap
comparisons. It does not rebuild the index or rerun either retrieval stage.
"""
    return text.encode("utf-8")


def _source_run(
    *,
    root: Path,
    representation_result: Mapping[str, Any],
    spec: RunSpec,
) -> tuple[Path, bytes, dict[str, list[tuple[str, float]]]]:
    artifact_paths = representation_result.get("artifact_paths")
    artifact_hashes = representation_result.get("artifact_sha256")
    if not isinstance(artifact_paths, dict) or not isinstance(artifact_hashes, dict):
        raise ReleaseArtifactError(
            f"source record has incomplete {spec.representation} artifact metadata"
        )
    relative_run = artifact_paths.get(spec.source_key)
    expected_hash = artifact_hashes.get(spec.source_key)
    if not isinstance(relative_run, str) or not isinstance(expected_hash, str):
        raise ReleaseArtifactError(
            f"source record has no path/hash for {spec.representation} {spec.system}"
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
    return run_path, data, read_run_tsv(run_path)


def _validate_candidate_sets(
    runs: Mapping[tuple[str, str], dict[str, list[tuple[str, float]]]],
    *,
    rerank_depth: int,
) -> int:
    checks = 0
    for representation in REPRESENTATIONS:
        bm25 = runs[(representation, "bm25")]
        rerank = runs[(representation, "bm25_ce")]
        if set(bm25) != set(rerank):
            raise ReleaseArtifactError(
                f"{representation}: BM25 and reranked qid sets differ"
            )
        for qid in sorted(bm25):
            bm25_candidates = {
                doc_id for doc_id, _score in bm25[qid][:rerank_depth]
            }
            rerank_candidates = {doc_id for doc_id, _score in rerank[qid]}
            if bm25_candidates != rerank_candidates:
                raise ReleaseArtifactError(
                    f"{representation}/{qid}: reranked candidates differ from "
                    f"BM25 top-{rerank_depth}"
                )
            checks += 1
    return checks


def build_release_bundle(
    *,
    source_record_path: Path | str,
    project_root: Path | str,
    output_archive: Path | str,
    artifact_id: str = DEFAULT_ARTIFACT_ID,
) -> dict[str, Any]:
    """Build a deterministic text-only bundle from the six canonical runs."""
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
    canonical_source_record = _canonical_json_bytes(source_record)

    query_set = source_record.get("query_set")
    experiment = source_record.get("experiment")
    model = source_record.get("model_revision")
    results = source_record.get("results")
    if not all(isinstance(value, dict) for value in (query_set, experiment, model, results)):
        raise ReleaseArtifactError("source record is missing release metadata")
    expected_queries = int(query_set.get("n_queries", 0))
    expected_qid_hash = str(query_set.get("qid_sha256", ""))
    dataset_id = str(experiment.get("dataset", ""))
    if expected_queries < 1 or len(expected_qid_hash) != 64:
        raise ReleaseArtifactError("source record has an invalid query-set contract")
    if dataset_id != DEFAULT_DATASET_ID:
        raise ReleaseArtifactError(f"source record dataset must be {DEFAULT_DATASET_ID}")

    bm25_config = experiment.get("bm25")
    reranker_config = experiment.get("reranker")
    evaluation = experiment.get("evaluation")
    if not all(
        isinstance(value, dict)
        for value in (bm25_config, reranker_config, evaluation)
    ):
        raise ReleaseArtifactError("source record has incomplete experiment metadata")
    bm25_depth = int(bm25_config.get("retrieval_depth", 0))
    rerank_depth = int(reranker_config.get("depth", 0))
    bootstrap_resamples = int(evaluation.get("bootstrap_resamples", 0))
    bootstrap_seed = int(evaluation.get("bootstrap_seed", 0))
    if min(bm25_depth, rerank_depth, bootstrap_resamples) < 1:
        raise ReleaseArtifactError("source record has invalid depth/bootstrap settings")

    members: dict[str, bytes] = {}
    run_records: list[dict[str, Any]] = []
    runs: dict[tuple[str, str], dict[str, list[tuple[str, float]]]] = {}
    reference_qids: set[str] | None = None
    for spec in RUN_SPECS:
        representation_result = results.get(spec.representation)
        if not isinstance(representation_result, dict):
            raise ReleaseArtifactError(
                f"source record is missing results.{spec.representation}"
            )
        system_result = representation_result.get(spec.result_key)
        if not isinstance(system_result, dict):
            raise ReleaseArtifactError(
                f"source record is missing {spec.representation} {spec.system} metrics"
            )
        run_path, data, run = _source_run(
            root=root,
            representation_result=representation_result,
            spec=spec,
        )
        expected_depth = bm25_depth if spec.system == "bm25" else rerank_depth
        qids = set(run)
        if len(qids) != expected_queries:
            raise ReleaseArtifactError(
                f"{run_path}: found {len(qids)} queries; expected {expected_queries}"
            )
        if any(len(rows) != expected_depth for rows in run.values()):
            raise ReleaseArtifactError(
                f"{run_path}: not every query has depth {expected_depth}"
            )
        if reference_qids is None:
            reference_qids = qids
        elif qids != reference_qids:
            raise ReleaseArtifactError("query-id sets differ across the six runs")

        row_count = sum(len(rows) for rows in run.values())
        members[spec.archive_path] = data
        runs[(spec.representation, spec.system)] = run
        run_records.append(
            _member_record(
                spec.archive_path,
                data,
                kind="nfcorpus_video_run",
                dataset_id=dataset_id,
                representation=spec.representation,
                system=spec.system,
                query_count=len(qids),
                row_count=row_count,
                max_depth=expected_depth,
                expected_metrics=_expected_metrics(system_result),
            )
        )

    if reference_qids is None or _qid_sha256(reference_qids) != expected_qid_hash:
        raise ReleaseArtifactError("run qids do not match the frozen cohort hash")
    candidate_checks = _validate_candidate_sets(runs, rerank_depth=rerank_depth)
    commits = _experiment_commits(source_record)
    expected_paired = _expected_paired_comparisons(results)

    readme = _bundle_readme(artifact_id, commits)
    members[README_NAME] = readme
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "artifact_id": artifact_id,
        "experiment_commits": commits,
        "source_record": {
            "repository_path": source_relative,
            "sha256": sha256_bytes(canonical_source_record),
            "serialization": "canonical-json-utf8-sorted-keys-compact",
        },
        "scope": {
            "dataset": dataset_id,
            "subset": str(experiment.get("subset")),
            "representations": list(REPRESENTATIONS),
            "systems": ["bm25", "bm25_ce"],
            "query_count": expected_queries,
            "qid_sha256": expected_qid_hash,
            "bm25_depth": bm25_depth,
            "rerank_depth": rerank_depth,
            "candidate_set_checks": candidate_checks,
            "binary_relevance_threshold": DEFAULT_REL_THRESHOLD,
            "metric_tolerance": DEFAULT_TOLERANCE,
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_seed": bootstrap_seed,
            "confidence_level": float(evaluation.get("confidence_level", 0.95)),
        },
        "model_revision": model,
        "expected_paired_comparisons": expected_paired,
        "redistribution": {
            "included": "ranked query/document identifiers, ranks, scores, and checksums",
            "excluded": [
                "query text",
                "document text",
                "qrels mirrors",
                "model weights and caches",
                "machine-local paths and manifests",
            ],
            "qrels_recovery": "ir_datasets beir/nfcorpus/test",
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
    """Verify the outer archive and every declared text member."""
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
    """Download, verify, and safely extract the pinned ablation bundle."""
    return _fetch_release_bundle(
        pointer_path=pointer_path,
        output_dir=output_dir,
        timeout=timeout,
        expected_schema=BUNDLE_SCHEMA,
    )


def _load_qrels(
    dataset_id: str,
    *,
    qrels_path: Path | None,
    cache_dir: Path | None,
) -> tuple[dict[str, dict[str, int]], str]:
    if qrels_path is not None:
        return read_qrels(qrels_path, qrels_format="auto"), str(qrels_path)
    benchmark = load_benchmark_queries(dataset_id, cache_dir=cache_dir)
    return benchmark.graded_qrels, benchmark.spec.dataset_id


def _per_query_scores(
    run: Mapping[str, list[tuple[str, float]]],
    qrels: Mapping[str, dict[str, int]],
    metric: str,
) -> list[float]:
    scores: list[float] = []
    for qid in sorted(qrels):
        judgments = qrels[qid]
        relevant = {
            doc_id
            for doc_id, relevance in judgments.items()
            if relevance >= DEFAULT_REL_THRESHOLD
        }
        ranked = [doc_id for doc_id, _score in run[qid]]
        if metric == "mrr@10":
            value = reciprocal_rank(ranked, relevant, 10)
        elif metric == "ndcg@10":
            value = graded_ndcg_at_k(ranked, judgments, k=10)
        elif metric == "recall@100":
            value = recall_at_k(ranked, relevant, 100)
        elif metric == "recall@1000":
            value = recall_at_k(ranked, relevant, 1000)
        else:
            raise ReleaseArtifactError(f"unsupported paired metric: {metric}")
        scores.append(float(value))
    return scores


def _no_hit_counts(
    baseline: Mapping[str, list[tuple[str, float]]],
    treatment: Mapping[str, list[tuple[str, float]]],
    qrels: Mapping[str, dict[str, int]],
    *,
    cutoff: int,
) -> dict[str, int]:
    baseline_miss: list[bool] = []
    treatment_miss: list[bool] = []
    for qid in sorted(qrels):
        relevant = {
            doc_id
            for doc_id, relevance in qrels[qid].items()
            if relevance >= DEFAULT_REL_THRESHOLD
        }
        baseline_ranked = [doc_id for doc_id, _score in baseline[qid]]
        treatment_ranked = [doc_id for doc_id, _score in treatment[qid]]
        baseline_miss.append(
            first_relevant_rank(baseline_ranked, relevant, cutoff) is None
        )
        treatment_miss.append(
            first_relevant_rank(treatment_ranked, relevant, cutoff) is None
        )
    return {
        "baseline": sum(baseline_miss),
        "treatment": sum(treatment_miss),
        "recovered": sum(
            before and not after
            for before, after in zip(baseline_miss, treatment_miss)
        ),
        "lost": sum(
            not before and after
            for before, after in zip(baseline_miss, treatment_miss)
        ),
    }


def _paired_comparisons(
    runs: Mapping[tuple[str, str], dict[str, list[tuple[str, float]]]],
    qrels: Mapping[str, dict[str, int]],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    pairs = (
        ("description_vs_title", "title", "description"),
        (
            "title_plus_description_vs_title",
            "title",
            "title_plus_description",
        ),
        (
            "title_plus_description_vs_description",
            "description",
            "title_plus_description",
        ),
    )
    comparisons: dict[str, Any] = {}
    for system, metric_names in SYSTEM_METRICS.items():
        system_comparisons: dict[str, Any] = {}
        for label, baseline_name, treatment_name in pairs:
            baseline_run = runs[(baseline_name, system)]
            treatment_run = runs[(treatment_name, system)]
            metrics: dict[str, Any] = {}
            for metric in metric_names:
                baseline_scores = _per_query_scores(baseline_run, qrels, metric)
                treatment_scores = _per_query_scores(treatment_run, qrels, metric)
                record = paired_bootstrap_diff(
                    baseline_scores,
                    treatment_scores,
                    n_resamples=n_resamples,
                    seed=seed,
                )
                deltas = [
                    treatment - baseline
                    for baseline, treatment in zip(
                        baseline_scores,
                        treatment_scores,
                    )
                ]
                record["wins"] = sum(delta > 0 for delta in deltas)
                record["ties"] = sum(delta == 0 for delta in deltas)
                record["losses"] = sum(delta < 0 for delta in deltas)
                metrics[metric] = record
            cutoffs = (100, 1000) if system == "bm25" else (100,)
            system_comparisons[label] = {
                "baseline": baseline_name,
                "treatment": treatment_name,
                "metrics": metrics,
                "no_hit_queries": {
                    f"@{cutoff}": _no_hit_counts(
                        baseline_run,
                        treatment_run,
                        qrels,
                        cutoff=cutoff,
                    )
                    for cutoff in cutoffs
                },
            }
        comparisons[system] = system_comparisons
    return comparisons


def _validate_expected_paired(
    observed: Mapping[str, Any],
    expected: Any,
    *,
    tolerance: float,
) -> float:
    if not isinstance(expected, dict):
        raise ReleaseArtifactError("bundle manifest has no paired expectations")
    max_delta = 0.0
    for system, comparisons in expected.items():
        if not isinstance(comparisons, dict) or system not in observed:
            raise ReleaseArtifactError(f"invalid paired expectation stage: {system}")
        for comparison, metrics in comparisons.items():
            try:
                observed_metrics = observed[system][comparison]["metrics"]
            except (KeyError, TypeError) as exc:
                raise ReleaseArtifactError(
                    f"missing reproduced paired comparison: {system}/{comparison}"
                ) from exc
            if not isinstance(metrics, dict):
                raise ReleaseArtifactError("paired metric expectations must be objects")
            for metric, expected_record in metrics.items():
                if not isinstance(expected_record, dict) or metric not in observed_metrics:
                    raise ReleaseArtifactError(
                        f"invalid paired expectation: {system}/{comparison}/{metric}"
                    )
                for field in ("mean_delta", "ci_low", "ci_high", "p_two_sided"):
                    delta = abs(
                        float(observed_metrics[metric][field])
                        - float(expected_record[field])
                    )
                    max_delta = max(max_delta, delta)
                    if delta > tolerance:
                        raise ReleaseArtifactError(
                            "paired reproduction drift for "
                            f"{system}/{comparison}/{metric}/{field}: "
                            f"{delta:.3g} exceeds {tolerance:g}"
                        )
    return max_delta


def _write_reproduction_table(path: Path, results: Mapping[str, Any]) -> None:
    lines = [
        "# Reproduced NFCorpus video query-representation metrics",
        "",
        "| Representation | System | MRR@10 | nDCG@10 | Recall@100 | "
        "Recall@1000 | Max abs. delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "title": "Title",
        "description": "Description",
        "title_plus_description": "Title + description",
    }
    for representation in REPRESENTATIONS:
        for system in ("bm25", "bm25_ce"):
            record = results[f"{representation}_{system}"]
            metrics = record["metrics"]
            recall_1000 = (
                f"{metrics['recall@1000']:.4f}"
                if "recall@1000" in metrics
                else "n/a"
            )
            lines.append(
                f"| {labels[representation]} | {system} | "
                f"{metrics['mrr@10']:.4f} | {metrics['ndcg@10']:.4f} | "
                f"{metrics['recall@100']:.4f} | {recall_1000} | "
                f"{record['max_abs_delta']:.2e} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def evaluate_release_bundle(
    *,
    bundle_dir: Path | str,
    output_dir: Path | str,
    qrels_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Recompute all six aggregate rows and the frozen paired comparisons."""
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ReleaseArtifactError("tolerance must be finite and non-negative")
    source = Path(bundle_dir).resolve()
    manifest = _load_json_object(source / MANIFEST_NAME)
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ReleaseArtifactError(f"bundle directory does not use {BUNDLE_SCHEMA}")
    _verify_extracted_directory(source, manifest)

    scope = manifest.get("scope")
    if not isinstance(scope, dict):
        raise ReleaseArtifactError("bundle manifest has no scope")
    dataset_id = str(scope.get("dataset", ""))
    expected_queries = int(scope.get("query_count", 0))
    expected_qid_hash = str(scope.get("qid_sha256", ""))
    bm25_depth = int(scope.get("bm25_depth", 0))
    rerank_depth = int(scope.get("rerank_depth", 0))
    n_resamples = int(scope.get("bootstrap_resamples", 0))
    seed = int(scope.get("bootstrap_seed", 0))
    if dataset_id != DEFAULT_DATASET_ID or min(
        expected_queries,
        bm25_depth,
        rerank_depth,
        n_resamples,
    ) < 1:
        raise ReleaseArtifactError("bundle manifest has an invalid scope")

    run_records = [
        record
        for record in _manifest_members(manifest).values()
        if record.get("kind") == "nfcorpus_video_run"
    ]
    if len(run_records) != len(RUN_SPECS):
        raise ReleaseArtifactError("bundle does not contain exactly six run records")
    runs: dict[tuple[str, str], dict[str, list[tuple[str, float]]]] = {}
    record_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    reference_qids: set[str] | None = None
    for record in run_records:
        representation = str(record.get("representation", ""))
        system = str(record.get("system", ""))
        key = (representation, system)
        if representation not in REPRESENTATIONS or system not in SYSTEM_METRICS:
            raise ReleaseArtifactError(f"invalid run identity: {key}")
        if key in runs:
            raise ReleaseArtifactError(f"duplicate run identity: {key}")
        run = read_run_tsv(source / str(record["path"]))
        expected_depth = bm25_depth if system == "bm25" else rerank_depth
        qids = set(run)
        row_count = sum(len(rows) for rows in run.values())
        if (
            len(qids) != expected_queries
            or row_count != int(record.get("row_count", -1))
            or any(len(rows) != expected_depth for rows in run.values())
        ):
            raise ReleaseArtifactError(f"{record['path']}: run structure drift")
        if reference_qids is None:
            reference_qids = qids
        elif qids != reference_qids:
            raise ReleaseArtifactError("query-id sets differ across extracted runs")
        runs[key] = run
        record_by_key[key] = record

    if reference_qids is None or _qid_sha256(reference_qids) != expected_qid_hash:
        raise ReleaseArtifactError("extracted run qids do not match the frozen cohort")
    candidate_checks = _validate_candidate_sets(runs, rerank_depth=rerank_depth)
    if candidate_checks != int(scope.get("candidate_set_checks", -1)):
        raise ReleaseArtifactError("candidate-set check count differs from the manifest")

    resolved_qrels_path = Path(qrels_path).resolve() if qrels_path is not None else None
    resolved_cache_dir = Path(cache_dir).resolve() if cache_dir is not None else None
    all_qrels, qrels_source = _load_qrels(
        dataset_id,
        qrels_path=resolved_qrels_path,
        cache_dir=resolved_cache_dir,
    )
    missing_qrels = sorted(reference_qids - set(all_qrels))
    if missing_qrels:
        raise ReleaseArtifactError(
            f"public qrels are missing {len(missing_qrels)} frozen query ids"
        )
    qrels = {qid: dict(all_qrels[qid]) for qid in sorted(reference_qids)}

    results: dict[str, Any] = {}
    aggregate_max_delta = 0.0
    for spec in RUN_SPECS:
        key = (spec.representation, spec.system)
        record = record_by_key[key]
        metrics = evaluate_internal_trec_scope(
            runs[key],
            qrels,
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
                f"{record['path']}: reproduced metrics drift by {max_delta:.3g}; "
                f"tolerance is {tolerance:g}"
            )
        aggregate_max_delta = max(aggregate_max_delta, max_delta)
        reproduced = {name: float(metrics[name]) for name in expected_metrics}
        reproduced["n_queries"] = int(metrics["n_queries"])
        results[f"{spec.representation}_{spec.system}"] = {
            "representation": spec.representation,
            "system": spec.system,
            "run": record["path"],
            "metrics": reproduced,
            "expected_metrics": expected_metrics,
            "absolute_deltas": deltas,
            "max_abs_delta": max_delta,
        }

    paired = _paired_comparisons(
        runs,
        qrels,
        n_resamples=n_resamples,
        seed=seed,
    )
    paired_max_delta = _validate_expected_paired(
        paired,
        manifest.get("expected_paired_comparisons"),
        tolerance=tolerance,
    )
    report = {
        "schema": REPRODUCTION_SCHEMA,
        "artifact_id": manifest.get("artifact_id"),
        "experiment_commits": manifest.get("experiment_commits"),
        "verified": True,
        "qrels_source": qrels_source,
        "tolerance": tolerance,
        "aggregate_max_abs_delta": aggregate_max_delta,
        "paired_max_abs_delta": paired_max_delta,
        "candidate_set_checks": candidate_checks,
        "results": results,
        "paired_comparisons": paired,
    }
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_bytes(_canonical_json_bytes(report))
    _write_reproduction_table(destination / "metrics.md", results)
    return report
