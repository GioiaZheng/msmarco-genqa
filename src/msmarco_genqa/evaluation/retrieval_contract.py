"""Strict data and metric contracts for fixed-output retrieval analysis."""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from msmarco_genqa.evaluation.trec import (
    QrelsFormatError,
    evaluate_internal_trec_scope,
    read_qrels,
    trec_metric_contract,
)
from msmarco_genqa.reranking.io import RunTsvFormatError, read_run_tsv


CONTRACT_SCHEMA = "msmarco-genqa.retrieval-data-metric-contract.v1"
REPORT_SCHEMA = "msmarco-genqa.retrieval-data-metric-contract-report.v1"


class RetrievalContractError(RuntimeError):
    """Raised when a frozen retrieval-analysis input violates its contract."""


def sha256_file(path: Path | str) -> str:
    """Return a lowercase SHA-256 digest without loading the whole file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalContractError(f"{label}: cannot read JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RetrievalContractError(f"{label}: expected a JSON object")
    return value


def _resolve_repo_path(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RetrievalContractError(
            f"{label}: expected a repository-relative POSIX path"
        )
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise RetrievalContractError(f"{label}: unsafe repository path {value!r}")
    path = root.joinpath(*pure.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RetrievalContractError(f"{label}: path escapes project root") from exc
    return path


def _require_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RetrievalContractError(
            f"{label}: expected an integer greater than or equal to {minimum}"
        )
    return value


def _require_unit_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RetrievalContractError(f"{label}: expected a number in [0, 1]")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise RetrievalContractError(f"{label}: expected a finite number in [0, 1]")
    return converted


def _verify_file_record(
    root: Path,
    record: Any,
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, dict):
        raise RetrievalContractError(f"{label}: expected a file record")
    path = _resolve_repo_path(root, record.get("path"), label=f"{label}.path")
    if not path.is_file():
        raise RetrievalContractError(f"{label}: missing input {record.get('path')}")
    expected_bytes = _require_int(record.get("bytes"), label=f"{label}.bytes")
    expected_sha = record.get("sha256")
    if (
        not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or expected_sha.lower() != expected_sha
    ):
        raise RetrievalContractError(f"{label}.sha256: expected a lowercase SHA-256")
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RetrievalContractError(
            f"{label}: byte-size drift; expected {expected_bytes}, got {actual_bytes}"
        )
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise RetrievalContractError(
            f"{label}: SHA-256 drift; expected {expected_sha}, got {actual_sha}"
        )
    return path, {
        "path": path.relative_to(root).as_posix(),
        "bytes": actual_bytes,
        "sha256": actual_sha,
    }


def _load_jsonl_ids_from_zip(
    archive_path: Path,
    *,
    member: Any,
    label: str,
) -> set[str]:
    if not isinstance(member, str) or not member:
        raise RetrievalContractError(f"{label}: archive member is missing")
    ids: set[str] = set()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            with archive.open(member) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", errors="strict")
                for line_number, raw_line in enumerate(text, start=1):
                    line = raw_line.strip()
                    if not line:
                        raise RetrievalContractError(
                            f"{label}:{line_number}: empty JSONL row"
                        )
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RetrievalContractError(
                            f"{label}:{line_number}: invalid JSON: {exc}"
                        ) from exc
                    item_id = value.get("_id") if isinstance(value, dict) else None
                    if (
                        not isinstance(item_id, str)
                        or not item_id
                        or "\ufffd" in item_id
                    ):
                        raise RetrievalContractError(
                            f"{label}:{line_number}: invalid _id"
                        )
                    if item_id in ids:
                        raise RetrievalContractError(
                            f"{label}:{line_number}: duplicate _id {item_id!r}"
                        )
                    ids.add(item_id)
    except KeyError as exc:
        raise RetrievalContractError(
            f"{label}: source archive is missing {member!r}"
        ) from exc
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        raise RetrievalContractError(f"{label}: cannot read source archive: {exc}") from exc
    return ids


def _validate_run_shape(
    run: dict[str, list[tuple[str, float]]],
    record: dict[str, Any],
    *,
    label: str,
) -> dict[str, int]:
    expected_queries = _require_int(
        record.get("query_count"), label=f"{label}.query_count", minimum=1
    )
    expected_rows = _require_int(
        record.get("row_count"), label=f"{label}.row_count", minimum=1
    )
    expected_depth = _require_int(
        record.get("depth"), label=f"{label}.depth", minimum=1
    )
    query_count = len(run)
    row_count = sum(len(rows) for rows in run.values())
    wrong_depth = sorted(qid for qid, rows in run.items() if len(rows) != expected_depth)
    if query_count != expected_queries:
        raise RetrievalContractError(
            f"{label}: expected {expected_queries} queries, got {query_count}"
        )
    if row_count != expected_rows:
        raise RetrievalContractError(
            f"{label}: expected {expected_rows} rows, got {row_count}"
        )
    if wrong_depth:
        preview = ", ".join(wrong_depth[:5])
        raise RetrievalContractError(
            f"{label}: {len(wrong_depth)} queries do not have depth "
            f"{expected_depth}; first: {preview}"
        )
    return {
        "query_count": query_count,
        "row_count": row_count,
        "depth": expected_depth,
    }


def _assert_same_ids(
    actual: set[str],
    expected: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RetrievalContractError(
            f"{label}: identifier mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )


def _assert_subset(actual: set[str], expected: set[str], *, label: str) -> None:
    unknown = sorted(actual - expected)
    if unknown:
        raise RetrievalContractError(
            f"{label}: {len(unknown)} identifiers are absent from the source; "
            f"first: {', '.join(unknown[:5])}"
        )


def verify_retrieval_contract(
    contract_path: Path | str,
    *,
    project_root: Path | str,
) -> dict[str, Any]:
    """Verify frozen NFCorpus inputs and independently recompute BM25 metrics."""
    root = Path(project_root).resolve()
    contract_source = Path(contract_path).resolve()
    contract = _load_json_object(contract_source, label="contract")
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise RetrievalContractError(
            f"contract: expected schema {CONTRACT_SCHEMA!r}"
        )

    inputs = contract.get("inputs")
    if not isinstance(inputs, dict):
        raise RetrievalContractError("contract.inputs: expected an object")
    verified_inputs: dict[str, dict[str, Any]] = {}
    resolved: dict[str, Path] = {}
    for key in ("release_asset", "source_archive", "qrels", "bm25_run", "ce_run"):
        resolved[key], verified_inputs[key] = _verify_file_record(
            root,
            inputs.get(key),
            label=f"inputs.{key}",
        )

    release = contract.get("release")
    if not isinstance(release, dict):
        raise RetrievalContractError("contract.release: expected an object")
    pointer_path = _resolve_repo_path(
        root, release.get("pointer_path"), label="release.pointer_path"
    )
    pointer = _load_json_object(pointer_path, label="release pointer")
    pointer_release = pointer.get("release")
    pointer_download = pointer.get("download")
    if not isinstance(pointer_release, dict) or not isinstance(pointer_download, dict):
        raise RetrievalContractError("release pointer is missing release/download")
    for key in ("repository", "tag", "asset"):
        if pointer_release.get(key) != release.get(key):
            raise RetrievalContractError(
                f"release pointer {key} drift: expected {release.get(key)!r}, "
                f"got {pointer_release.get(key)!r}"
            )
    release_asset = verified_inputs["release_asset"]
    if (
        pointer_download.get("bytes") != release_asset["bytes"]
        or pointer_download.get("sha256") != release_asset["sha256"]
    ):
        raise RetrievalContractError(
            "release asset bytes/SHA-256 do not match the tracked pointer"
        )

    source_record = inputs["source_archive"]
    query_ids = _load_jsonl_ids_from_zip(
        resolved["source_archive"],
        member=source_record.get("query_member"),
        label="source queries",
    )
    corpus_ids = _load_jsonl_ids_from_zip(
        resolved["source_archive"],
        member=source_record.get("corpus_member"),
        label="source corpus",
    )
    try:
        qrels = read_qrels(
            resolved["qrels"],
            qrels_format=str(contract.get("qrels_format", "auto")),
        )
        bm25_run = read_run_tsv(resolved["bm25_run"])
        ce_run = read_run_tsv(resolved["ce_run"])
    except (QrelsFormatError, RunTsvFormatError) as exc:
        raise RetrievalContractError(str(exc)) from exc

    expected_counts = contract.get("expected_counts")
    if not isinstance(expected_counts, dict):
        raise RetrievalContractError("contract.expected_counts: expected an object")
    observed_counts = {
        "source_queries": len(query_ids),
        "corpus_documents": len(corpus_ids),
        "test_queries": len(qrels),
        "qrels_judgments": sum(len(rows) for rows in qrels.values()),
    }
    for key, observed in observed_counts.items():
        expected = _require_int(
            expected_counts.get(key), label=f"expected_counts.{key}", minimum=1
        )
        if observed != expected:
            raise RetrievalContractError(
                f"{key}: expected {expected}, got {observed}"
            )

    test_qids = set(qrels)
    _assert_subset(test_qids, query_ids, label="qrels query ids")
    _assert_same_ids(set(bm25_run), test_qids, label="BM25 query ids")
    _assert_same_ids(set(ce_run), test_qids, label="CE query ids")

    qrels_doc_ids = {
        doc_id for judgments in qrels.values() for doc_id in judgments
    }
    bm25_doc_ids = {
        doc_id for rows in bm25_run.values() for doc_id, _score in rows
    }
    ce_doc_ids = {
        doc_id for rows in ce_run.values() for doc_id, _score in rows
    }
    _assert_subset(qrels_doc_ids, corpus_ids, label="qrels document ids")
    _assert_subset(bm25_doc_ids, corpus_ids, label="BM25 document ids")
    _assert_subset(ce_doc_ids, corpus_ids, label="CE document ids")

    bm25_shape = _validate_run_shape(
        bm25_run, inputs["bm25_run"], label="BM25 run"
    )
    ce_shape = _validate_run_shape(ce_run, inputs["ce_run"], label="CE run")
    candidate_mismatches = [
        qid
        for qid in sorted(test_qids)
        if {doc_id for doc_id, _score in bm25_run[qid][: ce_shape["depth"]]}
        != {doc_id for doc_id, _score in ce_run[qid]}
    ]
    if candidate_mismatches:
        raise RetrievalContractError(
            "CE candidate set differs from BM25 top-100 for "
            f"{len(candidate_mismatches)} queries; first: "
            f"{', '.join(candidate_mismatches[:5])}"
        )

    rel_threshold = _require_int(
        contract.get("binary_relevance_threshold"),
        label="binary_relevance_threshold",
        minimum=1,
    )
    relevance_levels = sorted(
        {
            relevance
            for judgments in qrels.values()
            for relevance in judgments.values()
        }
    )
    if any(relevance < 0 for relevance in relevance_levels):
        raise RetrievalContractError("qrels relevance levels must be non-negative")
    no_positive = sorted(
        qid
        for qid, judgments in qrels.items()
        if not any(relevance >= rel_threshold for relevance in judgments.values())
    )
    if no_positive:
        raise RetrievalContractError(
            f"{len(no_positive)} qrels topics have no relevance >= "
            f"{rel_threshold}; first: {', '.join(no_positive[:5])}"
        )

    metrics = evaluate_internal_trec_scope(
        bm25_run,
        qrels,
        rel_threshold=rel_threshold,
    )
    expected_metrics = contract.get("expected_bm25_metrics")
    if not isinstance(expected_metrics, dict) or not expected_metrics:
        raise RetrievalContractError("expected_bm25_metrics: expected an object")
    tolerance_value = contract.get("metric_tolerance", 0.0)
    if isinstance(tolerance_value, bool) or not isinstance(
        tolerance_value, (int, float)
    ):
        raise RetrievalContractError("metric_tolerance must be a number")
    tolerance = float(tolerance_value)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise RetrievalContractError("metric_tolerance must be finite and non-negative")
    deltas: dict[str, float] = {}
    normalized_expected_metrics: dict[str, float] = {}
    for name, expected in expected_metrics.items():
        if name not in metrics:
            raise RetrievalContractError(f"recomputed metrics are missing {name}")
        normalized_expected = _require_unit_float(
            expected,
            label=f"expected_bm25_metrics.{name}",
        )
        normalized_expected_metrics[name] = normalized_expected
        delta = abs(float(metrics[name]) - normalized_expected)
        deltas[name] = delta
        if delta > tolerance:
            raise RetrievalContractError(
                f"{name}: metric drift {delta:.3g} exceeds tolerance {tolerance:g}"
            )

    try:
        contract_relative = contract_source.relative_to(root).as_posix()
        pointer_relative = pointer_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise RetrievalContractError(
            "contract and release pointer must be inside the project root"
        ) from exc

    return {
        "schema": REPORT_SCHEMA,
        "verified": True,
        "contract": contract_relative,
        "dataset_id": contract.get("dataset_id"),
        "analysis_base_commit": contract.get("analysis_base_commit"),
        "experiment_commit": contract.get("experiment_commit"),
        "release": {
            "pointer_path": pointer_relative,
            "repository": release.get("repository"),
            "tag": release.get("tag"),
            "asset": release.get("asset"),
        },
        "inputs": verified_inputs,
        "scope": {
            **observed_counts,
            "bm25": bm25_shape,
            "ce": ce_shape,
        },
        "integrity": {
            "qrels_queries_covered_by_source": True,
            "run_queries_equal_qrels_queries": True,
            "qrels_documents_covered_by_corpus": True,
            "run_documents_covered_by_corpus": True,
            "bm25_ce_candidate_sets_equal": True,
            "duplicates_rank_gaps_and_non_finite_scores_absent": True,
        },
        "evaluation": trec_metric_contract(
            rel_threshold=rel_threshold,
            ks_mrr=(10,),
            ks_ndcg=(10,),
            ks_recall=(100, 1000),
            run_depth=bm25_shape["depth"],
        ),
        "qrels_relevance_levels": relevance_levels,
        "bm25_metrics": {
            name: float(metrics[name]) for name in expected_metrics
        },
        "expected_bm25_metrics": normalized_expected_metrics,
        "absolute_deltas": deltas,
        "max_abs_delta": max(deltas.values(), default=0.0),
        "metric_tolerance": tolerance,
    }
