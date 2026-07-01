"""Export selected MS MARCO GenQA rows for rag-observatory ingestion."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from msmarco_genqa.evaluation.rag_triad import (
    DEFAULT_LOW_SCORE_THRESHOLD,
    score_prediction_row,
)


EXPORT_FORMAT = "msmarco-genqa.trace-export.v1"
SWEEP_EXPORT_FORMAT = "msmarco-genqa.trace-sweep.v1"

_TOP_LEVEL_FIELDS = {
    "format",
    "run",
    "query",
    "retrieved_documents",
    "reranked_documents",
    "selected_context",
    "prompt",
    "answer",
    "metrics",
    "failures",
    "diagnostic_notes",
    "extra",
}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class RagObservatoryExportError(ValueError):
    """Raised when an export cannot satisfy the rag-observatory adapter contract."""


def load_prediction_rows(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RagObservatoryExportError(f"{p}:{line_number}: invalid JSONL row") from exc
            if not isinstance(row, dict):
                raise RagObservatoryExportError(f"{p}:{line_number}: row must be a JSON object")
            if not row.get("query_id"):
                raise RagObservatoryExportError(f"{p}:{line_number}: missing query_id")
            rows.append(row)
    if not rows:
        raise RagObservatoryExportError(f"{p}: no prediction rows found")
    return rows


def select_prediction_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    query_id: str | None = None,
) -> Mapping[str, Any]:
    if query_id is None:
        return rows[0]
    matches = [row for row in rows if str(row.get("query_id")) == query_id]
    if not matches:
        raise RagObservatoryExportError(f"query_id not found in predictions: {query_id}")
    if len(matches) > 1:
        raise RagObservatoryExportError(f"query_id appears more than once: {query_id}")
    return matches[0]


def load_qrels(path: Path | str | None) -> dict[str, set[str]] | None:
    if path is None:
        return None
    from msmarco_genqa.evaluation.retrieval_report import load_qrels_tsv

    return load_qrels_tsv(path)


def build_trace_export(
    row: Mapping[str, Any],
    *,
    run_id: str | None = None,
    timestamp: str | None = None,
    dataset: str | None = None,
    config_hash: str | None = None,
    code_version: str | None = None,
    retriever: str | None = None,
    reranker: str | None = None,
    generator: str | None = None,
    evaluator: str | None = "deterministic-rag-triad",
    random_seed: int | None = None,
    config_id: str | None = None,
    config: Mapping[str, Any] | None = None,
    qrels: Mapping[str, set[str]] | None = None,
    manifest_path: str | None = None,
    source_predictions: str | None = None,
    source_qrels: str | None = None,
    export_profile: str = "single-query",
    low_score_threshold: float = DEFAULT_LOW_SCORE_THRESHOLD,
) -> dict[str, Any]:
    qid = _required_text(row, "query_id", "prediction row")
    query_text = _required_text(row, "query", "prediction row")
    references = _text_list(row.get("references"))
    top_doc_ids = _text_list(row.get("top_doc_ids"))
    passages = _text_list(row.get("passages"))
    if len(top_doc_ids) != len(passages):
        raise RagObservatoryExportError(
            "prediction row must provide the same number of top_doc_ids and passages"
        )
    if not top_doc_ids:
        raise RagObservatoryExportError("prediction row must include at least one context document")

    relevant_doc_ids = qrels.get(qid, set()) if qrels is not None else None
    if config_id is not None:
        _validate_safe_id(config_id, "config_id")
    scored = score_prediction_row(
        row,
        config_name=str(row.get("retrieval_source") or retriever or "unknown"),
        qrels=qrels,
        low_score_threshold=low_score_threshold,
    )
    citations = _citations(row)
    retrieved_documents = _documents(
        doc_ids=top_doc_ids,
        passages=passages,
        scores=_optional_float_list(row.get("top_doc_scores")),
        source=str(row.get("retrieval_source") or retriever or "prediction-row"),
        relevant_doc_ids=relevant_doc_ids,
    )
    reranked_documents = _reranked_documents(row, relevant_doc_ids=relevant_doc_ids)
    selected_context = _selected_context(qid=qid, doc_ids=top_doc_ids, passages=passages)
    prompt = _prompt(row)
    model_answer = _required_text(row, "prediction", "prediction row")

    notes = [
        {
            "stage": "export",
            "note": "Single-query export produced from a prediction JSONL row.",
        }
    ]
    if not citations:
        notes.append(
            {
                "stage": "generation",
                "note": "No explicit answer citation spans were present in the source row.",
            }
        )
    if reranked_documents is None:
        notes.append(
            {
                "stage": "reranking",
                "note": "Reranked candidates were not present in the source row.",
            }
        )
    if prompt is None:
        notes.append(
            {
                "stage": "prompt",
                "note": "Prompt text was not present in the source row.",
            }
        )

    export = {
        "format": EXPORT_FORMAT,
        "run": {
            "run_id": run_id or f"msmarco-genqa-{qid}",
            "timestamp": timestamp
            or datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "dataset": dataset or "msmarco-genqa",
            "config_hash": config_hash,
            "code_version": code_version,
            "retriever": retriever or str(row.get("retrieval_source") or "unknown"),
            "reranker": reranker,
            "generator": generator,
            "evaluator": evaluator,
            "random_seed": random_seed,
            "pipeline_stages": {
                "retrieval": True,
                "reranking": reranked_documents is not None,
                "context_selection": True,
                "generation": True,
                "citations": bool(citations),
                "evaluation": True,
            },
            "extra": {
                "source_repository": "msmarco-genqa",
                "export_profile": export_profile,
                "manifest_path": manifest_path,
                "config_id": config_id,
                "config": dict(config or {}),
            },
        },
        "query": {
            "query_id": qid,
            "text": query_text,
            "gold_answer": references[0] if references else None,
            "extra": {
                "references": references,
                "query_type": row.get("query_type"),
            },
        },
        "retrieved_documents": retrieved_documents,
        "reranked_documents": reranked_documents,
        "selected_context": selected_context,
        "prompt": prompt,
        "answer": {
            "text": model_answer,
            "citations": citations,
            "extra": {
                "retrieval_source": row.get("retrieval_source"),
            },
        },
        "metrics": _metrics(scored, low_score_threshold=low_score_threshold),
        "failures": [],
        "diagnostic_notes": notes,
        "extra": {
            "source_predictions": source_predictions,
            "source_qrels": source_qrels,
            "low_dimensions": scored["flags"]["low_dimensions"],
        },
    }
    validate_trace_export(export)
    return export


def build_sweep_manifest(
    trace_exports: Sequence[Mapping[str, Any]],
    *,
    trace_paths: Sequence[str],
    sweep_id: str,
    timestamp: str,
    dataset: str,
    supported_dimensions: Sequence[str] | None = None,
    deferred_dimensions: Sequence[str] | None = None,
) -> dict[str, Any]:
    if not trace_exports:
        raise RagObservatoryExportError("sweep must contain at least one trace export")
    if len(trace_exports) != len(trace_paths):
        raise RagObservatoryExportError("trace_exports and trace_paths must have the same length")
    _validate_safe_id(sweep_id, "sweep_id")

    configs = []
    query_ids: set[str] = set()
    metric_names: set[str] = set()
    rows = []
    for export, trace_path in zip(trace_exports, trace_paths):
        validate_trace_export(export)
        run = _expect_mapping(export["run"], "export.run")
        run_extra = _expect_mapping(run.get("extra", {}), "export.run.extra")
        config_id = str(run_extra.get("config_id") or run.get("run_id"))
        _validate_safe_id(config_id, "config_id")
        query = _expect_mapping(export["query"], "export.query")
        query_id = str(query["query_id"])
        query_ids.add(query_id)
        metrics = _metric_map(export)
        metric_names.update(metrics)

        config = dict(_expect_mapping(run_extra.get("config", {}), "export.run.extra.config"))
        config.setdefault("retriever", run.get("retriever"))
        config.setdefault("reranker", run.get("reranker"))
        config.setdefault("generator", run.get("generator"))
        config.setdefault("top_k", len(_expect_list(export["selected_context"], "selected_context")))
        config.setdefault("has_reranked_documents", export.get("reranked_documents") is not None)
        configs.append(
            {
                "config_id": config_id,
                "run_id": run.get("run_id"),
                "trace_path": trace_path,
                "query_id": query_id,
                "config": config,
            }
        )
        rows.append(
            {
                "config_id": config_id,
                "trace_path": trace_path,
                "query_id": query_id,
                "metrics": metrics,
            }
        )

    manifest = {
        "format": SWEEP_EXPORT_FORMAT,
        "sweep": {
            "sweep_id": sweep_id,
            "timestamp": timestamp,
            "dataset": dataset,
            "export_contract": EXPORT_FORMAT,
            "query_ids": sorted(query_ids),
            "supported_dimensions": list(
                supported_dimensions
                or ["config_id", "retriever", "reranker", "generator", "top_k"]
            ),
            "deferred_dimensions": list(
                deferred_dimensions or ["query_rewriting", "context_compression"]
            ),
        },
        "configurations": configs,
        "comparison": {
            "metric_names": sorted(metric_names),
            "rows": rows,
        },
    }
    validate_sweep_manifest(manifest)
    return manifest


def validate_sweep_manifest(manifest: Mapping[str, Any]) -> None:
    data = _expect_mapping(manifest, "sweep manifest")
    if data.get("format") != SWEEP_EXPORT_FORMAT:
        raise RagObservatoryExportError(
            f"sweep manifest format must be {SWEEP_EXPORT_FORMAT!r}"
        )
    sweep = _expect_mapping(data.get("sweep"), "sweep manifest.sweep")
    _validate_safe_id(_required_text(sweep, "sweep_id", "sweep"), "sweep_id")
    _required_text(sweep, "timestamp", "sweep")
    _required_text(sweep, "dataset", "sweep")
    if sweep.get("export_contract") != EXPORT_FORMAT:
        raise RagObservatoryExportError("sweep.export_contract must match the trace export format")
    configurations = _expect_list(data.get("configurations"), "sweep manifest.configurations")
    if not configurations:
        raise RagObservatoryExportError("sweep manifest must contain at least one configuration")
    seen_ids: set[str] = set()
    for index, item in enumerate(configurations):
        config = _expect_mapping(item, f"configurations[{index}]")
        config_id = _required_text(config, "config_id", f"configurations[{index}]")
        _validate_safe_id(config_id, f"configurations[{index}].config_id")
        if config_id in seen_ids:
            raise RagObservatoryExportError(f"duplicate config_id in sweep: {config_id}")
        seen_ids.add(config_id)
        _required_text(config, "trace_path", f"configurations[{index}]")
        _expect_mapping(config.get("config"), f"configurations[{index}].config")
    comparison = _expect_mapping(data.get("comparison"), "sweep manifest.comparison")
    _expect_list(comparison.get("rows"), "sweep manifest.comparison.rows")


def write_sweep_manifest(path: Path | str, manifest: Mapping[str, Any]) -> Path:
    validate_sweep_manifest(manifest)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def write_trace_export(path: Path | str, export: Mapping[str, Any]) -> Path:
    validate_trace_export(export)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(export, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def validate_trace_export(export: Mapping[str, Any]) -> None:
    data = _expect_mapping(export, "export")
    _reject_unknown(data, _TOP_LEVEL_FIELDS, "export")
    if data.get("format") != EXPORT_FORMAT:
        raise RagObservatoryExportError(f"export.format must be {EXPORT_FORMAT!r}")
    _validate_run(_expect_mapping(data.get("run"), "export.run"))
    _validate_query(_expect_mapping(data.get("query"), "export.query"))
    _validate_documents(_expect_list(data.get("retrieved_documents"), "export.retrieved_documents"))
    reranked = data.get("reranked_documents")
    if reranked is not None:
        _validate_documents(_expect_list(reranked, "export.reranked_documents"))
    _validate_context(_expect_list(data.get("selected_context"), "export.selected_context"))
    prompt = data.get("prompt")
    if prompt is not None:
        _expect_mapping(prompt, "export.prompt")
    _validate_answer(_expect_mapping(data.get("answer"), "export.answer"))
    _expect_list(data.get("metrics", []), "export.metrics")
    _expect_list(data.get("failures", []), "export.failures")
    _expect_list(data.get("diagnostic_notes", []), "export.diagnostic_notes")
    _expect_mapping(data.get("extra", {}), "export.extra")


def _documents(
    *,
    doc_ids: Sequence[str],
    passages: Sequence[str],
    scores: Sequence[float] | None,
    source: str,
    relevant_doc_ids: set[str] | None,
    id_namespace: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (doc_id, passage) in enumerate(zip(doc_ids, passages), start=1):
        export_doc_id = f"{id_namespace}:{doc_id}" if id_namespace else doc_id
        extra = {"original_doc_id": doc_id} if id_namespace else {}
        rows.append(
            {
                "doc_id": export_doc_id,
                "text": _non_empty_passage(passage, doc_id=doc_id),
                "title": None,
                "source": source,
                "score": None if scores is None or index > len(scores) else scores[index - 1],
                "rank": index,
                "is_relevant": None if relevant_doc_ids is None else doc_id in relevant_doc_ids,
                "extra": extra,
            }
        )
    return rows


def _reranked_documents(
    row: Mapping[str, Any],
    *,
    relevant_doc_ids: set[str] | None,
) -> list[dict[str, Any]] | None:
    doc_ids = _text_list(row.get("reranked_doc_ids"))
    passages = _text_list(row.get("reranked_passages"))
    if not doc_ids and not passages:
        return None
    if len(doc_ids) != len(passages):
        raise RagObservatoryExportError(
            "reranked_doc_ids and reranked_passages must have the same length"
        )
    return _documents(
        doc_ids=doc_ids,
        passages=passages,
        scores=_optional_float_list(row.get("reranked_scores")),
        source="reranked",
        relevant_doc_ids=relevant_doc_ids,
        id_namespace="reranked",
    )


def _selected_context(*, qid: str, doc_ids: Sequence[str], passages: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "context_id": f"{qid}:ctx:{rank}",
            "doc_id": doc_id,
            "text": _non_empty_passage(passage, doc_id=doc_id),
            "rank": rank,
            "token_count": len(passage.split()),
            "extra": {
                "source_rank": rank,
            },
        }
        for rank, (doc_id, passage) in enumerate(zip(doc_ids, passages), start=1)
    ]


def _prompt(row: Mapping[str, Any]) -> dict[str, Any] | None:
    value = row.get("prompt")
    if value is None:
        return None
    if isinstance(value, str):
        return {
            "content": value,
            "template_id": _optional_text(row.get("prompt_template_id")),
            "variables": {},
            "extra": {},
        }
    if isinstance(value, Mapping):
        return dict(value)
    raise RagObservatoryExportError("prompt must be a string, object, or null")


def _citations(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("citations")
    if raw is None:
        return []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RagObservatoryExportError("citations must be a list")
    citations: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        data = _expect_mapping(item, f"citations[{index}]")
        doc_id = _required_text(data, "doc_id", f"citations[{index}]")
        citations.append(
            {
                "doc_id": doc_id,
                "quote": _optional_text(data.get("quote")),
                "span_start": data.get("span_start"),
                "span_end": data.get("span_end"),
                "extra": dict(data.get("extra") or {}),
            }
        )
    return citations


def _metrics(scored: Mapping[str, Any], *, low_score_threshold: float) -> list[dict[str, Any]]:
    metrics = []
    for name, value in sorted(scored["scores"].items()):
        threshold = low_score_threshold if name in {
            "context_relevance",
            "groundedness",
            "answer_relevance",
            "triad",
        } else None
        metric = {
            "name": name,
            "value": value,
            "passed": None if threshold is None else float(value) >= threshold,
            "threshold": threshold,
            "notes": "deterministic query-level signal",
        }
        metrics.append(metric)
    return metrics


def _metric_map(export: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for index, item in enumerate(_expect_list(export.get("metrics", []), "export.metrics")):
        metric = _expect_mapping(item, f"metrics[{index}]")
        name = _required_text(metric, "name", f"metrics[{index}]")
        value = metric.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[name] = float(value)
    return values


def _validate_run(data: Mapping[str, Any]) -> None:
    _required_text(data, "run_id", "run")
    _required_text(data, "timestamp", "run")
    pipeline = _expect_mapping(data.get("pipeline_stages", {}), "run.pipeline_stages")
    for key, value in pipeline.items():
        if not isinstance(key, str) or not isinstance(value, bool):
            raise RagObservatoryExportError("run.pipeline_stages must map strings to booleans")


def _validate_query(data: Mapping[str, Any]) -> None:
    _required_text(data, "query_id", "query")
    _required_text(data, "text", "query")


def _validate_documents(values: Sequence[Any]) -> None:
    if not values:
        raise RagObservatoryExportError("document list must not be empty")
    for index, value in enumerate(values):
        data = _expect_mapping(value, f"documents[{index}]")
        _required_text(data, "doc_id", f"documents[{index}]")
        _required_text(data, "text", f"documents[{index}]")


def _validate_context(values: Sequence[Any]) -> None:
    if not values:
        raise RagObservatoryExportError("selected_context must not be empty")
    for index, value in enumerate(values):
        data = _expect_mapping(value, f"selected_context[{index}]")
        _required_text(data, "context_id", f"selected_context[{index}]")
        _required_text(data, "doc_id", f"selected_context[{index}]")
        _required_text(data, "text", f"selected_context[{index}]")


def _validate_answer(data: Mapping[str, Any]) -> None:
    _required_text(data, "text", "answer")
    _expect_list(data.get("citations", []), "answer.citations")


def _text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value if item is not None]
    raise RagObservatoryExportError(f"expected a list of text values, got {type(value).__name__}")


def _optional_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RagObservatoryExportError("score fields must be numeric lists")
    out: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise RagObservatoryExportError("score fields must contain only numbers")
        out.append(float(item))
    return out


def _non_empty_passage(value: str, *, doc_id: str) -> str:
    text = str(value).strip()
    if not text:
        raise RagObservatoryExportError(f"missing passage text for document {doc_id!r}")
    return text


def _required_text(data: Mapping[str, Any], key: str, label: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RagObservatoryExportError(f"{label}.{key} must be a non-empty string")
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RagObservatoryExportError("optional text fields must be strings when present")
    return value


def _expect_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RagObservatoryExportError(f"{label} must be an object")
    return value


def _expect_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RagObservatoryExportError(f"{label} must be a list")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise RagObservatoryExportError(
            f"{label} contains unknown field(s): {', '.join(unknown)}"
        )


def _validate_safe_id(value: str, label: str) -> None:
    if not _SAFE_ID.fullmatch(value):
        raise RagObservatoryExportError(
            f"{label} must start with an alphanumeric character and contain only "
            "letters, numbers, dots, underscores, or hyphens"
        )
