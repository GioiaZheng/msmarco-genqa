"""TREC-compatible run validation and metric cross-checking.

The repository's retrieval stages already write six-column TREC run files.
This module adds the missing boundary around those artifacts: strict qrels
parsing, deterministic canonical exports, and an optional independent metric
calculation through :mod:`ir_measures`.
"""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from typing import Any, Literal

from msmarco_genqa.evaluation.retrieval import evaluate_retrieval
from msmarco_genqa.reranking.io import read_run_tsv


QrelsFormat = Literal["auto", "trec", "irds-tsv", "three-column"]


class QrelsFormatError(ValueError):
    """Raised when a qrels file is malformed or ambiguous."""

    def __init__(self, path: Path | str, line_number: int | None, message: str) -> None:
        self.path = Path(path)
        self.line_number = line_number
        self.reason = message
        location = str(self.path) if line_number is None else f"{self.path}:{line_number}"
        super().__init__(f"{location}: {message}")


class OptionalEvaluatorUnavailable(RuntimeError):
    """Raised when a requested optional evaluator is not installed."""


class MetricCrossCheckError(RuntimeError):
    """Raised when independent and internal metric results disagree."""


def _parse_relevance(path: Path, line_number: int, value: str) -> int:
    try:
        relevance = int(value)
    except ValueError as exc:
        raise QrelsFormatError(
            path,
            line_number,
            f"relevance is not an integer: {value!r}",
        ) from exc
    return relevance


def _parse_qrels_fields(
    path: Path,
    line_number: int,
    fields: list[str],
    qrels_format: QrelsFormat,
) -> tuple[str, str, int]:
    if qrels_format == "three-column" or (qrels_format == "auto" and len(fields) == 3):
        if len(fields) != 3:
            raise QrelsFormatError(path, line_number, "expected 3 fields")
        qid, doc_id, relevance_text = fields
        return qid, doc_id, _parse_relevance(path, line_number, relevance_text)

    if len(fields) != 4:
        raise QrelsFormatError(
            path,
            line_number,
            f"expected 4 fields, got {len(fields)}",
        )

    if qrels_format == "trec":
        qid, _iteration, doc_id, relevance_text = fields
    elif qrels_format == "irds-tsv":
        qid, doc_id, relevance_text, _iteration = fields
    elif qrels_format == "auto":
        # TREC qrels are ``qid iteration doc_id relevance``; ir_datasets TSV
        # exports are ``qid doc_id relevance iteration``. The canonical MS
        # MARCO iteration is 0, which makes these layouts distinguishable in
        # normal files. Ambiguous numeric rows must be resolved explicitly.
        if fields[1] == "0" and fields[3] != "0":
            qid, _iteration, doc_id, relevance_text = fields
        elif fields[3] == "0" and fields[1] != "0":
            qid, doc_id, relevance_text, _iteration = fields
        else:
            raise QrelsFormatError(
                path,
                line_number,
                "ambiguous 4-field qrels row; pass --qrels-format trec or irds-tsv",
            )
    else:  # pragma: no cover - argparse and the Literal type guard this.
        raise ValueError(f"unsupported qrels format: {qrels_format}")
    return qid, doc_id, _parse_relevance(path, line_number, relevance_text)


def read_qrels(
    path: Path | str,
    *,
    qrels_format: QrelsFormat = "auto",
) -> dict[str, dict[str, int]]:
    """Read TREC, ir_datasets TSV, or three-column qrels.

    Duplicate ``(qid, doc_id)`` judgments fail fast. Relevance values are
    retained as integers; retrieval metrics treat values greater than zero as
    relevant, matching the binary MS MARCO dev/small judgments.
    """
    p = Path(path)
    qrels: dict[str, dict[str, int]] = {}
    seen: set[tuple[str, str]] = set()
    try:
        with p.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    raise QrelsFormatError(p, line_number, "empty line")
                qid, doc_id, relevance = _parse_qrels_fields(
                    p,
                    line_number,
                    line.split(),
                    qrels_format,
                )
                if not qid:
                    raise QrelsFormatError(p, line_number, "empty query id")
                if not doc_id:
                    raise QrelsFormatError(p, line_number, "empty document id")
                if "\ufffd" in qid or "\ufffd" in doc_id:
                    raise QrelsFormatError(
                        p,
                        line_number,
                        "replacement character found in identifier field",
                    )
                key = (qid, doc_id)
                if key in seen:
                    raise QrelsFormatError(
                        p,
                        line_number,
                        f"duplicate judgment for query {qid!r}, document {doc_id!r}",
                    )
                seen.add(key)
                qrels.setdefault(qid, {})[doc_id] = relevance
    except UnicodeDecodeError as exc:
        raise QrelsFormatError(p, None, f"file is not valid UTF-8: {exc}") from exc
    if not qrels:
        raise QrelsFormatError(p, None, "qrels file contains no judgments")
    return qrels


def _positive_qrels(qrels: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {
        qid: judgments
        for qid, judgments in qrels.items()
        if any(relevance > 0 for relevance in judgments.values())
    }


def write_canonical_run(
    path: Path | str,
    run: dict[str, list[tuple[str, float]]],
    *,
    run_id: str,
) -> None:
    """Write a deterministic six-column run with rank-preserving scores.

    TREC evaluators rank by score rather than trusting the rank column. The
    exported score is therefore ``1 / rank`` so tied model scores cannot
    silently reorder documents. Original model scores remain in the source
    ``run.tsv`` artifact.
    """
    if not run_id or any(char.isspace() for char in run_id) or "\ufffd" in run_id:
        raise ValueError("run_id must be a non-empty identifier without whitespace")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as handle:
        for qid in sorted(run):
            for rank, (doc_id, _original_score) in enumerate(run[qid], start=1):
                handle.write(
                    f"{qid}\tQ0\t{doc_id}\t{rank}\t{1.0 / rank:.12g}\t{run_id}\n"
                )


def write_canonical_qrels(
    path: Path | str,
    qrels: dict[str, dict[str, int]],
) -> None:
    """Write deterministic four-column TREC qrels."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as handle:
        for qid in sorted(qrels):
            for doc_id in sorted(qrels[qid]):
                handle.write(f"{qid}\t0\t{doc_id}\t{qrels[qid][doc_id]}\n")


def evaluate_internal_trec_scope(
    run: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]],
) -> dict[str, float]:
    """Evaluate all positive-qrels topics, including absent runs as zeroes."""
    positive = _positive_qrels(qrels)
    ranked_docs = {
        qid: [doc_id for doc_id, _score in run.get(qid, [])]
        for qid in positive
    }
    binary_qrels = {
        qid: {doc_id for doc_id, relevance in judgments.items() if relevance > 0}
        for qid, judgments in positive.items()
    }
    return evaluate_retrieval(
        ranked_docs,
        binary_qrels,
        ks_mrr=(10,),
        ks_ndcg=(10,),
        ks_recall=(1000,),
    )


def evaluate_ir_measures(
    run: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]],
    *,
    module: Any | None = None,
) -> dict[str, float]:
    """Evaluate with the optional :mod:`ir_measures` backend."""
    if module is None:
        try:
            module = importlib.import_module("ir_measures")
        except ModuleNotFoundError as exc:
            raise OptionalEvaluatorUnavailable(
                "ir-measures is not installed; install the evaluation extra with "
                "pip install -e '.[evaluation]'"
            ) from exc

    positive = _positive_qrels(qrels)
    # Rank-preserving scores match the canonical export and remove evaluator
    # ambiguity when the original retrieval backend emitted tied scores.
    scored_run = {
        qid: {
            doc_id: 1.0 / rank
            for rank, (doc_id, _score) in enumerate(run.get(qid, []), start=1)
        }
        for qid in positive
    }
    measures = {
        "mrr@10": module.RR @ 10,
        "ndcg@10": module.nDCG @ 10,
        "recall@1000": module.R @ 1000,
    }
    raw = module.calc_aggregate(list(measures.values()), positive, scored_run)
    return {name: float(raw[measure]) for name, measure in measures.items()}


def compare_metric_sets(
    internal: dict[str, float],
    external: dict[str, float],
    *,
    tolerance: float,
) -> dict[str, float]:
    """Return absolute deltas or raise when any headline metric disagrees."""
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be a finite non-negative number")
    names = ("mrr@10", "ndcg@10", "recall@1000")
    missing = [name for name in names if name not in internal or name not in external]
    if missing:
        raise MetricCrossCheckError(f"missing metrics: {', '.join(missing)}")
    deltas = {name: abs(float(internal[name]) - float(external[name])) for name in names}
    failures = {name: delta for name, delta in deltas.items() if delta > tolerance}
    if failures:
        details = ", ".join(f"{name} delta={delta:.3g}" for name, delta in failures.items())
        raise MetricCrossCheckError(
            f"internal and external metrics differ beyond tolerance {tolerance:g}: {details}"
        )
    return deltas


def run_trec_cross_check(
    *,
    run_path: Path | str,
    qrels_path: Path | str,
    output_dir: Path | str,
    run_id: str = "msmarco-genqa",
    qrels_format: QrelsFormat = "auto",
    backend: Literal["auto", "ir-measures", "none"] = "auto",
    tolerance: float = 1e-12,
    ir_measures_module: Any | None = None,
) -> dict[str, Any]:
    """Validate artifacts, export canonical files, and cross-check metrics."""
    run_source = Path(run_path)
    qrels_source = Path(qrels_path)
    destination = Path(output_dir)
    run = read_run_tsv(run_source)
    if not run:
        raise ValueError(f"{run_source}: run file contains no records")
    qrels = read_qrels(qrels_source, qrels_format=qrels_format)
    relevance_levels = {
        relevance
        for judgments in qrels.values()
        for relevance in judgments.values()
    }
    if not relevance_levels <= {0, 1}:
        raise ValueError(
            f"{qrels_source}: only binary relevance levels 0 and 1 are supported; "
            f"got {sorted(relevance_levels)}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    canonical_run = destination / "run.trec"
    canonical_qrels = destination / "qrels.trec"
    write_canonical_run(canonical_run, run, run_id=run_id)
    write_canonical_qrels(canonical_qrels, qrels)

    internal = evaluate_internal_trec_scope(run, qrels)
    if not internal.get("n_queries"):
        raise ValueError(f"{qrels_source}: qrels contain no positive judgments")
    report: dict[str, Any] = {
        "schema_version": 1,
        "source_run": str(run_source),
        "source_qrels": str(qrels_source),
        "canonical_run": str(canonical_run),
        "canonical_qrels": str(canonical_qrels),
        "run_id": run_id,
        "qrels_format": qrels_format,
        "scope": {
            "run_queries": len(run),
            "qrels_queries": len(qrels),
            "evaluated_queries": int(internal.get("n_queries", 0)),
        },
        "internal_metrics": internal,
        "external_evaluator": {
            "backend": None,
            "status": "disabled" if backend == "none" else "unavailable",
            "tolerance": tolerance,
        },
    }
    cross_check_error: MetricCrossCheckError | None = None
    if backend != "none":
        try:
            external = evaluate_ir_measures(
                run,
                qrels,
                module=ir_measures_module,
            )
        except OptionalEvaluatorUnavailable as exc:
            if backend == "ir-measures":
                raise
            report["external_evaluator"]["reason"] = str(exc)
        else:
            try:
                deltas = compare_metric_sets(internal, external, tolerance=tolerance)
            except MetricCrossCheckError as exc:
                deltas = {
                    name: abs(float(internal[name]) - float(external[name]))
                    for name in ("mrr@10", "ndcg@10", "recall@1000")
                    if name in internal and name in external
                }
                cross_check_error = exc
                status = "failed"
            else:
                status = "passed"
            report["external_evaluator"].update(
                {
                    "backend": "ir-measures",
                    "status": status,
                    "metrics": external,
                    "absolute_deltas": deltas,
                }
            )
            if cross_check_error is not None:
                report["external_evaluator"]["reason"] = str(cross_check_error)

    metrics_path = destination / "metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if cross_check_error is not None:
        raise cross_check_error
    return report
