"""Deterministic RAG triad evaluation.

The triad report connects three query-level signals:

- context relevance: whether the retrieved context appears useful for the query
- groundedness: whether the generated text is supported by the shown context
- answer relevance: whether the generated text matches the reference answers

The default evaluator is intentionally lightweight and reproducible. It uses
existing qrels when provided, otherwise falls back to lexical query-context
overlap for the context dimension. Groundedness and answer relevance reuse the
repository's deterministic grounding and generation metrics.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from msmarco_genqa.evaluation.generation import exact_match, token_f1
from msmarco_genqa.evaluation.grounding import (
    is_vacuously_grounded_lex,
    is_vacuously_grounded_ngram,
    lexical_grounding,
    ngram_grounding,
)
from msmarco_genqa.evaluation.query_form import classify_question_form
from msmarco_genqa.evaluation.retrieval import (
    first_relevant_rank,
    recall_at_k,
    retrieval_shift_bucket,
)


SUPPORTED_EVALUATORS = ("deterministic",)
DEFAULT_LOW_SCORE_THRESHOLD = 0.5


class UnsupportedEvaluatorError(ValueError):
    """Raised when a triad evaluator is not explicitly supported."""


class PredictionPairingError(ValueError):
    """Raised when prediction files cannot be compared query-by-query."""


def validate_evaluator(evaluator: str) -> str:
    normalised = (evaluator or "").strip().lower()
    if normalised not in SUPPORTED_EVALUATORS:
        allowed = ", ".join(SUPPORTED_EVALUATORS)
        raise UnsupportedEvaluatorError(
            f"unsupported triad evaluator {evaluator!r}; supported: {allowed}"
        )
    return normalised


def load_predictions_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "query_id" not in row:
                raise ValueError(f"{p}:{line_number}: missing query_id")
            rows.append(row)
    return rows


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _clip(values: list[str], top_k: int | None) -> list[str]:
    if top_k is None or top_k <= 0:
        return values
    return values[:top_k]


def _qid_sequence(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row["query_id"]) for row in rows]


def assert_aligned_predictions(predictions_by_config: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    """Require identical query id order across all configs."""
    items = list(predictions_by_config.items())
    if len(items) <= 1:
        return
    base_name, base_rows = items[0]
    base_qids = _qid_sequence(base_rows)
    base_set = set(base_qids)
    for name, rows in items[1:]:
        qids = _qid_sequence(rows)
        if qids == base_qids:
            continue
        if set(qids) == base_set:
            raise PredictionPairingError(
                f"prediction rows for {name!r} contain the same query ids as "
                f"{base_name!r} but in a different order"
            )
        missing = sorted(base_set - set(qids))[:5]
        extra = sorted(set(qids) - base_set)[:5]
        raise PredictionPairingError(
            f"prediction rows for {name!r} do not match {base_name!r}; "
            f"missing={missing}, extra={extra}"
        )


def _context_relevance(
    *,
    qid: str,
    query: str,
    top_doc_ids: Sequence[str],
    passages: Sequence[str],
    qrels: Mapping[str, set[str]] | None,
) -> dict[str, Any]:
    if qrels is not None:
        relevant = set(qrels.get(qid, set()))
        first_rank = first_relevant_rank(top_doc_ids, relevant)
        n_relevant = sum(1 for doc_id in top_doc_ids if doc_id in relevant)
        return {
            "context_relevance": 1.0 if first_rank is not None else 0.0,
            "context_recall": recall_at_k(top_doc_ids, relevant, len(top_doc_ids)),
            "context_metric": "qrels_hit",
            "n_relevant_in_context": n_relevant,
            "n_relevant_qrels": len(relevant),
            "first_relevant_rank": first_rank,
            "qrels_available": True,
        }

    return {
        "context_relevance": lexical_grounding(query, passages),
        "context_recall": None,
        "context_metric": "query_context_lexical_overlap",
        "n_relevant_in_context": None,
        "n_relevant_qrels": None,
        "first_relevant_rank": None,
        "qrels_available": False,
    }


def _mean(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def _dimension_flags(scores: Mapping[str, float], threshold: float) -> list[str]:
    return [name for name, value in scores.items() if value < threshold]


def score_prediction_row(
    row: Mapping[str, Any],
    *,
    config_name: str,
    qrels: Mapping[str, set[str]] | None = None,
    context_top_k: int | None = None,
    ngram_n: int = 3,
    low_score_threshold: float = DEFAULT_LOW_SCORE_THRESHOLD,
) -> dict[str, Any]:
    """Score one prediction row and return the triad JSON schema."""
    qid = str(row["query_id"])
    query = str(row.get("query") or "")
    prediction = str(row.get("prediction") or "")
    passages = _clip(_as_text_list(row.get("passages")), context_top_k)
    top_doc_ids = _clip(_as_text_list(row.get("top_doc_ids")), context_top_k)
    references = _as_text_list(row.get("references"))

    context = _context_relevance(
        qid=qid,
        query=query,
        top_doc_ids=top_doc_ids,
        passages=passages,
        qrels=qrels,
    )
    lexical = lexical_grounding(prediction, passages)
    ngram = ngram_grounding(prediction, passages, n=ngram_n)
    answer_f1 = token_f1(prediction, references)
    answer_em = exact_match(prediction, references)

    dimension_scores = {
        "context_relevance": float(context["context_relevance"]),
        "groundedness": float(lexical),
        "answer_relevance": float(answer_f1),
    }
    triad_score = _mean(list(dimension_scores.values()))
    low_dimensions = _dimension_flags(dimension_scores, low_score_threshold)

    return {
        "config": config_name,
        "query_id": qid,
        "query": query,
        "query_form": classify_question_form(query),
        "query_type": row.get("query_type"),
        "retrieval_source": row.get("retrieval_source") or config_name,
        "scores": {
            **dimension_scores,
            "triad": triad_score,
            "lexical_grounding": float(lexical),
            "ngram_grounding": float(ngram),
            "exact_match": float(answer_em),
        },
        "context": {
            "metric": context["context_metric"],
            "top_doc_ids": top_doc_ids,
            "passage_snippets": [
                passage.replace("\n", " ")[:240]
                for passage in passages
            ],
            "n_passages": len(passages),
            "qrels_available": bool(context["qrels_available"]),
            "first_relevant_rank": context["first_relevant_rank"],
            "n_relevant_in_context": context["n_relevant_in_context"],
            "n_relevant_qrels": context["n_relevant_qrels"],
            "context_recall": context["context_recall"],
        },
        "generation": {
            "prediction": prediction,
            "references": references,
        },
        "flags": {
            "empty_context": len(passages) == 0 or not any(p.strip() for p in passages),
            "missing_references": not references,
            "vacuous_lexical_grounding": is_vacuously_grounded_lex(prediction),
            "vacuous_ngram_grounding": is_vacuously_grounded_ngram(prediction, n=ngram_n),
            "low_dimensions": low_dimensions,
        },
    }


def _attach_movement(
    scored_rows: list[dict[str, Any]],
    *,
    baseline_config: str | None,
) -> None:
    if baseline_config is None:
        return
    baseline_by_qid = {
        row["query_id"]: row
        for row in scored_rows
        if row["config"] == baseline_config
    }
    for row in scored_rows:
        if row["config"] == baseline_config:
            row["movement"] = {"baseline_config": baseline_config, "bucket": "baseline"}
            continue
        baseline = baseline_by_qid.get(row["query_id"])
        if baseline is None:
            row["movement"] = {"baseline_config": baseline_config, "bucket": "missing_baseline"}
            continue
        if not row["context"]["qrels_available"] or not baseline["context"]["qrels_available"]:
            row["movement"] = {"baseline_config": baseline_config, "bucket": "qrels_unavailable"}
            continue
        before_rank = baseline["context"]["first_relevant_rank"]
        after_rank = row["context"]["first_relevant_rank"]
        row["movement"] = {
            "baseline_config": baseline_config,
            "bucket": retrieval_shift_bucket(before_rank, after_rank),
            "baseline_first_relevant_rank": before_rank,
            "first_relevant_rank": after_rank,
        }


def build_triad_report(
    predictions_by_config: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    qrels: Mapping[str, set[str]] | None = None,
    evaluator: str = "deterministic",
    baseline_config: str | None = None,
    context_top_k: int | None = None,
    ngram_n: int = 3,
    low_score_threshold: float = DEFAULT_LOW_SCORE_THRESHOLD,
    max_low_score_cases: int = 100,
) -> dict[str, Any]:
    evaluator = validate_evaluator(evaluator)
    if not predictions_by_config:
        raise ValueError("at least one prediction config is required")
    if baseline_config is not None and baseline_config not in predictions_by_config:
        raise ValueError(f"baseline_config {baseline_config!r} is not among prediction configs")
    assert_aligned_predictions(predictions_by_config)

    scored_rows: list[dict[str, Any]] = []
    for config_name, rows in predictions_by_config.items():
        for row in rows:
            scored_rows.append(
                score_prediction_row(
                    row,
                    config_name=config_name,
                    qrels=qrels,
                    context_top_k=context_top_k,
                    ngram_n=ngram_n,
                    low_score_threshold=low_score_threshold,
                )
            )
    _attach_movement(scored_rows, baseline_config=baseline_config)

    summaries: dict[str, Any] = {}
    for config_name in predictions_by_config:
        rows = [row for row in scored_rows if row["config"] == config_name]
        movement_counts = Counter(
            row.get("movement", {}).get("bucket")
            for row in rows
            if row.get("movement") and row.get("movement", {}).get("bucket") != "baseline"
        )
        summaries[config_name] = _summarize_config(
            rows,
            movement_counts,
            low_score_threshold=low_score_threshold,
        )

    low_cases = _low_score_cases(
        scored_rows,
        threshold=low_score_threshold,
        max_cases=max_low_score_cases,
    )
    return {
        "summary": {
            "task": "rag_triad_evaluation",
            "evaluator": evaluator,
            "dimensions": {
                "context_relevance": (
                    "qrels hit in the shown context when qrels are provided; "
                    "otherwise lexical query-context overlap"
                ),
                "groundedness": "lexical content-token support in the shown passages",
                "answer_relevance": "Token-F1 against reference answers",
            },
            "settings": {
                "context_top_k": context_top_k,
                "ngram_n": ngram_n,
                "low_score_threshold": low_score_threshold,
                "baseline_config": baseline_config,
                "qrels_available": qrels is not None,
            },
            "configs": summaries,
            "n_rows": len(scored_rows),
            "n_query_ids": len(next(iter(predictions_by_config.values()))),
        },
        "per_query": scored_rows,
        "low_score_cases": low_cases,
    }


def _summarize_config(
    rows: Sequence[Mapping[str, Any]],
    movement_counts: Counter[str | None],
    *,
    low_score_threshold: float,
) -> dict[str, Any]:
    if not rows:
        return {"n_queries": 0, "metrics": {}}
    metric_keys = ("context_relevance", "groundedness", "answer_relevance", "triad")
    metrics = {
        f"mean_{key}": _mean([float(row["scores"][key]) for row in rows])
        for key in metric_keys
    }
    low_by_dimension = Counter(
        dim for row in rows for dim in row["flags"]["low_dimensions"]
    )
    return {
        "n_queries": len(rows),
        "metrics": metrics,
        "diagnostics": {
            "n_empty_context": sum(1 for row in rows if row["flags"]["empty_context"]),
            "n_missing_references": sum(1 for row in rows if row["flags"]["missing_references"]),
            "n_low_triad": sum(1 for row in rows if row["scores"]["triad"] < low_score_threshold),
            "low_dimension_counts": dict(sorted(low_by_dimension.items())),
            "movement_buckets": dict(sorted((k, v) for k, v in movement_counts.items() if k)),
        },
    }


def _low_score_cases(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    max_cases: int,
    snippet_chars: int = 240,
) -> list[dict[str, Any]]:
    candidates = [
        row for row in rows
        if row["flags"]["low_dimensions"] or row["scores"]["triad"] < threshold
    ]
    candidates.sort(key=lambda row: (float(row["scores"]["triad"]), row["config"], row["query_id"]))
    cases: list[dict[str, Any]] = []
    for row in candidates[:max_cases]:
        cases.append(
            {
                "config": row["config"],
                "query_id": row["query_id"],
                "query": row["query"],
                "query_form": row["query_form"],
                "query_type": row.get("query_type"),
                "scores": row["scores"],
                "low_dimensions": row["flags"]["low_dimensions"],
                "movement": row.get("movement"),
                "top_doc_ids": row["context"]["top_doc_ids"],
                "prediction": row["generation"]["prediction"],
                "references": row["generation"]["references"],
                "passage_snippets": list(row["context"]["passage_snippets"]),
            }
        )
    return cases


def render_triad_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# RAG triad evaluation",
        "",
        f"- Evaluator: `{summary['evaluator']}`",
        f"- Query ids: **{summary['n_query_ids']}**",
        f"- Rows: **{summary['n_rows']}**",
        f"- Low-score threshold: **{summary['settings']['low_score_threshold']}**",
        "",
        "| config | n | context relevance | groundedness | answer relevance | triad |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config_name, config in summary["configs"].items():
        metrics = config["metrics"]
        lines.append(
            "| {name} | {n} | {context:.4f} | {grounded:.4f} | {answer:.4f} | {triad:.4f} |".format(
                name=config_name,
                n=config["n_queries"],
                context=metrics.get("mean_context_relevance", 0.0),
                grounded=metrics.get("mean_groundedness", 0.0),
                answer=metrics.get("mean_answer_relevance", 0.0),
                triad=metrics.get("mean_triad", 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "| config | empty context | missing references | low context | low groundedness | low answer |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for config_name, config in summary["configs"].items():
        diag = config["diagnostics"]
        low = diag["low_dimension_counts"]
        lines.append(
            "| {name} | {empty} | {refs} | {context} | {grounded} | {answer} |".format(
                name=config_name,
                empty=diag["n_empty_context"],
                refs=diag["n_missing_references"],
                context=low.get("context_relevance", 0),
                grounded=low.get("groundedness", 0),
                answer=low.get("answer_relevance", 0),
            )
        )
    lines.extend(
        [
            "",
            "The deterministic baseline should be read as a regression monitor. "
            "It is not a replacement for model-assisted faithfulness or human review.",
            "",
        ]
    )
    return "\n".join(lines)


def write_triad_outputs(
    output_dir: Path | str,
    report: Mapping[str, Any],
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    per_query_path = output / "per_query_triad.jsonl"
    cases_path = output / "low_score_cases.jsonl"
    report_path = output / "report.md"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(report["summary"], f, indent=2, sort_keys=True)
        f.write("\n")
    with per_query_path.open("w", encoding="utf-8") as f:
        for row in report["per_query"]:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with cases_path.open("w", encoding="utf-8") as f:
        for row in report["low_score_cases"]:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    report_path.write_text(render_triad_markdown(report), encoding="utf-8")
    return {
        "metrics": metrics_path,
        "per_query": per_query_path,
        "low_score_cases": cases_path,
        "report": report_path,
    }
