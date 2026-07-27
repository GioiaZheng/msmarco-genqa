"""Review-case construction and validation for first-stage retrieval failures."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


TAXONOMY_SCHEMA = "msmarco-genqa.first-stage-review-taxonomy.v1"
REVIEW_CASE_SCHEMA = "msmarco-genqa.first-stage-review-case.v1"
REVIEW_SUMMARY_SCHEMA = "msmarco-genqa.first-stage-review-summary.v1"
TARGET_FIRST_HIT_BUCKETS = {
    "ranks_101_1000": "depth_recoverable_101_1000",
    "miss_top_1000": "miss_top_1000",
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
CONTENT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)


class FirstStageReviewError(ValueError):
    """Raised when review evidence or annotations violate the review contract."""


def load_review_taxonomy(path: Path | str) -> dict[str, Any]:
    """Load and validate the retrieval-review taxonomy."""
    source = Path(path)
    try:
        taxonomy = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FirstStageReviewError(f"{source}: cannot read taxonomy: {exc}") from exc
    if not isinstance(taxonomy, dict) or taxonomy.get("schema") != TAXONOMY_SCHEMA:
        raise FirstStageReviewError(
            f"{source}: expected taxonomy schema {TAXONOMY_SCHEMA!r}"
        )
    labels = taxonomy.get("labels")
    statuses = taxonomy.get("review_statuses")
    selection = taxonomy.get("selection")
    if not isinstance(labels, dict) or not labels:
        raise FirstStageReviewError("taxonomy labels must be a non-empty object")
    if not isinstance(statuses, list) or set(statuses) != {
        "pending",
        "reviewed",
        "needs_adjudication",
    }:
        raise FirstStageReviewError("taxonomy review_statuses are invalid")
    if not isinstance(selection, dict) or not isinstance(
        selection.get("cohorts"), dict
    ):
        raise FirstStageReviewError("taxonomy selection.cohorts is missing")
    for label, record in labels.items():
        if not isinstance(label, str) or not label:
            raise FirstStageReviewError("taxonomy contains an invalid label")
        if not isinstance(record, dict) or not all(
            isinstance(record.get(key), str) and record[key].strip()
            for key in ("definition", "experiment_implication")
        ):
            raise FirstStageReviewError(
                f"taxonomy label {label!r} lacks definition/implication"
            )
    return taxonomy


def surface_tokens(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Return stable, interpretable surface tokens for review hints."""
    tokens = TOKEN_PATTERN.findall(text.casefold())
    if drop_stopwords:
        tokens = [token for token in tokens if token not in CONTENT_STOPWORDS]
    return tokens


def _document_text(document: Mapping[str, Any]) -> str:
    return "\n".join(
        value.strip()
        for key in ("title", "text")
        if isinstance((value := document.get(key)), str) and value.strip()
    )


def _snippet(text: str, *, max_chars: int) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."


def _overlap_record(query_tokens: set[str], text: str) -> dict[str, Any]:
    document_tokens = set(surface_tokens(text))
    shared = sorted(query_tokens & document_tokens)
    return {
        "shared_query_tokens": shared,
        "query_token_recall": len(shared) / len(query_tokens) if query_tokens else 0.0,
        "document_content_token_count": len(surface_tokens(text)),
    }


def _review_digest(seed: str, cohort: str, qid: str) -> str:
    return hashlib.sha256(f"{seed}\0{cohort}\0{qid}".encode()).hexdigest()


def _query_source_evidence(
    qid: str,
    query_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    record = query_records.get(qid)
    if record is None:
        raise FirstStageReviewError(f"query source is missing record {qid!r}")
    metadata = record.get("metadata")
    url = metadata.get("url") if isinstance(metadata, Mapping) else None
    if not isinstance(url, str) or not url.strip():
        raise FirstStageReviewError(f"{qid}: query source URL is missing")
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise FirstStageReviewError(f"{qid}: query source URL is invalid")
    path_parts = [part for part in parsed.path.split("/") if part]
    first_path_part = path_parts[0] if path_parts else ""
    source_type = {
        "topics": "topic",
        "video": "video",
        "questions": "question",
    }.get(first_path_part)
    if source_type is None:
        source_type = (
            "dated_article"
            if re.fullmatch(r"20\d{2}", first_path_part)
            else "other"
        )
    return {
        "query_source_url": url.strip(),
        "query_source_type": source_type,
    }


def summarize_query_source_diagnostics(
    per_query_rows: Sequence[Mapping[str, Any]],
    query_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize first-stage coverage by the query's source-page type."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in per_query_rows:
        qid = str(row["qid"])
        source_type = _query_source_evidence(qid, query_records)[
            "query_source_type"
        ]
        recall_100 = float(row["recall@100"])
        recall_1000 = float(row["recall@1000"])
        if not math.isfinite(recall_100) or not math.isfinite(recall_1000):
            raise FirstStageReviewError(f"{qid}: recall must be finite")
        groups.setdefault(source_type, []).append(row)

    diagnostics: dict[str, Any] = {}
    for source_type in sorted(groups):
        rows = groups[source_type]
        n_queries = len(rows)
        no_hit_100 = sum(float(row["recall@100"]) == 0.0 for row in rows)
        depth_recoverable = sum(
            str(row["first_hit_bucket"]) == "ranks_101_1000" for row in rows
        )
        miss_1000 = sum(
            str(row["first_hit_bucket"]) == "miss_top_1000" for row in rows
        )
        diagnostics[source_type] = {
            "n_queries": n_queries,
            "no_relevant_top_100": no_hit_100,
            "no_relevant_top_100_rate": no_hit_100 / n_queries,
            "depth_recoverable_101_1000": depth_recoverable,
            "miss_top_1000": miss_1000,
            "macro_recall@100": sum(
                float(row["recall@100"]) for row in rows
            )
            / n_queries,
            "macro_recall@1000": sum(
                float(row["recall@1000"]) for row in rows
            )
            / n_queries,
        }
    return diagnostics


def partition_query_ids_by_source(
    query_ids: Sequence[str],
    query_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Partition query ids by normalized NutritionFacts source-page type."""
    groups: dict[str, list[str]] = {}
    seen: set[str] = set()
    for raw_qid in query_ids:
        qid = str(raw_qid)
        if qid in seen:
            raise FirstStageReviewError(f"duplicate query id {qid!r}")
        seen.add(qid)
        source_type = _query_source_evidence(qid, query_records)[
            "query_source_type"
        ]
        groups.setdefault(source_type, []).append(qid)
    return {
        source_type: sorted(qids)
        for source_type, qids in sorted(groups.items())
    }


def _evidence_document(
    doc_id: str,
    *,
    corpus: Mapping[str, Mapping[str, Any]],
    query_tokens: set[str],
    max_chars: int,
    relevance: int | None = None,
    rank: int | None = None,
    score: float | None = None,
) -> dict[str, Any]:
    document = corpus.get(doc_id)
    if document is None:
        raise FirstStageReviewError(f"corpus is missing document {doc_id!r}")
    text = _document_text(document)
    record: dict[str, Any] = {
        "doc_id": doc_id,
        "title": str(document.get("title") or ""),
        "snippet": _snippet(text, max_chars=max_chars),
        **_overlap_record(query_tokens, text),
    }
    if relevance is not None:
        record["relevance"] = int(relevance)
    if rank is not None:
        record["rank"] = int(rank)
    if score is not None:
        if not math.isfinite(float(score)):
            raise FirstStageReviewError(f"{doc_id}: score must be finite")
        record["score"] = float(score)
    return record


def build_first_stage_review_cases(
    per_query_rows: Sequence[Mapping[str, Any]],
    run: Mapping[str, Sequence[tuple[str, float]]],
    qrels: Mapping[str, Mapping[str, int]],
    corpus: Mapping[str, Mapping[str, Any]],
    taxonomy: Mapping[str, Any],
    *,
    query_records: Mapping[str, Mapping[str, Any]],
    rel_threshold: int = 1,
    top_documents: int = 3,
    relevant_documents: int = 3,
    snippet_chars: int = 360,
) -> list[dict[str, Any]]:
    """Build the complete, deterministically ordered 24+48 review census."""
    if top_documents < 1 or relevant_documents < 1 or snippet_chars < 80:
        raise FirstStageReviewError("review evidence limits are invalid")
    selection = taxonomy["selection"]
    seed = str(selection["seed"])
    expected_cohorts = {
        str(key): int(value) for key, value in selection["cohorts"].items()
    }
    cases: list[dict[str, Any]] = []
    for source in per_query_rows:
        first_hit_bucket = str(source["first_hit_bucket"])
        cohort = TARGET_FIRST_HIT_BUCKETS.get(first_hit_bucket)
        if cohort is None:
            continue
        qid = str(source["qid"])
        if qid not in run or qid not in qrels:
            raise FirstStageReviewError(f"{qid}: missing run or qrels")
        query_source = _query_source_evidence(qid, query_records)
        query = str(source["query"])
        query_tokens = set(surface_tokens(query))
        positive = {
            doc_id: int(relevance)
            for doc_id, relevance in qrels[qid].items()
            if int(relevance) >= rel_threshold
        }
        if not positive:
            raise FirstStageReviewError(f"{qid}: no positive qrels")
        ranked = {
            doc_id: (rank, float(score))
            for rank, (doc_id, score) in enumerate(run[qid], start=1)
        }
        retrieved_relevant = sorted(
            (
                (ranked[doc_id][0], doc_id, relevance, ranked[doc_id][1])
                for doc_id, relevance in positive.items()
                if doc_id in ranked
            ),
            key=lambda item: (item[0], item[1]),
        )
        qrel_order = sorted(
            positive,
            key=lambda doc_id: (
                -positive[doc_id],
                ranked.get(doc_id, (10**9, 0.0))[0],
                doc_id,
            ),
        )
        representative_ids: list[str] = []
        if retrieved_relevant:
            representative_ids.append(retrieved_relevant[0][1])
        representative_ids.extend(
            doc_id for doc_id in qrel_order if doc_id not in representative_ids
        )
        representative_ids = representative_ids[:relevant_documents]
        relevant_evidence = [
            _evidence_document(
                doc_id,
                corpus=corpus,
                query_tokens=query_tokens,
                max_chars=snippet_chars,
                relevance=positive[doc_id],
                rank=ranked.get(doc_id, (None, None))[0],
                score=ranked.get(doc_id, (None, None))[1],
            )
            for doc_id in representative_ids
        ]
        top_evidence = [
            _evidence_document(
                doc_id,
                corpus=corpus,
                query_tokens=query_tokens,
                max_chars=snippet_chars,
                rank=rank,
                score=float(score),
            )
            for rank, (doc_id, score) in enumerate(
                run[qid][:top_documents], start=1
            )
        ]
        all_relevant_overlaps = [
            _overlap_record(query_tokens, _document_text(corpus[doc_id]))[
                "query_token_recall"
            ]
            for doc_id in positive
        ]
        top_overlaps = [
            float(document["query_token_recall"]) for document in top_evidence
        ]
        rank_100_score = float(run[qid][99][1]) if len(run[qid]) >= 100 else None
        first_rank = source.get("first_relevant_rank")
        first_score = (
            float(run[qid][int(first_rank) - 1][1])
            if isinstance(first_rank, int)
            else None
        )
        flags: list[str] = []
        if len(query_tokens) <= 1:
            flags.append("single_content_token_query")
        if max(all_relevant_overlaps, default=0.0) == 0.0:
            flags.append("no_exact_content_token_in_positive_qrels")
        if max(top_overlaps, default=0.0) >= max(
            all_relevant_overlaps, default=0.0
        ):
            flags.append("top_results_match_query_as_well_as_positive_qrels")
        if first_score == 0.0:
            flags.append("first_relevant_score_is_zero")
        if len(positive) >= 20:
            flags.append("at_least_20_positive_qrels")
        relevance_counts = Counter(positive.values())
        if set(relevance_counts) == {1}:
            flags.append("only_relevance_level_1_qrels")

        cases.append(
            {
                "schema": REVIEW_CASE_SCHEMA,
                "qid": qid,
                "cohort": cohort,
                "selection_sha256": _review_digest(seed, cohort, qid),
                "query": query,
                **query_source,
                "query_content_tokens": sorted(query_tokens),
                "n_positive_qrels": len(positive),
                "positive_qrels_by_relevance": {
                    str(level): relevance_counts[level]
                    for level in sorted(relevance_counts)
                },
                "first_relevant_rank": first_rank,
                "recall@100": float(source["recall@100"]),
                "recall@1000": float(source["recall@1000"]),
                "rank_100_score": rank_100_score,
                "first_relevant_score": first_score,
                "max_positive_qrel_query_token_recall": max(
                    all_relevant_overlaps, default=0.0
                ),
                "max_top_result_query_token_recall": max(
                    top_overlaps, default=0.0
                ),
                "diagnostic_flags": flags,
                "representative_relevant_documents": relevant_evidence,
                "top_bm25_documents": top_evidence,
                "review_status": "pending",
                "primary_label": "",
                "secondary_label": "",
                "evidence_note": "",
            }
        )

    cases.sort(key=lambda row: (str(row["cohort"]), str(row["selection_sha256"])))
    counts = Counter(str(row["cohort"]) for row in cases)
    if dict(counts) != expected_cohorts:
        raise FirstStageReviewError(
            f"review cohort drift; expected {expected_cohorts}, got {dict(counts)}"
        )
    for review_order, row in enumerate(cases, start=1):
        row["review_order"] = review_order
    return cases


def validate_review_annotations(
    annotations: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    taxonomy: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate annotation coverage and summarize only completed reviews."""
    labels = set(taxonomy["labels"])
    statuses = set(taxonomy["review_statuses"])
    min_note_chars = int(taxonomy.get("evidence_note_min_chars", 12))
    expected = {str(row["qid"]): str(row["cohort"]) for row in cases}
    observed: dict[str, Mapping[str, Any]] = {}
    errors: list[str] = []
    for index, annotation in enumerate(annotations, start=1):
        qid = str(annotation.get("qid") or "")
        if not qid:
            errors.append(f"row {index}: qid is missing")
            continue
        if qid in observed:
            errors.append(f"row {index}: duplicate qid {qid}")
            continue
        observed[qid] = annotation
        cohort = str(annotation.get("cohort") or "")
        status = str(annotation.get("review_status") or "")
        primary = str(annotation.get("primary_label") or "").strip()
        secondary = str(annotation.get("secondary_label") or "").strip()
        note = str(annotation.get("evidence_note") or "").strip()
        if expected.get(qid) != cohort:
            errors.append(f"{qid}: cohort does not match the frozen case")
        if status not in statuses:
            errors.append(f"{qid}: invalid review_status {status!r}")
        if primary and primary not in labels:
            errors.append(f"{qid}: invalid primary_label {primary!r}")
        if secondary and secondary not in labels:
            errors.append(f"{qid}: invalid secondary_label {secondary!r}")
        if secondary and secondary == primary:
            errors.append(f"{qid}: secondary_label duplicates primary_label")
        if status == "pending" and (primary or secondary or note):
            errors.append(f"{qid}: pending rows must not contain review judgments")
        if status == "reviewed":
            if not primary:
                errors.append(f"{qid}: reviewed row requires primary_label")
            if len(note) < min_note_chars:
                errors.append(
                    f"{qid}: reviewed evidence_note must contain at least "
                    f"{min_note_chars} characters"
                )
        if status == "needs_adjudication":
            if primary:
                errors.append(
                    f"{qid}: needs_adjudication must not assert a primary_label"
                )
            if len(note) < min_note_chars:
                errors.append(
                    f"{qid}: adjudication note must contain at least "
                    f"{min_note_chars} characters"
                )
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing:
        errors.append(f"annotations are missing {len(missing)} qids: {missing[:5]}")
    if extra:
        errors.append(f"annotations contain {len(extra)} unknown qids: {extra[:5]}")
    if errors:
        raise FirstStageReviewError("; ".join(errors))

    status_counts = Counter(
        str(annotation["review_status"]) for annotation in annotations
    )
    reviewed = [
        annotation
        for annotation in annotations
        if annotation["review_status"] == "reviewed"
    ]
    label_counts = Counter(str(annotation["primary_label"]) for annotation in reviewed)
    qrel_evidence_counts = Counter(
        (
            "only_relevance_level_1"
            if set(row["positive_qrels_by_relevance"]) == {"1"}
            else "has_relevance_level_2_or_higher"
        )
        for row in cases
    )
    cohort_summaries: dict[str, dict[str, Any]] = {}
    for cohort in sorted(set(expected.values())):
        cohort_rows = [
            annotation
            for annotation in annotations
            if annotation["cohort"] == cohort
        ]
        cohort_reviewed = [
            annotation
            for annotation in cohort_rows
            if annotation["review_status"] == "reviewed"
        ]
        cohort_labels = Counter(
            str(annotation["primary_label"]) for annotation in cohort_reviewed
        )
        cohort_summaries[cohort] = {
            "n_cases": len(cohort_rows),
            "n_reviewed": len(cohort_reviewed),
            "review_coverage": len(cohort_reviewed) / len(cohort_rows),
            "primary_label_counts": dict(sorted(cohort_labels.items())),
        }
    return {
        "schema": REVIEW_SUMMARY_SCHEMA,
        "n_cases": len(cases),
        "n_reviewed": len(reviewed),
        "review_coverage": len(reviewed) / len(cases),
        "status_counts": dict(sorted(status_counts.items())),
        "primary_label_counts": dict(sorted(label_counts.items())),
        "objective_qrel_evidence_counts": dict(
            sorted(qrel_evidence_counts.items())
        ),
        "cohorts": cohort_summaries,
    }
