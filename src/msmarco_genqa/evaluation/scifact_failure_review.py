"""SciFact residual first-stage failure review utilities."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from msmarco_genqa.evaluation.first_stage_review import surface_tokens


REVIEW_SCHEMA = "msmarco-genqa.scifact-failure-review.v1"
SUMMARY_SCHEMA = "msmarco-genqa.scifact-failure-review-summary.v1"
TARGET_BUCKETS = {
    "ranks_101_1000": "depth_recoverable_101_1000",
    "miss_top_1000": "miss_top_1000",
}
DIRECTIONAL_TOKENS = frozenset(
    {
        "activated",
        "activation",
        "decreases",
        "different",
        "increases",
        "inhibited",
        "inhibits",
        "not",
        "precipitates",
        "reduces",
    }
)
PRIMARY_LABELS = {
    "terminology_or_evidence_form_mismatch": (
        "The judged evidence uses a substantially different surface formulation "
        "from the claim, so exact lexical matching gives it weak BM25 support."
    ),
    "lexical_competition_at_depth_cutoff": (
        "The judged evidence has moderate surface overlap, but higher-ranked "
        "documents match the claim text more strongly and push it beyond the "
        "top-100 reranker cutoff."
    ),
    "short_or_broad_claim": (
        "The claim is short or broad enough that exact terms retrieve many "
        "plausible non-relevant documents before the judged evidence."
    ),
    "other_unclear": (
        "The available query, qrels, and BM25 evidence do not support a more "
        "specific descriptive label."
    ),
}


class SciFactFailureReviewError(ValueError):
    """Raised when SciFact failure-review inputs are inconsistent."""


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
        raise SciFactFailureReviewError(f"corpus is missing document {doc_id!r}")
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
        record["score"] = float(score)
    return record


def _primary_label(flags: Sequence[str]) -> str:
    flag_set = set(flags)
    if "short_or_broad_claim" in flag_set:
        return "short_or_broad_claim"
    if "low_positive_surface_overlap" in flag_set:
        return "terminology_or_evidence_form_mismatch"
    if "top_lexical_competition" in flag_set:
        return "lexical_competition_at_depth_cutoff"
    return "other_unclear"


def build_scifact_failure_cases(
    per_query_rows: Sequence[Mapping[str, Any]],
    run: Mapping[str, Sequence[tuple[str, float]]],
    qrels: Mapping[str, Mapping[str, int]],
    corpus: Mapping[str, Mapping[str, Any]],
    *,
    rel_threshold: int = 1,
    top_documents: int = 3,
    relevant_documents: int = 2,
    snippet_chars: int = 320,
) -> list[dict[str, Any]]:
    """Build a bounded residual-failure review over SciFact no-hit@100 cases."""
    if top_documents < 1 or relevant_documents < 1 or snippet_chars < 80:
        raise SciFactFailureReviewError("review evidence limits are invalid")

    cases: list[dict[str, Any]] = []
    for row in per_query_rows:
        bucket = str(row.get("first_hit_bucket"))
        cohort = TARGET_BUCKETS.get(bucket)
        if cohort is None:
            continue
        qid = str(row.get("qid") or "")
        if not qid:
            raise SciFactFailureReviewError("per-query row is missing qid")
        if qid not in run or qid not in qrels:
            raise SciFactFailureReviewError(f"{qid}: missing run or qrels")
        query = str(row.get("query") or "")
        query_tokens = set(surface_tokens(query))
        positive = {
            doc_id: int(relevance)
            for doc_id, relevance in qrels[qid].items()
            if int(relevance) >= rel_threshold
        }
        if not positive:
            raise SciFactFailureReviewError(f"{qid}: no positive qrels")
        ranked = {
            doc_id: (rank, float(score))
            for rank, (doc_id, score) in enumerate(run[qid], start=1)
        }
        qrel_order = sorted(
            positive,
            key=lambda doc_id: (
                ranked.get(doc_id, (10**9, 0.0))[0],
                -positive[doc_id],
                doc_id,
            ),
        )
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
            for doc_id in qrel_order[:relevant_documents]
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
        max_positive_overlap = max(
            float(document["query_token_recall"]) for document in relevant_evidence
        )
        max_top_overlap = max(
            float(document["query_token_recall"]) for document in top_evidence
        )
        flags: list[str] = []
        if len(query_tokens) <= 4:
            flags.append("short_or_broad_claim")
        if max_positive_overlap <= 0.25:
            flags.append("low_positive_surface_overlap")
        if max_top_overlap > max_positive_overlap:
            flags.append("top_lexical_competition")
        if query_tokens & DIRECTIONAL_TOKENS:
            flags.append("polarity_or_directional_claim")
        if cohort == "depth_recoverable_101_1000":
            flags.append("depth_recoverable_after_reranker_cutoff")
        else:
            flags.append("not_recovered_by_bm25_top_1000")

        cases.append(
            {
                "schema": REVIEW_SCHEMA,
                "qid": qid,
                "cohort": cohort,
                "query": query,
                "query_content_tokens": sorted(query_tokens),
                "n_query_content_tokens": len(query_tokens),
                "n_positive_qrels": len(positive),
                "positive_doc_ids": sorted(positive),
                "first_relevant_rank": row.get("first_relevant_rank"),
                "recall@100": float(row["recall@100"]),
                "recall@1000": float(row["recall@1000"]),
                "max_positive_qrel_query_token_recall": max_positive_overlap,
                "max_top_result_query_token_recall": max_top_overlap,
                "diagnostic_flags": flags,
                "primary_label": _primary_label(flags),
                "representative_relevant_documents": relevant_evidence,
                "top_bm25_documents": top_evidence,
            }
        )

    shared_docs: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        for doc_id in case["positive_doc_ids"]:
            shared_docs[str(doc_id)].append(str(case["qid"]))
    shared_docs = {
        doc_id: sorted(qids) for doc_id, qids in shared_docs.items() if len(qids) > 1
    }
    for case in cases:
        if any(doc_id in shared_docs for doc_id in case["positive_doc_ids"]):
            case["diagnostic_flags"].append("shared_positive_evidence_doc")

    cases.sort(key=lambda item: (str(item["cohort"]), str(item["qid"])))
    for review_order, case in enumerate(cases, start=1):
        case["review_order"] = review_order
    return cases


def summarize_scifact_failure_review(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize the residual SciFact review cases."""
    if not cases:
        raise SciFactFailureReviewError("review requires at least one case")
    cohort_counts = Counter(str(case["cohort"]) for case in cases)
    label_counts = Counter(str(case["primary_label"]) for case in cases)
    flag_counts: Counter[str] = Counter()
    shared_docs: dict[str, list[str]] = defaultdict(list)
    overlap_values = []
    top_overlap_values = []
    first_ranks = []
    for case in cases:
        for flag in case.get("diagnostic_flags", []):
            flag_counts[str(flag)] += 1
        for doc_id in case.get("positive_doc_ids", []):
            shared_docs[str(doc_id)].append(str(case["qid"]))
        overlap_values.append(float(case["max_positive_qrel_query_token_recall"]))
        top_overlap_values.append(float(case["max_top_result_query_token_recall"]))
        rank = case.get("first_relevant_rank")
        if isinstance(rank, int):
            first_ranks.append(rank)
    shared_docs = {
        doc_id: sorted(qids) for doc_id, qids in shared_docs.items() if len(qids) > 1
    }
    return {
        "schema": SUMMARY_SCHEMA,
        "n_cases": len(cases),
        "cohort_counts": dict(sorted(cohort_counts.items())),
        "primary_label_counts": dict(sorted(label_counts.items())),
        "diagnostic_flag_counts": dict(sorted(flag_counts.items())),
        "shared_positive_evidence_docs": dict(sorted(shared_docs.items())),
        "n_shared_positive_evidence_doc_groups": len(shared_docs),
        "n_queries_in_shared_positive_evidence_doc_groups": sum(
            len(qids) for qids in shared_docs.values()
        ),
        "max_positive_qrel_query_token_recall_mean": sum(overlap_values)
        / len(overlap_values),
        "max_top_result_query_token_recall_mean": sum(top_overlap_values)
        / len(top_overlap_values),
        "depth_recoverable_first_rank_min": min(first_ranks) if first_ranks else None,
        "depth_recoverable_first_rank_max": max(first_ranks) if first_ranks else None,
    }


def scifact_failure_fingerprint(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact expected counts for drift checks."""
    return {
        "n_cases": int(summary["n_cases"]),
        "cohort_counts": dict(summary["cohort_counts"]),
        "primary_label_counts": dict(summary["primary_label_counts"]),
        "diagnostic_flag_counts": {
            key: int(summary["diagnostic_flag_counts"].get(key, 0))
            for key in (
                "low_positive_surface_overlap",
                "polarity_or_directional_claim",
                "shared_positive_evidence_doc",
                "short_or_broad_claim",
                "top_lexical_competition",
            )
        },
        "n_shared_positive_evidence_doc_groups": int(
            summary["n_shared_positive_evidence_doc_groups"]
        ),
        "n_queries_in_shared_positive_evidence_doc_groups": int(
            summary["n_queries_in_shared_positive_evidence_doc_groups"]
        ),
    }


def assert_scifact_failure_fingerprint(
    summary: Mapping[str, Any],
    expected: Any,
) -> None:
    """Reject drift against the checked SciFact residual-review contract."""
    if not isinstance(expected, Mapping):
        raise SciFactFailureReviewError("expected fingerprint must be an object")
    observed = scifact_failure_fingerprint(summary)
    if observed != expected:
        differing = sorted(
            key
            for key in set(observed) | set(expected)
            if observed.get(key) != expected.get(key)
        )
        raise SciFactFailureReviewError(
            "SciFact residual review drift in: " + ", ".join(differing)
        )


def render_scifact_failure_review_markdown(
    cases: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> str:
    """Render a concise Markdown report for the SciFact residual review."""
    n_cases = int(summary["n_cases"])
    cohort = summary["cohort_counts"]
    labels = summary["primary_label_counts"]
    flags = summary["diagnostic_flag_counts"]
    lines = [
        "# SciFact Residual First-Stage Failure Review",
        "",
        "## Main Finding",
        "",
        (
            "The 35 SciFact queries with no judged relevant document in BM25 top 100 "
            "do not reproduce the NFCorpus source-context failure pattern. They are "
            "mostly claim/evidence formulation misses: judged positive abstracts often "
            "share few exact content tokens with the claim, while higher-ranked BM25 "
            "documents match the claim wording more directly."
        ),
        "",
        "## Scope",
        "",
        "- Dataset: `beir/scifact/test`.",
        "- Review population: all 35 queries with BM25 Recall@100 equal to 0.",
        "- Cohorts: 24 depth-recoverable cases and 11 top-1000 misses.",
        "- Evidence: frozen BM25 depth-1000 run, public qrels, SciFact claims, and corpus abstracts.",
        "- Boundary: retrieval-only; no retrieval, reranking, or generation is rerun.",
        "",
        "## Residual Failure Categories",
        "",
        "| Primary label | Cases | Share | Interpretation |",
        "|---|---:|---:|---|",
    ]
    for label, definition in PRIMARY_LABELS.items():
        count = int(labels.get(label, 0))
        lines.append(f"| `{label}` | {count} | {count / n_cases:.1%} | {definition} |")
    lines.extend(
        [
            "",
            "## Cohorts",
            "",
            "| Cohort | Cases | Interpretation |",
            "|---|---:|---|",
            (
                "| `depth_recoverable_101_1000` | "
                f"{int(cohort.get('depth_recoverable_101_1000', 0))} | "
                "BM25 eventually retrieves a judged positive document, but below the fixed top-100 reranker cutoff. |"
            ),
            (
                "| `miss_top_1000` | "
                f"{int(cohort.get('miss_top_1000', 0))} | "
                "The judged positive document is absent even after extending BM25 to depth 1000. |"
            ),
            "",
            "## Cross-Cutting Signals",
            "",
            "| Signal | Cases | Interpretation |",
            "|---|---:|---|",
            (
                "| `top_lexical_competition` | "
                f"{int(flags.get('top_lexical_competition', 0))} | "
                "The strongest BM25 candidates match more claim tokens than the judged positive evidence. |"
            ),
            (
                "| `low_positive_surface_overlap` | "
                f"{int(flags.get('low_positive_surface_overlap', 0))} | "
                "The judged positive evidence covers at most 25% of claim content tokens. |"
            ),
            (
                "| `polarity_or_directional_claim` | "
                f"{int(flags.get('polarity_or_directional_claim', 0))} | "
                "The claim contains a negation, direction, activation, inhibition, or comparative cue. |"
            ),
            (
                "| `shared_positive_evidence_doc` | "
                f"{int(flags.get('shared_positive_evidence_doc', 0))} | "
                "Multiple residual claims point to the same judged evidence document, often with paired or directional wording. |"
            ),
            "",
            "Shared judged-positive evidence groups:",
            "",
        ]
    )
    shared = summary.get("shared_positive_evidence_docs", {})
    if isinstance(shared, Mapping) and shared:
        for doc_id, qids in shared.items():
            lines.append(f"- `{doc_id}`: {', '.join(f'`{qid}`' for qid in qids)}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The answer to the residual SciFact question is narrower than a new retrieval architecture. "
                "The cross-encoder can only rerank the fixed top-100 candidate set, and these 35 cases show "
                "that the first-stage candidate boundary still matters. However, SciFact does not show the "
                "same broad candidate-set collapse as NFCorpus: 259/300 SciFact queries already have complete "
                "relevant-document coverage at top 100."
            ),
            "",
            (
                "The strongest observed SciFact pattern is lexical competition under scientific claim formulation. "
                "Many claims are written as compact statements, while the judged evidence appears in abstracts with "
                "different wording, background framing, or polarity/directional structure. Increasing candidate depth "
                "could recover the 24 depth-recoverable cases, but it would change latency and reranking cost. The "
                "11 top-1000 misses are better treated as terminology/formulation or scope limitations before selecting "
                "a new retriever."
            ),
            "",
            "## Decision",
            "",
            (
                "Keep the pipeline frozen for the current report. The next intervention should be predeclared and "
                "retrieval-side only if it targets this failure mode directly, such as candidate-depth sensitivity, "
                "query rewriting for scientific claims, or hybrid lexical/dense retrieval. The current evidence does "
                "not justify changing the reranker or generator."
            ),
            "",
            "## Reproduction",
            "",
            "```bash",
            "make review-scifact-first-stage",
            "```",
            "",
            "The target writes:",
            "",
            "- `outputs/analysis/scifact_first_stage/review/review_cases.jsonl`",
            "- `outputs/analysis/scifact_first_stage/review/review_summary.json`",
            "- `outputs/analysis/scifact_first_stage/review/review.md`",
            "",
            "## Limitations",
            "",
            "- The labels are a bounded descriptive review aided by exact token-overlap features; they are not causal ground truth.",
            "- The review uses public qrels only and does not infer relevance for unjudged documents.",
            "- The review covers residual no-hit@100 cases, not all 300 SciFact test queries.",
            "- It does not evaluate generated answers or groundedness on SciFact.",
            "",
        ]
    )
    return "\n".join(lines)
