"""Query-level first-stage coverage diagnostics for fixed retrieval runs."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from statistics import mean, median
from typing import Any, Mapping, Sequence


FIRST_HIT_BUCKETS = (
    "top_10",
    "ranks_11_100",
    "ranks_101_1000",
    "miss_top_1000",
)
COVERAGE_AT_100_BUCKETS = (
    "none",
    "partial_lt_25pct",
    "partial_25_to_lt_50pct",
    "partial_50_to_lt_100pct",
    "complete",
)
DEFAULT_CUTOFFS = (10, 100, 1000)
REPORT_SCHEMA = "msmarco-genqa.first-stage-coverage-report.v1"


class FirstStageCoverageError(ValueError):
    """Raised when first-stage diagnostic inputs are inconsistent."""


def first_hit_bucket(first_relevant_rank: int | None) -> str:
    """Map a first relevant rank onto fixed reranker-reachability buckets."""
    if first_relevant_rank is None:
        return "miss_top_1000"
    if (
        isinstance(first_relevant_rank, bool)
        or not isinstance(first_relevant_rank, int)
        or first_relevant_rank < 1
    ):
        raise FirstStageCoverageError("first relevant rank must be positive or None")
    if first_relevant_rank <= 10:
        return "top_10"
    if first_relevant_rank <= 100:
        return "ranks_11_100"
    if first_relevant_rank <= 1000:
        return "ranks_101_1000"
    raise FirstStageCoverageError(
        "first relevant rank exceeds the fixed BM25 depth of 1000"
    )


def relevant_coverage_at_100_bucket(value: float) -> str:
    """Bucket a query's Recall@100 without conflating partial and full coverage."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FirstStageCoverageError("coverage must be a number in [0, 1]")
    coverage = float(value)
    if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
        raise FirstStageCoverageError("coverage must be a finite number in [0, 1]")
    if coverage == 0.0:
        return "none"
    if coverage < 0.25:
        return "partial_lt_25pct"
    if coverage < 0.50:
        return "partial_25_to_lt_50pct"
    if coverage < 1.0:
        return "partial_50_to_lt_100pct"
    return "complete"


def _positive_doc_ids(
    judgments: Mapping[str, int],
    *,
    rel_threshold: int,
) -> set[str]:
    if (
        isinstance(rel_threshold, bool)
        or not isinstance(rel_threshold, int)
        or rel_threshold < 1
    ):
        raise FirstStageCoverageError("rel_threshold must be an integer >= 1")
    return {
        doc_id
        for doc_id, relevance in judgments.items()
        if relevance >= rel_threshold
    }


def analyze_first_stage_query(
    qid: str,
    run_rows: Sequence[tuple[str, float]],
    judgments: Mapping[str, int],
    *,
    query_text: str,
    rel_threshold: int = 1,
    cutoffs: tuple[int, int, int] = DEFAULT_CUTOFFS,
) -> dict[str, Any]:
    """Build an auditable coverage record for one query."""
    if cutoffs != DEFAULT_CUTOFFS:
        raise FirstStageCoverageError(
            "NFCorpus first-stage analysis requires cutoffs (10, 100, 1000)"
        )
    relevant = _positive_doc_ids(judgments, rel_threshold=rel_threshold)
    if not relevant:
        raise FirstStageCoverageError(f"{qid}: no judgments with rel >= {rel_threshold}")
    if len(run_rows) > cutoffs[-1]:
        raise FirstStageCoverageError(
            f"{qid}: run depth {len(run_rows)} exceeds fixed depth {cutoffs[-1]}"
        )

    ranked_hits = [
        {
            "doc_id": doc_id,
            "rank": rank,
            "score": float(score),
        }
        for rank, (doc_id, score) in enumerate(run_rows, start=1)
        if doc_id in relevant
    ]
    first_rank = int(ranked_hits[0]["rank"]) if ranked_hits else None
    hit_doc_ids = {str(hit["doc_id"]) for hit in ranked_hits}
    hit_counts = {
        cutoff: sum(1 for hit in ranked_hits if int(hit["rank"]) <= cutoff)
        for cutoff in cutoffs
    }
    recalls = {
        cutoff: hit_counts[cutoff] / len(relevant)
        for cutoff in cutoffs
    }
    missing_after_1000 = sorted(relevant - hit_doc_ids)

    return {
        "qid": qid,
        "query": query_text,
        "n_relevant": len(relevant),
        "first_relevant_rank": first_rank,
        "first_hit_bucket": first_hit_bucket(first_rank),
        "coverage_at_100_bucket": relevant_coverage_at_100_bucket(recalls[100]),
        "hit_count@10": hit_counts[10],
        "hit_count@100": hit_counts[100],
        "hit_count@1000": hit_counts[1000],
        "recall@10": recalls[10],
        "recall@100": recalls[100],
        "recall@1000": recalls[1000],
        "additional_hit_count_101_1000": hit_counts[1000] - hit_counts[100],
        "missing_count@100": len(relevant) - hit_counts[100],
        "missing_count@1000": len(missing_after_1000),
        "relevant_hits_top_100": [
            hit for hit in ranked_hits if int(hit["rank"]) <= 100
        ],
        "relevant_hits_101_1000": [
            hit for hit in ranked_hits if int(hit["rank"]) > 100
        ],
        "missing_relevant_doc_ids_after_1000": missing_after_1000,
    }


def _bucket_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    order: Sequence[str],
) -> list[dict[str, int | float | str]]:
    counts = Counter(str(row[key]) for row in rows)
    total = len(rows)
    unknown = sorted(set(counts) - set(order))
    if unknown:
        raise FirstStageCoverageError(
            f"{key}: unexpected buckets: {', '.join(unknown)}"
        )
    return [
        {
            "bucket": bucket,
            "n_queries": counts.get(bucket, 0),
            "share": counts.get(bucket, 0) / total if total else 0.0,
        }
        for bucket in order
    ]


def analyze_first_stage_coverage(
    run: Mapping[str, Sequence[tuple[str, float]]],
    qrels: Mapping[str, Mapping[str, int]],
    queries: Mapping[str, str],
    *,
    rel_threshold: int = 1,
) -> dict[str, Any]:
    """Analyze every qrels query and exactly reconcile the coverage decomposition."""
    qids = set(qrels)
    missing_run_qids = sorted(qids - set(run))
    extra_run_qids = sorted(set(run) - qids)
    missing_query_texts = sorted(qids - set(queries))
    if missing_run_qids or extra_run_qids:
        raise FirstStageCoverageError(
            "run/qrels qid mismatch: "
            f"missing={missing_run_qids[:5]}, extra={extra_run_qids[:5]}"
        )
    if missing_query_texts:
        raise FirstStageCoverageError(
            "query texts missing for qids: " + ", ".join(missing_query_texts[:5])
        )

    rows = [
        analyze_first_stage_query(
            qid,
            run[qid],
            qrels[qid],
            query_text=queries[qid],
            rel_threshold=rel_threshold,
        )
        for qid in sorted(qids)
    ]
    if not rows:
        raise FirstStageCoverageError("analysis requires at least one query")

    total_relevant = sum(int(row["n_relevant"]) for row in rows)
    macro_recall = {
        cutoff: mean(float(row[f"recall@{cutoff}"]) for row in rows)
        for cutoff in DEFAULT_CUTOFFS
    }
    total_hits = {
        cutoff: sum(int(row[f"hit_count@{cutoff}"]) for row in rows)
        for cutoff in DEFAULT_CUTOFFS
    }
    micro_recall = {
        cutoff: total_hits[cutoff] / total_relevant
        for cutoff in DEFAULT_CUTOFFS
    }
    first_hit_summary = _bucket_summary(
        rows,
        key="first_hit_bucket",
        order=FIRST_HIT_BUCKETS,
    )
    coverage_summary = _bucket_summary(
        rows,
        key="coverage_at_100_bucket",
        order=COVERAGE_AT_100_BUCKETS,
    )
    first_hit_counts = {
        str(row["bucket"]): int(row["n_queries"]) for row in first_hit_summary
    }
    coverage_counts = {
        str(row["bucket"]): int(row["n_queries"]) for row in coverage_summary
    }
    n_queries = len(rows)
    if sum(first_hit_counts.values()) != n_queries:
        raise FirstStageCoverageError("first-hit buckets do not reconcile")
    if sum(coverage_counts.values()) != n_queries:
        raise FirstStageCoverageError("coverage@100 buckets do not reconcile")

    n_relevant_values = [int(row["n_relevant"]) for row in rows]
    qids_gaining_after_100 = sum(
        int(row["additional_hit_count_101_1000"]) > 0 for row in rows
    )
    report = {
        "schema": REPORT_SCHEMA,
        "settings": {
            "relevance_rule": f"relevance >= {rel_threshold}",
            "rel_threshold": rel_threshold,
            "cutoffs": list(DEFAULT_CUTOFFS),
            "reranker_candidate_depth": 100,
            "headline_aggregation": "macro average over qrels queries",
        },
        "scope": {
            "n_queries": n_queries,
            "n_positive_qrels": total_relevant,
            "run_depth": max(len(run[qid]) for qid in qids),
            "relevant_per_query": {
                "minimum": min(n_relevant_values),
                "median": float(median(n_relevant_values)),
                "mean": mean(n_relevant_values),
                "maximum": max(n_relevant_values),
            },
        },
        "macro_recall": {
            f"recall@{cutoff}": macro_recall[cutoff]
            for cutoff in DEFAULT_CUTOFFS
        },
        "micro_qrels_coverage": {
            f"recall@{cutoff}": micro_recall[cutoff]
            for cutoff in DEFAULT_CUTOFFS
        },
        "first_hit_buckets": first_hit_summary,
        "coverage_at_100_buckets": coverage_summary,
        "candidate_set_diagnostic": {
            "queries_with_any_relevant_in_top_100": (
                n_queries - coverage_counts["none"]
            ),
            "queries_with_no_relevant_in_top_100": coverage_counts["none"],
            "queries_with_complete_relevant_coverage_at_100": coverage_counts[
                "complete"
            ],
            "queries_with_partial_relevant_coverage_at_100": (
                n_queries - coverage_counts["none"] - coverage_counts["complete"]
            ),
            "positive_qrels_in_top_100": total_hits[100],
            "positive_qrels_outside_top_100": total_relevant - total_hits[100],
        },
        "depth_100_to_1000_diagnostic": {
            "macro_recall_gain": macro_recall[1000] - macro_recall[100],
            "micro_qrels_coverage_gain": micro_recall[1000] - micro_recall[100],
            "additional_positive_qrels_found": total_hits[1000] - total_hits[100],
            "queries_gaining_relevant_documents": qids_gaining_after_100,
            "queries_with_first_hit_only_at_ranks_101_1000": first_hit_counts[
                "ranks_101_1000"
            ],
            "queries_still_missing_at_depth_1000": first_hit_counts[
                "miss_top_1000"
            ],
            "positive_qrels_still_missing_at_depth_1000": (
                total_relevant - total_hits[1000]
            ),
        },
        "reconciliation": {
            "first_hit_bucket_queries": sum(first_hit_counts.values()),
            "coverage_at_100_bucket_queries": sum(coverage_counts.values()),
            "positive_qrels": total_relevant,
            "positive_qrels_top_100_plus_outside": (
                total_hits[100] + total_relevant - total_hits[100]
            ),
        },
        "per_query": rows,
    }
    return report


def deterministic_bucket_samples(
    rows: Sequence[Mapping[str, Any]],
    *,
    dimensions: Sequence[str] = (
        "first_hit_bucket",
        "coverage_at_100_bucket",
    ),
    per_bucket: int = 3,
    seed: str = "nfcorpus-first-stage-errors-v1",
) -> list[dict[str, Any]]:
    """Select stable, non-cherry-picked examples within every observed bucket."""
    if per_bucket < 1:
        raise FirstStageCoverageError("per_bucket must be at least 1")
    samples: list[dict[str, Any]] = []
    for dimension in dimensions:
        buckets = sorted({str(row[dimension]) for row in rows})
        for bucket in buckets:
            candidates: list[tuple[str, Mapping[str, Any]]] = []
            for row in rows:
                if str(row[dimension]) != bucket:
                    continue
                qid = str(row["qid"])
                digest = hashlib.sha256(
                    f"{seed}\0{dimension}\0{bucket}\0{qid}".encode()
                ).hexdigest()
                candidates.append((digest, row))
            for selection_rank, (digest, row) in enumerate(
                sorted(candidates, key=lambda item: (item[0], str(item[1]["qid"])))[:per_bucket],
                start=1,
            ):
                samples.append(
                    {
                        "sample_dimension": dimension,
                        "sample_bucket": bucket,
                        "selection_rank": selection_rank,
                        "selection_sha256": digest,
                        **dict(row),
                    }
                )
    return samples


def first_stage_diagnostic_fingerprint(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact aggregate counts pinned after the first verified run."""
    first_hit_counts = {
        str(row["bucket"]): int(row["n_queries"])
        for row in report["first_hit_buckets"]
    }
    coverage_counts = {
        str(row["bucket"]): int(row["n_queries"])
        for row in report["coverage_at_100_buckets"]
    }
    candidate = report["candidate_set_diagnostic"]
    depth = report["depth_100_to_1000_diagnostic"]
    positive_top_100 = int(candidate["positive_qrels_in_top_100"])
    additional = int(depth["additional_positive_qrels_found"])
    return {
        "n_queries": int(report["scope"]["n_queries"]),
        "n_positive_qrels": int(report["scope"]["n_positive_qrels"]),
        "first_hit_bucket_counts": first_hit_counts,
        "coverage_at_100_bucket_counts": coverage_counts,
        "positive_qrels_in_top_100": positive_top_100,
        "positive_qrels_in_top_1000": positive_top_100 + additional,
        "queries_gaining_relevant_documents_101_1000": int(
            depth["queries_gaining_relevant_documents"]
        ),
        "positive_qrels_missing_at_1000": int(
            depth["positive_qrels_still_missing_at_depth_1000"]
        ),
    }


def assert_first_stage_diagnostic_fingerprint(
    report: Mapping[str, Any],
    expected: Any,
) -> None:
    """Reject aggregate diagnostic drift against an explicit JSON contract."""
    if not isinstance(expected, dict):
        raise FirstStageCoverageError(
            "expected_first_stage_diagnostics must be an object"
        )
    observed = first_stage_diagnostic_fingerprint(report)
    if observed != expected:
        differing = sorted(
            key
            for key in observed.keys() | expected.keys()
            if observed.get(key) != expected.get(key)
        )
        raise FirstStageCoverageError(
            "first-stage diagnostic drift in: " + ", ".join(differing)
        )


def render_first_stage_coverage_markdown(
    report: Mapping[str, Any],
    *,
    dataset_id: str,
    contract_path: str,
    release_tag: str,
    samples: Sequence[Mapping[str, Any]],
) -> str:
    """Render an answer-first diagnostic report from verified query-level evidence."""
    scope = report["scope"]
    candidate = report["candidate_set_diagnostic"]
    depth = report["depth_100_to_1000_diagnostic"]
    settings = report["settings"]
    lines = [
        "# NFCorpus First-Stage Coverage Analysis",
        "",
        "## Result",
        "",
        (
            f"BM25 retrieves at least one relevant document inside the fixed top-100 "
            f"candidate set for **{candidate['queries_with_any_relevant_in_top_100']}/"
            f"{scope['n_queries']}** queries; "
            f"**{candidate['queries_with_no_relevant_in_top_100']}/"
            f"{scope['n_queries']}** queries give the reranker no relevant candidate."
        ),
        (
            f"Extending the same BM25 run from depth 100 to 1000 raises macro Recall "
            f"from **{report['macro_recall']['recall@100']:.4f}** to "
            f"**{report['macro_recall']['recall@1000']:.4f}**, but "
            f"**{depth['positive_qrels_still_missing_at_depth_1000']}** positive qrels "
            "remain unretrieved."
        ),
        "",
        "This diagnoses the fixed first-stage candidate set; it does not evaluate a "
        "new retriever or justify an architectural change.",
        "",
        "## Frozen Scope",
        "",
        f"- Dataset: `{dataset_id}`",
        f"- Data/metric contract: `{contract_path}`",
        f"- Published evidence release: `{release_tag}`",
        f"- Relevance rule: `{settings['relevance_rule']}`",
        f"- Queries: **{scope['n_queries']}**",
        f"- Positive qrels: **{scope['n_positive_qrels']}**",
        f"- BM25 depth: **{scope['run_depth']}**",
        "",
        "## Bucket Definitions",
        "",
        "| first-hit bucket | definition | reranker interpretation |",
        "|---|---|---|",
        "| `top_10` | first relevant document at rank 1-10 | already visible at the headline rank cutoff |",
        "| `ranks_11_100` | first relevant document at rank 11-100 | reachable by the fixed top-100 reranker |",
        "| `ranks_101_1000` | first relevant document at rank 101-1000 | excluded from the reranker candidate set |",
        "| `miss_top_1000` | no relevant document in BM25 top 1000 | not recovered by deeper BM25 retrieval |",
        "",
        "| coverage@100 bucket | definition |",
        "|---|---|",
        "| `none` | Recall@100 = 0 |",
        "| `partial_lt_25pct` | 0 < Recall@100 < 0.25 |",
        "| `partial_25_to_lt_50pct` | 0.25 <= Recall@100 < 0.50 |",
        "| `partial_50_to_lt_100pct` | 0.50 <= Recall@100 < 1 |",
        "| `complete` | Recall@100 = 1 |",
        "",
        "## Aggregate Coverage",
        "",
        "| aggregation | Recall@10 | Recall@100 | Recall@1000 |",
        "|---|---:|---:|---:|",
        (
            f"| Macro (headline definition) | "
            f"{report['macro_recall']['recall@10']:.4f} | "
            f"{report['macro_recall']['recall@100']:.4f} | "
            f"{report['macro_recall']['recall@1000']:.4f} |"
        ),
        (
            f"| Micro qrels coverage (diagnostic only) | "
            f"{report['micro_qrels_coverage']['recall@10']:.4f} | "
            f"{report['micro_qrels_coverage']['recall@100']:.4f} | "
            f"{report['micro_qrels_coverage']['recall@1000']:.4f} |"
        ),
        "",
        "Macro recall gives every query equal weight and remains the reported metric. "
        "Micro coverage weights queries by their number of positive qrels and is shown "
        "only to explain where relevant-document mass is lost.",
        "",
        "## First Relevant Hit",
        "",
        "| bucket | queries | share |",
        "|---|---:|---:|",
    ]
    for row in report["first_hit_buckets"]:
        lines.append(
            f"| `{row['bucket']}` | {row['n_queries']} | {row['share']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Relevant-Document Coverage at 100",
            "",
            "| bucket | queries | share |",
            "|---|---:|---:|",
        ]
    )
    for row in report["coverage_at_100_buckets"]:
        lines.append(
            f"| `{row['bucket']}` | {row['n_queries']} | {row['share']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Depth 100 to 1000",
            "",
            "| diagnostic | value |",
            "|---|---:|",
            f"| Macro Recall gain | {depth['macro_recall_gain']:+.4f} |",
            f"| Additional positive qrels found | {depth['additional_positive_qrels_found']} |",
            f"| Queries gaining at least one relevant document | {depth['queries_gaining_relevant_documents']} |",
            f"| Queries whose first hit is only at rank 101-1000 | {depth['queries_with_first_hit_only_at_ranks_101_1000']} |",
            f"| Queries still missing at depth 1000 | {depth['queries_still_missing_at_depth_1000']} |",
            f"| Positive qrels still missing at depth 1000 | {depth['positive_qrels_still_missing_at_depth_1000']} |",
            "",
            "## Deterministic Examples",
            "",
            "Examples are selected by SHA-256 over the fixed seed, dimension, bucket, "
            "and qid. They are not manually chosen.",
            "",
            "| dimension | bucket | qid | query | first rank | Recall@100 | Recall@1000 |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in samples:
        query = (
            str(row["query"])
            .replace("|", "\\|")
            .replace("\r", " ")
            .replace("\n", " ")
        )
        if len(query) > 100:
            query = query[:97] + "..."
        first_rank = row["first_relevant_rank"]
        lines.append(
            f"| `{row['sample_dimension']}` | `{row['sample_bucket']}` | "
            f"`{row['qid']}` | {query} | "
            f"{first_rank if first_rank is not None else '-'} | "
            f"{float(row['recall@100']):.3f} | "
            f"{float(row['recall@1000']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "The analysis uses one frozen BM25 output and public graded qrels. It shows "
            "candidate-set reachability and missing relevant-document coverage, not why "
            "BM25 missed a document semantically. A later qualitative review may inspect "
            "lexical mismatch, terminology, query specificity, and document length, but "
            "those explanations are not asserted here.",
            "",
        ]
    )
    return "\n".join(lines)
