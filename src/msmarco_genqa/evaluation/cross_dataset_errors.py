"""Cross-dataset first-stage error comparison utilities."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


REPORT_SCHEMA = "msmarco-genqa.cross-dataset-error-analysis.v1"


class CrossDatasetErrorAnalysisError(ValueError):
    """Raised when cross-dataset comparison inputs are inconsistent."""


def _bucket_count(
    rows: Sequence[Mapping[str, Any]],
    bucket: str,
) -> int:
    for row in rows:
        if row.get("bucket") == bucket:
            value = row.get("n_queries")
            if isinstance(value, bool) or not isinstance(value, int):
                raise CrossDatasetErrorAnalysisError(
                    f"{bucket}: n_queries must be an integer"
                )
            return value
    raise CrossDatasetErrorAnalysisError(f"missing bucket {bucket!r}")


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossDatasetErrorAnalysisError(f"{label}: expected object")
    return value


def _require_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CrossDatasetErrorAnalysisError(f"{label}: expected number")
    return float(value)


def _dataset_row(label: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    scope = _require_mapping(summary.get("scope"), label=f"{label}.scope")
    candidate = _require_mapping(
        summary.get("candidate_set_diagnostic"),
        label=f"{label}.candidate_set_diagnostic",
    )
    depth = _require_mapping(
        summary.get("depth_100_to_1000_diagnostic"),
        label=f"{label}.depth_100_to_1000_diagnostic",
    )
    macro = _require_mapping(
        summary.get("macro_recall"),
        label=f"{label}.macro_recall",
    )
    first_hit_buckets = summary.get("first_hit_buckets")
    coverage_buckets = summary.get("coverage_at_100_buckets")
    if not isinstance(first_hit_buckets, Sequence) or isinstance(
        first_hit_buckets, (str, bytes)
    ):
        raise CrossDatasetErrorAnalysisError(
            f"{label}.first_hit_buckets: expected list"
        )
    if not isinstance(coverage_buckets, Sequence) or isinstance(
        coverage_buckets, (str, bytes)
    ):
        raise CrossDatasetErrorAnalysisError(
            f"{label}.coverage_at_100_buckets: expected list"
        )

    n_queries = int(scope["n_queries"])
    n_positive = int(scope["n_positive_qrels"])
    if n_queries <= 0 or n_positive <= 0:
        raise CrossDatasetErrorAnalysisError(
            f"{label}: n_queries and n_positive_qrels must be positive"
        )
    no_top_100 = int(candidate["queries_with_no_relevant_in_top_100"])
    complete_top_100 = int(
        candidate["queries_with_complete_relevant_coverage_at_100"]
    )
    partial_top_100 = int(
        candidate["queries_with_partial_relevant_coverage_at_100"]
    )
    any_top_100 = int(candidate["queries_with_any_relevant_in_top_100"])
    if no_top_100 + any_top_100 != n_queries:
        raise CrossDatasetErrorAnalysisError(
            f"{label}: top-100 candidate counts do not reconcile"
        )
    if no_top_100 + partial_top_100 + complete_top_100 != n_queries:
        raise CrossDatasetErrorAnalysisError(
            f"{label}: coverage buckets do not reconcile"
        )

    first_hit_101_1000 = _bucket_count(first_hit_buckets, "ranks_101_1000")
    miss_top_1000 = _bucket_count(first_hit_buckets, "miss_top_1000")
    if first_hit_101_1000 + miss_top_1000 != no_top_100:
        raise CrossDatasetErrorAnalysisError(
            f"{label}: no-hit top-100 partition does not reconcile"
        )

    return {
        "dataset": label,
        "dataset_id": summary.get("dataset_id", label),
        "n_queries": n_queries,
        "n_positive_qrels": n_positive,
        "positive_qrels_per_query_mean": float(
            _require_mapping(scope.get("relevant_per_query"), label="relevant")[
                "mean"
            ]
        ),
        "positive_qrels_per_query_median": float(
            _require_mapping(scope.get("relevant_per_query"), label="relevant")[
                "median"
            ]
        ),
        "bm25_recall_at_100": _require_number(
            macro.get("recall@100"),
            label=f"{label}.recall@100",
        ),
        "bm25_recall_at_1000": _require_number(
            macro.get("recall@1000"),
            label=f"{label}.recall@1000",
        ),
        "queries_with_no_relevant_top_100": no_top_100,
        "queries_with_no_relevant_top_100_share": no_top_100 / n_queries,
        "queries_with_any_relevant_top_100": any_top_100,
        "queries_with_any_relevant_top_100_share": any_top_100 / n_queries,
        "queries_with_complete_coverage_top_100": complete_top_100,
        "queries_with_complete_coverage_top_100_share": (
            complete_top_100 / n_queries
        ),
        "queries_with_partial_coverage_top_100": partial_top_100,
        "queries_with_partial_coverage_top_100_share": partial_top_100 / n_queries,
        "first_relevant_only_at_101_1000": first_hit_101_1000,
        "first_relevant_only_at_101_1000_share": first_hit_101_1000 / n_queries,
        "queries_still_missing_at_1000": miss_top_1000,
        "queries_still_missing_at_1000_share": miss_top_1000 / n_queries,
        "positive_qrels_missing_at_1000": int(
            depth["positive_qrels_still_missing_at_depth_1000"]
        ),
        "positive_qrels_missing_at_1000_share": (
            int(depth["positive_qrels_still_missing_at_depth_1000"]) / n_positive
        ),
        "additional_positive_qrels_found_101_1000": int(
            depth["additional_positive_qrels_found"]
        ),
        "queries_gaining_relevant_documents_101_1000": int(
            depth["queries_gaining_relevant_documents"]
        ),
    }


def summarize_taxonomy_rows(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Summarize a compact manual-review table."""
    reviewed = [row for row in rows if row.get("review_status") == "reviewed"]
    if len(reviewed) != len(rows):
        raise CrossDatasetErrorAnalysisError(
            "taxonomy review contains non-reviewed rows"
        )
    labels = Counter(str(row.get("primary_label", "")) for row in reviewed)
    if "" in labels:
        raise CrossDatasetErrorAnalysisError(
            "taxonomy review contains missing primary labels"
        )
    cohorts = Counter(str(row.get("cohort", "")) for row in reviewed)
    if "" in cohorts:
        raise CrossDatasetErrorAnalysisError("taxonomy review contains missing cohorts")
    total = len(reviewed)
    return {
        "n_reviewed": total,
        "primary_label_counts": dict(sorted(labels.items())),
        "primary_label_shares": {
            label: count / total for label, count in sorted(labels.items())
        }
        if total
        else {},
        "cohort_counts": dict(sorted(cohorts.items())),
    }


def build_cross_dataset_error_analysis(
    datasets: Mapping[str, Mapping[str, Any]],
    *,
    nfcorpus_taxonomy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare fixed first-stage diagnostics across named datasets."""
    required = {"NFCorpus", "SciFact"}
    missing = sorted(required - set(datasets))
    if missing:
        raise CrossDatasetErrorAnalysisError(
            "missing required datasets: " + ", ".join(missing)
        )

    rows = [_dataset_row(label, datasets[label]) for label in sorted(datasets)]
    row_by_dataset = {str(row["dataset"]): row for row in rows}
    nfcorpus = row_by_dataset["NFCorpus"]
    scifact = row_by_dataset["SciFact"]

    comparison = {
        "recall_at_100_gap_scifact_minus_nfcorpus": (
            scifact["bm25_recall_at_100"] - nfcorpus["bm25_recall_at_100"]
        ),
        "recall_at_1000_gap_scifact_minus_nfcorpus": (
            scifact["bm25_recall_at_1000"] - nfcorpus["bm25_recall_at_1000"]
        ),
        "no_relevant_top_100_share_gap_nfcorpus_minus_scifact": (
            nfcorpus["queries_with_no_relevant_top_100_share"]
            - scifact["queries_with_no_relevant_top_100_share"]
        ),
        "complete_top_100_share_gap_scifact_minus_nfcorpus": (
            scifact["queries_with_complete_coverage_top_100_share"]
            - nfcorpus["queries_with_complete_coverage_top_100_share"]
        ),
        "miss_top_1000_share_gap_nfcorpus_minus_scifact": (
            nfcorpus["queries_still_missing_at_1000_share"]
            - scifact["queries_still_missing_at_1000_share"]
        ),
    }

    interpretation = {
        "candidate_set_absence": (
            "NFCorpus has the stronger top-100 candidate-set bottleneck; "
            "SciFact still has residual misses, but at a smaller rate."
        ),
        "partial_coverage": (
            "NFCorpus is dominated by partial relevant-document coverage, "
            "while most SciFact queries have complete top-100 coverage."
        ),
        "reranker_boundary": (
            "The cross-encoder can improve early ranking only when relevant "
            "documents are already inside the fixed top-100 candidate set."
        ),
        "pipeline_decision": (
            "The evidence supports keeping the pipeline frozen while "
            "separating dataset/query-form effects from retrieval-capacity "
            "changes."
        ),
    }
    if nfcorpus_taxonomy:
        interpretation["qualitative_boundary"] = (
            "Manual taxonomy evidence is currently complete for the NFCorpus "
            "no-hit-at-100 census only; SciFact has a quantitative first-stage "
            "diagnostic but not a matched manual taxonomy census."
        )

    report = {
        "schema": REPORT_SCHEMA,
        "scope": {
            "datasets": ["NFCorpus", "SciFact"],
            "analysis_type": "retrieval-only fixed-output first-stage comparison",
            "reranker_candidate_depth": 100,
            "bm25_depth": 1000,
            "generation_evaluated": False,
            "retrieval_or_reranking_rerun": False,
        },
        "datasets": rows,
        "comparison": comparison,
        "interpretation": interpretation,
    }
    if nfcorpus_taxonomy:
        report["nfcorpus_manual_taxonomy"] = dict(nfcorpus_taxonomy)
    return report


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def render_cross_dataset_error_markdown(
    report: Mapping[str, Any],
    *,
    nfcorpus_doc: str,
    scifact_doc: str,
) -> str:
    """Render the cross-dataset comparison as a compact Markdown report."""
    datasets = {
        str(row["dataset"]): row
        for row in report.get("datasets", [])
        if isinstance(row, Mapping)
    }
    if {"NFCorpus", "SciFact"} - set(datasets):
        raise CrossDatasetErrorAnalysisError(
            "render requires NFCorpus and SciFact dataset rows"
        )
    nf = datasets["NFCorpus"]
    sf = datasets["SciFact"]
    comparison = _require_mapping(
        report.get("comparison"),
        label="comparison",
    )
    taxonomy = report.get("nfcorpus_manual_taxonomy")

    lines = [
        "# Cross-Dataset First-Stage Error Analysis",
        "",
        "## Result",
        "",
        (
            "NFCorpus and SciFact fail in different first-stage regimes under the "
            "same frozen BM25 -> cross-encoder setup. NFCorpus has the stronger "
            "candidate-set ceiling; SciFact has a smaller residual coverage tail."
        ),
        "",
        "| Diagnostic | NFCorpus | SciFact |",
        "|---|---:|---:|",
        (
            f"| Queries | {nf['n_queries']} | {sf['n_queries']} |"
        ),
        (
            f"| Positive qrels | {nf['n_positive_qrels']} | "
            f"{sf['n_positive_qrels']} |"
        ),
        (
            f"| BM25 Recall@100 | {nf['bm25_recall_at_100']:.4f} | "
            f"{sf['bm25_recall_at_100']:.4f} |"
        ),
        (
            f"| BM25 Recall@1000 | {nf['bm25_recall_at_1000']:.4f} | "
            f"{sf['bm25_recall_at_1000']:.4f} |"
        ),
        (
            f"| No relevant document in top 100 | "
            f"{nf['queries_with_no_relevant_top_100']} "
            f"({_pct(nf['queries_with_no_relevant_top_100_share'])}) | "
            f"{sf['queries_with_no_relevant_top_100']} "
            f"({_pct(sf['queries_with_no_relevant_top_100_share'])}) |"
        ),
        (
            f"| Complete relevant coverage at top 100 | "
            f"{nf['queries_with_complete_coverage_top_100']} "
            f"({_pct(nf['queries_with_complete_coverage_top_100_share'])}) | "
            f"{sf['queries_with_complete_coverage_top_100']} "
            f"({_pct(sf['queries_with_complete_coverage_top_100_share'])}) |"
        ),
        (
            f"| First relevant hit only at ranks 101-1000 | "
            f"{nf['first_relevant_only_at_101_1000']} "
            f"({_pct(nf['first_relevant_only_at_101_1000_share'])}) | "
            f"{sf['first_relevant_only_at_101_1000']} "
            f"({_pct(sf['first_relevant_only_at_101_1000_share'])}) |"
        ),
        (
            f"| No relevant hit at depth 1000 | "
            f"{nf['queries_still_missing_at_1000']} "
            f"({_pct(nf['queries_still_missing_at_1000_share'])}) | "
            f"{sf['queries_still_missing_at_1000']} "
            f"({_pct(sf['queries_still_missing_at_1000_share'])}) |"
        ),
        "",
        "## Cross-Dataset Deltas",
        "",
        "| Delta | Value | Interpretation |",
        "|---|---:|---|",
        (
            "| SciFact - NFCorpus Recall@100 | "
            f"{comparison['recall_at_100_gap_scifact_minus_nfcorpus']:+.4f} | "
            "SciFact has a much healthier fixed top-100 candidate set. |"
        ),
        (
            "| SciFact - NFCorpus complete top-100 coverage share | "
            f"{comparison['complete_top_100_share_gap_scifact_minus_nfcorpus']:+.1%} | "
            "Complete relevant-document coverage is common on SciFact and rare on NFCorpus. |"
        ),
        (
            "| NFCorpus - SciFact no-hit-at-100 share | "
            f"{comparison['no_relevant_top_100_share_gap_nfcorpus_minus_scifact']:+.1%} | "
            "The reranker is more often given no relevant candidate on NFCorpus. |"
        ),
        (
            "| NFCorpus - SciFact no-hit-at-1000 share | "
            f"{comparison['miss_top_1000_share_gap_nfcorpus_minus_scifact']:+.1%} | "
            "The unrecovered lexical first-stage tail is larger on NFCorpus. |"
        ),
        "",
        "## Failure Partition",
        "",
        "| Partition | NFCorpus | SciFact | Meaning |",
        "|---|---:|---:|---|",
        (
            f"| Candidate-set absence at 100 | "
            f"{nf['queries_with_no_relevant_top_100']} | "
            f"{sf['queries_with_no_relevant_top_100']} | "
            "No judged relevant document is available to the fixed top-100 reranker. |"
        ),
        (
            f"| Depth-recoverable absence | "
            f"{nf['first_relevant_only_at_101_1000']} | "
            f"{sf['first_relevant_only_at_101_1000']} | "
            "BM25 can find a relevant document, but only after the reranker cutoff. |"
        ),
        (
            f"| Residual top-1000 miss | "
            f"{nf['queries_still_missing_at_1000']} | "
            f"{sf['queries_still_missing_at_1000']} | "
            "Deeper BM25 still does not retrieve a judged relevant document. |"
        ),
        (
            f"| Partial top-100 coverage | "
            f"{nf['queries_with_partial_coverage_top_100']} | "
            f"{sf['queries_with_partial_coverage_top_100']} | "
            "At least one relevant document is reachable, but some positive qrels remain outside the candidate set. |"
        ),
        (
            f"| Complete top-100 coverage | "
            f"{nf['queries_with_complete_coverage_top_100']} | "
            f"{sf['queries_with_complete_coverage_top_100']} | "
            "All judged positive qrels are already reachable by the reranker. |"
        ),
        "",
        "## Qualitative Boundary",
        "",
    ]
    if isinstance(taxonomy, Mapping):
        counts = _require_mapping(
            taxonomy.get("primary_label_counts"),
            label="nfcorpus_manual_taxonomy.primary_label_counts",
        )
        source_context = int(counts.get("source_context_dependency", 0))
        reviewed = int(taxonomy.get("n_reviewed", 0))
        lines.extend(
            [
                (
                    "The only complete manual taxonomy currently covers the 72 "
                    "NFCorpus queries with no relevant document in BM25 top 100. "
                    f"It labels {source_context}/{reviewed} cases as "
                    "`source_context_dependency`, which supports a compact-query "
                    "representation explanation for that dataset."
                ),
                "",
                "SciFact has the matched quantitative first-stage diagnostic, but "
                "not a separate manual failure taxonomy census. The comparison "
                "therefore supports a retrieval-reachability conclusion, not a "
                "claim that SciFact has the same semantic failure causes.",
            ]
        )
    else:
        lines.append(
            "No manual taxonomy table is attached to this cross-dataset report."
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Do not change the pipeline yet. The cross-dataset evidence separates "
            "three effects:",
            "",
            "- NFCorpus has a severe fixed-candidate limitation and a documented "
            "compact-query/source-context component.",
            "- SciFact generalizes better at the first retrieval stage under the "
            "same frozen BM25 setup.",
            "- The cross-encoder improves ranking when relevant documents are "
            "present, but cannot recover candidates missing from the first stage.",
            "",
            "The next controlled change, if any, should be retrieval-side and "
            "predeclared: candidate depth, query representation, hybrid retrieval, "
            "or a stronger first-stage retriever. It should not be selected merely "
            "from the already-inspected failure cases.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "make analyze-cross-dataset-errors",
            "```",
            "",
            "The target first verifies the BEIR release bundle and both dataset "
            "contracts, then writes:",
            "",
            "- `outputs/analysis/cross_dataset_errors/summary.json`",
            "- `outputs/analysis/cross_dataset_errors/report.md`",
            "",
            "Source diagnostics:",
            "",
            f"- [{nfcorpus_doc}]({nfcorpus_doc})",
            f"- [{scifact_doc}]({scifact_doc})",
            "",
            "## Limitations",
            "",
            "- This is retrieval-only evidence; no generation run is evaluated on "
            "NFCorpus or SciFact.",
            "- The analysis uses public qrels as the relevance ground truth and "
            "does not infer relevance for unjudged documents.",
            "- The comparison uses the frozen BM25 and cross-encoder outputs from "
            "the released BEIR bundle; it does not measure a new retriever.",
            "- NFCorpus and SciFact have very different qrels densities, so macro "
            "recall, query-count failures, and positive-qrel mass are reported "
            "separately.",
            "",
        ]
    )
    return "\n".join(lines)


def cross_dataset_fingerprint(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact counts suitable for a checked contract."""
    rows = {
        str(row["dataset"]): row
        for row in report.get("datasets", [])
        if isinstance(row, Mapping)
    }
    return {
        label: {
            "n_queries": int(row["n_queries"]),
            "n_positive_qrels": int(row["n_positive_qrels"]),
            "bm25_recall_at_100": round(float(row["bm25_recall_at_100"]), 10),
            "bm25_recall_at_1000": round(float(row["bm25_recall_at_1000"]), 10),
            "queries_with_no_relevant_top_100": int(
                row["queries_with_no_relevant_top_100"]
            ),
            "queries_with_complete_coverage_top_100": int(
                row["queries_with_complete_coverage_top_100"]
            ),
            "first_relevant_only_at_101_1000": int(
                row["first_relevant_only_at_101_1000"]
            ),
            "queries_still_missing_at_1000": int(
                row["queries_still_missing_at_1000"]
            ),
        }
        for label, row in sorted(rows.items())
    }


def assert_cross_dataset_fingerprint(
    report: Mapping[str, Any],
    expected: Any,
) -> None:
    """Reject drift against a compact cross-dataset evidence contract."""
    if not isinstance(expected, Mapping):
        raise CrossDatasetErrorAnalysisError("expected fingerprint must be an object")
    observed = cross_dataset_fingerprint(report)
    if observed != expected:
        differing = sorted(
            key
            for key in set(observed) | set(expected)
            if observed.get(key) != expected.get(key)
        )
        raise CrossDatasetErrorAnalysisError(
            "cross-dataset diagnostic drift in: " + ", ".join(differing)
        )
