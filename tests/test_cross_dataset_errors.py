"""Tests for cross-dataset first-stage error analysis."""

from __future__ import annotations

import pytest

from msmarco_genqa.evaluation.cross_dataset_errors import (
    CrossDatasetErrorAnalysisError,
    assert_cross_dataset_fingerprint,
    build_cross_dataset_error_analysis,
    cross_dataset_fingerprint,
    render_cross_dataset_error_markdown,
    summarize_taxonomy_rows,
)


def _summary(
    *,
    dataset_id: str,
    n_queries: int,
    n_positive: int,
    recall_100: float,
    recall_1000: float,
    no_top100: int,
    partial_top100: int,
    complete_top100: int,
    first_101_1000: int,
    miss_1000: int,
) -> dict:
    return {
        "dataset_id": dataset_id,
        "scope": {
            "n_queries": n_queries,
            "n_positive_qrels": n_positive,
            "relevant_per_query": {
                "mean": n_positive / n_queries,
                "median": 1.0,
            },
        },
        "macro_recall": {
            "recall@100": recall_100,
            "recall@1000": recall_1000,
        },
        "first_hit_buckets": [
            {"bucket": "top_10", "n_queries": n_queries - no_top100 - 1},
            {"bucket": "ranks_11_100", "n_queries": 1},
            {"bucket": "ranks_101_1000", "n_queries": first_101_1000},
            {"bucket": "miss_top_1000", "n_queries": miss_1000},
        ],
        "coverage_at_100_buckets": [
            {"bucket": "none", "n_queries": no_top100},
            {"bucket": "partial_lt_25pct", "n_queries": 0},
            {"bucket": "partial_25_to_lt_50pct", "n_queries": 0},
            {"bucket": "partial_50_to_lt_100pct", "n_queries": partial_top100},
            {"bucket": "complete", "n_queries": complete_top100},
        ],
        "candidate_set_diagnostic": {
            "queries_with_any_relevant_in_top_100": n_queries - no_top100,
            "queries_with_no_relevant_in_top_100": no_top100,
            "queries_with_complete_relevant_coverage_at_100": complete_top100,
            "queries_with_partial_relevant_coverage_at_100": partial_top100,
            "positive_qrels_in_top_100": n_positive - 5,
            "positive_qrels_outside_top_100": 5,
        },
        "depth_100_to_1000_diagnostic": {
            "positive_qrels_still_missing_at_depth_1000": 2,
            "additional_positive_qrels_found": 3,
            "queries_gaining_relevant_documents": first_101_1000,
        },
    }


def _analysis():
    taxonomy = summarize_taxonomy_rows(
        [
            {
                "qid": "q1",
                "cohort": "miss_top_1000",
                "review_status": "reviewed",
                "primary_label": "source_context_dependency",
            },
            {
                "qid": "q2",
                "cohort": "depth_recoverable_101_1000",
                "review_status": "reviewed",
                "primary_label": "vocabulary_or_form_mismatch",
            },
        ]
    )
    scifact_review = {
        "n_cases": 2,
        "primary_label_counts": {
            "terminology_or_evidence_form_mismatch": 1,
            "lexical_competition_at_depth_cutoff": 1,
            "short_or_broad_claim": 0,
        },
    }
    return build_cross_dataset_error_analysis(
        {
            "NFCorpus": _summary(
                dataset_id="beir/nfcorpus/test",
                n_queries=10,
                n_positive=100,
                recall_100=0.20,
                recall_1000=0.50,
                no_top100=4,
                partial_top100=5,
                complete_top100=1,
                first_101_1000=1,
                miss_1000=3,
            ),
            "SciFact": _summary(
                dataset_id="beir/scifact/test",
                n_queries=10,
                n_positive=10,
                recall_100=0.80,
                recall_1000=0.90,
                no_top100=2,
                partial_top100=1,
                complete_top100=7,
                first_101_1000=1,
                miss_1000=1,
            ),
        },
        nfcorpus_taxonomy=taxonomy,
        scifact_residual_review=scifact_review,
    )


def test_cross_dataset_analysis_computes_expected_gaps():
    report = _analysis()

    assert report["schema"] == "msmarco-genqa.cross-dataset-error-analysis.v1"
    assert report["comparison"]["recall_at_100_gap_scifact_minus_nfcorpus"] == (
        pytest.approx(0.60)
    )
    assert report["comparison"][
        "no_relevant_top_100_share_gap_nfcorpus_minus_scifact"
    ] == pytest.approx(0.20)
    assert report["comparison"][
        "complete_top_100_share_gap_scifact_minus_nfcorpus"
    ] == pytest.approx(0.60)
    assert report["nfcorpus_manual_taxonomy"]["n_reviewed"] == 2


def test_cross_dataset_analysis_rejects_unreconciled_counts():
    bad = _summary(
        dataset_id="beir/nfcorpus/test",
        n_queries=10,
        n_positive=100,
        recall_100=0.20,
        recall_1000=0.50,
        no_top100=4,
        partial_top100=5,
        complete_top100=1,
        first_101_1000=2,
        miss_1000=3,
    )

    with pytest.raises(CrossDatasetErrorAnalysisError, match="partition"):
        build_cross_dataset_error_analysis(
            {
                "NFCorpus": bad,
                "SciFact": _summary(
                    dataset_id="beir/scifact/test",
                    n_queries=10,
                    n_positive=10,
                    recall_100=0.80,
                    recall_1000=0.90,
                    no_top100=2,
                    partial_top100=1,
                    complete_top100=7,
                    first_101_1000=1,
                    miss_1000=1,
                ),
            }
        )


def test_cross_dataset_fingerprint_rejects_drift():
    report = _analysis()
    expected = cross_dataset_fingerprint(report)

    assert_cross_dataset_fingerprint(report, expected)
    drifted = {
        **expected,
        "SciFact": {
            **expected["SciFact"],
            "queries_with_no_relevant_top_100": 999,
        },
    }
    with pytest.raises(CrossDatasetErrorAnalysisError, match="drift"):
        assert_cross_dataset_fingerprint(report, drifted)


def test_cross_dataset_markdown_states_decision_boundary():
    markdown = render_cross_dataset_error_markdown(
        _analysis(),
        nfcorpus_doc="docs/nfcorpus.md",
        scifact_doc="docs/scifact.md",
    )

    assert markdown.startswith("# Cross-Dataset First-Stage Error Analysis")
    assert "Do not change the pipeline yet" in markdown
    assert "retrieval-only evidence" in markdown
    assert "source_context_dependency" in markdown
    assert "terminology_or_evidence_form_mismatch" in markdown
