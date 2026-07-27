from __future__ import annotations

from scripts.analyze_nfcorpus_query_representations import (
    build_paired_comparisons,
    build_per_query_rows,
)


def test_per_query_rows_keep_graded_ndcg_and_binary_recall() -> None:
    runs = {
        "title": {
            "q1": [("d2", 2.0), ("d1", 1.0)],
            "q2": [("x", 1.0), ("d3", 0.5)],
        },
        "description": {
            "q1": [("d1", 2.0), ("d2", 1.0)],
            "q2": [("d3", 1.0), ("x", 0.5)],
        },
        "title_plus_description": {
            "q1": [("d1", 2.0), ("d2", 1.0)],
            "q2": [("d3", 1.0), ("x", 0.5)],
        },
    }
    qrels = {
        "q1": {"d1": 2, "d2": 1},
        "q2": {"d3": 1},
    }

    rows = build_per_query_rows(runs, qrels)

    assert len(rows) == 2
    assert rows[0]["conditions"]["description"]["rr@10"] == 1.0
    assert rows[0]["conditions"]["description"]["ndcg@10"] == 1.0
    assert rows[1]["conditions"]["title"]["recall@100"] == 1.0
    assert rows[1]["conditions"]["title"]["first_relevant_rank@100"] == 2


def test_paired_comparison_reports_recovery_and_win_counts() -> None:
    runs = {
        "title": {
            "q1": [("x", 1.0)],
            "q2": [("d2", 1.0)],
        },
        "description": {
            "q1": [("d1", 1.0)],
            "q2": [("d2", 1.0)],
        },
        "title_plus_description": {
            "q1": [("d1", 1.0)],
            "q2": [("d2", 1.0)],
        },
    }
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    rows = build_per_query_rows(runs, qrels)

    comparisons = build_paired_comparisons(
        rows,
        n_resamples=200,
        seed=7,
    )

    result = comparisons["description_vs_title"]
    assert result["metrics"]["recall@100"]["mean_delta"] == 0.5
    assert result["metrics"]["recall@100"]["wins"] == 1
    assert result["metrics"]["recall@100"]["ties"] == 1
    assert result["metrics"]["recall@100"]["losses"] == 0
    assert result["no_hit_queries"]["@100"] == {
        "baseline": 1,
        "treatment": 0,
        "recovered": 1,
        "lost": 0,
    }
