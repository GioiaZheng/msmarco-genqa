from __future__ import annotations

from pathlib import Path

import pytest

from scripts.analyze_nfcorpus_query_representations import (
    Condition,
    build_paired_comparisons,
    build_per_query_rows,
    validate_rerank_contract,
)
from msmarco_genqa.data.nfcorpus_video import (
    FIXED_RERANKER_MODEL,
    FIXED_RERANKER_REVISION,
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


def _condition(
    name: str,
    stage: str,
    run: dict[str, list[tuple[str, float]]],
) -> Condition:
    query_summary = {
        "representation": name,
        "qid_sha256": "qids",
        "official_query_records_sha256": "records",
        "effective_queries_sha256": f"queries-{name}",
    }
    metrics_payload: dict[str, object] = {}
    if stage == "cross_encoder_rerank":
        metrics_payload["config"] = {
            "reranker": {
                "model_name": FIXED_RERANKER_MODEL,
                "revision": FIXED_RERANKER_REVISION,
                "rerank_top_k": 100,
                "max_length": 512,
            }
        }
    return Condition(
        name=name,
        stage=stage,
        directory=Path(name) / stage,
        run=run,
        metrics_payload=metrics_payload,
        manifest={"git": {"commit": "abc123", "dirty": False}},
        query_summary=query_summary,
    )


def test_rerank_contract_requires_exact_bm25_candidate_sets() -> None:
    bm25 = {
        name: _condition(
            name,
            "bm25",
            {"q1": [("d1", 2.0), ("d2", 1.0), ("d3", 0.0)]},
        )
        for name in ("title", "description", "title_plus_description")
    }
    rerank = {
        name: _condition(
            name,
            "cross_encoder_rerank",
            {"q1": [("d2", 2.0), ("d1", 1.0)]},
        )
        for name in ("title", "description", "title_plus_description")
    }

    contract = validate_rerank_contract(
        bm25,
        rerank,
        expected_queries=1,
        expected_depth=2,
    )

    assert contract["candidate_set_checks"] == 3
    assert contract["candidate_sets_match_bm25_top_100"] is True

    rerank["description"].run["q1"] = [("d2", 2.0), ("other", 1.0)]
    with pytest.raises(ValueError, match="candidate set differs"):
        validate_rerank_contract(
            bm25,
            rerank,
            expected_queries=1,
            expected_depth=2,
        )


def test_rerank_rows_omit_invalid_recall_at_1000() -> None:
    runs = {
        name: {"q1": [("d1", 1.0)]}
        for name in ("title", "description", "title_plus_description")
    }
    rows = build_per_query_rows(
        runs,
        {"q1": {"d1": 1}},
        include_recall_at_1000=False,
    )
    comparisons = build_paired_comparisons(
        rows,
        n_resamples=20,
        seed=7,
        metric_names=("rr@10", "ndcg@10", "recall@100"),
        no_hit_cutoffs=(100,),
    )

    assert "recall@1000" not in rows[0]["conditions"]["title"]
    assert "recall@1000" not in comparisons["description_vs_title"]["metrics"]
    assert set(comparisons["description_vs_title"]["no_hit_queries"]) == {"@100"}
