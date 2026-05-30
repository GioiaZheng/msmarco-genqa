"""Tests for query-level retrieval lift diagnostics."""

from __future__ import annotations

import pytest

from msmarco_genqa.evaluation.retrieval import (
    compare_retrieval_runs_per_query,
    first_relevant_rank,
    query_retrieval_delta,
    retrieval_shift_bucket,
)


def test_first_relevant_rank_respects_cutoff():
    retrieved = ["d0", "d1", "d2", "d3"]

    assert first_relevant_rank(retrieved, {"d2"}, k=3) == 3
    assert first_relevant_rank(retrieved, {"d3"}, k=3) is None
    assert first_relevant_rank(retrieved, {"d3"}) == 4


@pytest.mark.parametrize(
    ("before", "after", "bucket"),
    [
        (None, None, "unchanged_miss"),
        (None, 3, "new_hit"),
        (4, None, "lost_hit"),
        (5, 2, "promoted"),
        (2, 5, "demoted"),
        (2, 2, "unchanged_hit"),
    ],
)
def test_retrieval_shift_bucket(before, after, bucket):
    assert retrieval_shift_bucket(before, after) == bucket


def test_query_retrieval_delta_promoted_case():
    row = query_retrieval_delta(
        "q1",
        before=["d0", "d1", "d2"],
        after=["d2", "d0", "d1"],
        relevant={"d2"},
        k_rank=10,
        k_recall=3,
    )

    assert row["bucket"] == "promoted"
    assert row["before_first_relevant_rank"] == 3
    assert row["after_first_relevant_rank"] == 1
    assert row["rank_movement"] == 2
    assert row["rr_delta@10"] == pytest.approx(1.0 - 1 / 3)
    assert row["recall_delta@3"] == 0.0


def test_query_retrieval_delta_new_hit_at_rank_cutoff():
    row = query_retrieval_delta(
        "q1",
        before=["d0", "d1", "d2", "d3"],
        after=["d9", "d8", "d7", "d4"],
        relevant={"d4"},
        k_rank=4,
        k_recall=4,
    )

    assert row["bucket"] == "new_hit"
    assert row["before_first_relevant_rank"] is None
    assert row["after_first_relevant_rank"] == 4
    assert row["rr_delta@4"] == pytest.approx(0.25)
    assert row["recall_delta@4"] == pytest.approx(1.0)


def test_compare_retrieval_runs_per_query_filters_to_shared_evaluable_qids():
    before = {
        "q1": ["d0", "d1"],
        "q2": ["d3"],
        "q3": ["d4"],
    }
    after = {
        "q1": ["d1", "d0"],
        "q2": ["d3"],
        "q4": ["d9"],
    }
    qrels = {
        "q1": {"d1"},
        "q2": set(),
        "q3": {"d4"},
        "q4": {"d9"},
    }

    rows = compare_retrieval_runs_per_query(before, after, qrels, k_rank=10, k_recall=2)

    assert [row["qid"] for row in rows] == ["q1"]
    assert rows[0]["bucket"] == "promoted"
