"""Unit tests for ``src.evaluation.retrieval``.

These are pure-Python; no network, no dependencies beyond pytest.
"""

from __future__ import annotations

import math

import pytest

from src.evaluation.retrieval import (
    evaluate_retrieval,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


# --------------------------------------------------------------------------- #
# reciprocal_rank
# --------------------------------------------------------------------------- #

class TestReciprocalRank:
    def test_relevant_at_rank_1(self):
        assert reciprocal_rank(["d1", "d2", "d3"], {"d1"}, k=10) == 1.0

    def test_relevant_at_rank_2(self):
        assert reciprocal_rank(["d2", "d1", "d3"], {"d1"}, k=10) == 0.5

    def test_relevant_at_rank_3(self):
        assert reciprocal_rank(["d2", "d3", "d1"], {"d1"}, k=10) == pytest.approx(1 / 3)

    def test_no_relevant_in_topk(self):
        assert reciprocal_rank(["d2", "d3"], {"d1"}, k=10) == 0.0

    def test_relevant_outside_k(self):
        # d1 at rank 11; k=10 -> miss
        retrieved = ["d2"] * 10 + ["d1"]
        assert reciprocal_rank(retrieved, {"d1"}, k=10) == 0.0

    def test_first_of_multiple_relevants(self):
        # rank 2 (d2) is relevant; rank 3 (d1) also; should pick the earlier
        assert reciprocal_rank(["d3", "d2", "d1"], {"d1", "d2"}, k=10) == 0.5

    def test_empty_relevant(self):
        assert reciprocal_rank(["d1"], set(), k=10) == 0.0


# --------------------------------------------------------------------------- #
# recall_at_k
# --------------------------------------------------------------------------- #

class TestRecallAtK:
    def test_full_hit(self):
        assert recall_at_k(["d1", "d2", "d3"], {"d1", "d2"}, k=3) == 1.0

    def test_partial_hit(self):
        # 1 of 2 relevant retrieved in top-2
        assert recall_at_k(["d1", "d3"], {"d1", "d2"}, k=2) == 0.5

    def test_no_hit(self):
        assert recall_at_k(["d3", "d4"], {"d1", "d2"}, k=2) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k(["d1"], set(), k=10) == 0.0

    def test_k_clips_retrieved(self):
        # d2 is relevant but at rank 5; k=3 should miss
        retrieved = ["d1", "d3", "d4", "d5", "d2"]
        assert recall_at_k(retrieved, {"d2"}, k=3) == 0.0
        assert recall_at_k(retrieved, {"d2"}, k=5) == 1.0


# --------------------------------------------------------------------------- #
# ndcg_at_k
# --------------------------------------------------------------------------- #

class TestNDCGAtK:
    def test_perfect_ranking(self):
        # The single relevant doc is at rank 1; nDCG should be 1.0
        assert ndcg_at_k(["d1", "d2", "d3"], {"d1"}, k=10) == 1.0

    def test_two_relevant_perfect(self):
        # Both relevant at top -> ideal DCG; nDCG = 1.0
        assert ndcg_at_k(["d1", "d2", "d3"], {"d1", "d2"}, k=10) == pytest.approx(1.0)

    def test_one_relevant_at_rank_2(self):
        # DCG = 1 / log2(3); IDCG = 1 / log2(2) = 1
        expected = 1 / math.log2(3)
        assert ndcg_at_k(["d2", "d1"], {"d1"}, k=10) == pytest.approx(expected)

    def test_no_relevant(self):
        assert ndcg_at_k(["d1", "d2"], {"d3"}, k=10) == 0.0

    def test_empty_relevant(self):
        assert ndcg_at_k(["d1", "d2"], set(), k=10) == 0.0


# --------------------------------------------------------------------------- #
# evaluate_retrieval (the aggregator)
# --------------------------------------------------------------------------- #

class TestEvaluateRetrieval:
    def test_basic_two_query_case(self):
        runs = {
            "q1": ["d1", "d2", "d3"],   # relevant d1 at rank 1
            "q2": ["d9", "d1", "d2"],   # relevant d2 at rank 3
        }
        qrels = {"q1": {"d1"}, "q2": {"d2"}}
        m = evaluate_retrieval(runs, qrels, ks_mrr=(10,), ks_ndcg=(10,), ks_recall=(2, 3))

        # MRR@10: (1 + 1/3) / 2
        assert m["mrr@10"] == pytest.approx((1.0 + 1 / 3) / 2)
        # nDCG@10:
        #   q1: dcg=1/log2(2)=1, idcg=1, ndcg=1
        #   q2: dcg=1/log2(4)=0.5, idcg=1, ndcg=0.5
        assert m["ndcg@10"] == pytest.approx(0.75)
        # Recall@2:
        #   q1 -> 1.0 (d1 in top2), q2 -> 0 (d2 at rank 3 misses top2)
        assert m["recall@2"] == pytest.approx(0.5)
        # Recall@3: both hit
        assert m["recall@3"] == pytest.approx(1.0)
        assert m["n_queries"] == 2

    def test_skips_queries_without_qrels(self):
        runs = {"q1": ["d1"], "q2": ["d2"]}
        qrels = {"q1": {"d1"}}  # q2 has no qrels -> skipped
        m = evaluate_retrieval(runs, qrels, ks_mrr=(10,), ks_ndcg=(10,), ks_recall=(10,))
        assert m["n_queries"] == 1
        # Only q1 contributes: MRR=1, nDCG=1, Recall=1
        assert m["mrr@10"] == 1.0
        assert m["ndcg@10"] == 1.0
        assert m["recall@10"] == 1.0

    def test_no_evaluable_queries(self):
        # All queries lack qrels -> empty result
        m = evaluate_retrieval({"q1": ["d1"]}, {}, ks_mrr=(10,), ks_ndcg=(10,), ks_recall=(10,))
        assert m == {"n_queries": 0}

    def test_multiple_ks(self):
        runs = {"q1": list(f"d{i}" for i in range(20))}
        qrels = {"q1": {"d15"}}
        m = evaluate_retrieval(
            runs, qrels, ks_mrr=(10, 20), ks_ndcg=(10, 20), ks_recall=(10, 20)
        )
        # d15 at rank 16: missed at k=10, hit at k=20
        assert m["mrr@10"] == 0.0
        assert m["mrr@20"] == pytest.approx(1 / 16)
        assert m["recall@10"] == 0.0
        assert m["recall@20"] == 1.0
        assert m["ndcg@10"] == 0.0
        assert m["ndcg@20"] == pytest.approx(1 / math.log2(17))
