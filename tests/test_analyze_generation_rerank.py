"""Tests for ``scripts/analyze_generation_rerank.py`` helpers.

We test the pure pieces only — bucketing, per-query metric assembly, and
aggregation — since the actual analysis script depends on prediction files,
qrels, and the MS MARCO QA dataset.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Scripts aren't a package — load by file path so we can import the helpers
# without needing a sys.path hack at module import time.
_spec = importlib.util.spec_from_file_location(
    "analyze_generation_rerank",
    PROJECT_ROOT / "scripts" / "analyze_generation_rerank.py",
)
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)


# --------------------------------------------------------------------------- #
# has_relevant_in_top3
# --------------------------------------------------------------------------- #


class TestHasRelevantInTop3:
    def test_relevant_at_rank_1(self):
        assert analyze.has_relevant_in_top3(["a", "b", "c"], {"a"}) is True

    def test_relevant_at_rank_3(self):
        assert analyze.has_relevant_in_top3(["x", "y", "rel"], {"rel"}) is True

    def test_relevant_at_rank_4_does_not_count(self):
        assert analyze.has_relevant_in_top3(["x", "y", "z", "rel"], {"rel"}) is False

    def test_empty_qrels(self):
        assert analyze.has_relevant_in_top3(["a", "b", "c"], set()) is False

    def test_empty_top_docs(self):
        assert analyze.has_relevant_in_top3([], {"rel"}) is False


# --------------------------------------------------------------------------- #
# assign_bucket
# --------------------------------------------------------------------------- #


class TestAssignBucket:
    def _bucket(self, bm25_f1, rerank_f1, bm25_ret, rerank_ret, improve=0.2, failing=0.3):
        return analyze.assign_bucket(
            bm25_f1=bm25_f1,
            rerank_f1=rerank_f1,
            bm25_retrieved=bm25_ret,
            rerank_retrieved=rerank_ret,
            f1_improve_threshold=improve,
            f1_failing_threshold=failing,
        )

    def test_rerank_fixed_and_generation_improved(self):
        # BM25 had no relevant doc; rerank brought one in; F1 jumped.
        assert self._bucket(0.1, 0.5, False, True) == "rerank_fixed_generation_improved"

    def test_rerank_fixed_but_generation_still_failing(self):
        # Retrieval gain didn't help the generator — F1 still low.
        assert self._bucket(0.05, 0.15, False, True) == "rerank_fixed_generation_still_failing"

    def test_regression(self):
        # Reranking made generation worse by a meaningful margin.
        assert self._bucket(0.7, 0.3, True, True) == "regression"

    def test_retrieval_equivalent_generation_differs(self):
        # Both retrieved relevant; generation differs by >threshold (improving).
        assert self._bucket(0.3, 0.6, True, True) == "retrieval_equivalent_generation_differs"

    def test_no_signal_small_delta(self):
        assert self._bucket(0.5, 0.55, True, True) == "no_signal"

    def test_no_signal_neither_retrieved(self):
        # Neither side has a relevant qrel in top-3, delta is small.
        assert self._bucket(0.1, 0.12, False, False) == "no_signal"

    def test_priority_fixed_improved_over_differs(self):
        # When both flags fire, the 'rerank_fixed_improved' label is more
        # informative and takes priority.
        assert (
            self._bucket(0.1, 0.5, False, True)
            == "rerank_fixed_generation_improved"
        )

    def test_threshold_boundary(self):
        # Exactly at the threshold: counts as a meaningful change.
        assert self._bucket(0.2, 0.4, True, True) == "retrieval_equivalent_generation_differs"


# --------------------------------------------------------------------------- #
# aggregate_by_key
# --------------------------------------------------------------------------- #


class TestAggregateByKey:
    def test_group_means_and_counts(self):
        rows = [
            {"query_type": "NUMERIC", "delta": 0.1},
            {"query_type": "NUMERIC", "delta": 0.3},
            {"query_type": "DESCRIPTION", "delta": 0.5},
        ]
        agg = analyze.aggregate_by_key(rows, "query_type", "delta")
        assert agg["NUMERIC"]["count"] == 2
        assert abs(agg["NUMERIC"]["mean"] - 0.2) < 1e-9
        assert agg["DESCRIPTION"]["count"] == 1
        assert agg["DESCRIPTION"]["mean"] == 0.5

    def test_unknown_key_bucket(self):
        rows = [
            {"query_type": None, "delta": 1.0},
            {"delta": 2.0},  # key missing entirely
            {"query_type": "ENTITY", "delta": 0.0},
        ]
        agg = analyze.aggregate_by_key(rows, "query_type", "delta")
        # Both None and "missing" should collapse into UNKNOWN.
        assert agg["UNKNOWN"]["count"] == 2
        assert abs(agg["UNKNOWN"]["mean"] - 1.5) < 1e-9

    def test_empty_rows(self):
        assert analyze.aggregate_by_key([], "query_type", "delta") == {}


# --------------------------------------------------------------------------- #
# load_predictions
# --------------------------------------------------------------------------- #


def test_load_predictions_basic(tmp_path: Path):
    p = tmp_path / "predictions.jsonl"
    p.write_text(
        '{"query_id": 1, "prediction": "a", "references": ["x"], "top_doc_ids": ["d1"], "passages": [], "query": "q1"}\n'
        '{"query_id": 2, "prediction": "b", "references": ["y"], "top_doc_ids": ["d2"], "passages": [], "query": "q2"}\n'
    )
    out = analyze.load_predictions(p)
    # Keys are stringified so the analysis can join against ms_marco query_ids,
    # which are strings.
    assert set(out.keys()) == {"1", "2"}
    assert out["1"]["prediction"] == "a"
