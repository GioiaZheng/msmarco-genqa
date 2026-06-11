"""Tests for ``msmarco_genqa.reranking``.

We stub the underlying ``CrossEncoder`` so the tests don't pull weights
from HuggingFace. The interesting behaviour is the per-query re-sort
and the run.tsv I/O — both are exercised here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from msmarco_genqa.reranking.cross_encoder import CrossEncoderReranker
from msmarco_genqa.reranking.io import (
    RunTsvFormatError,
    append_run_tsv,
    collect_unique_doc_ids,
    prune_partial_qids,
    read_done_qids,
    read_run_tsv,
    truncate_top_k,
    write_run_tsv,
)


# --------------------------------------------------------------------------- #
# Stub cross-encoder
# --------------------------------------------------------------------------- #


class _StubCE:
    """Returns one score per (query, doc) pair: a deterministic function
    of how many shared tokens (case-insensitive) appear in the pair.

    Score = #shared_tokens + 0.01 * len(doc) so that, for ties on overlap,
    longer docs sort higher (a stable, easy-to-test rule).
    """

    def predict(self, pairs, batch_size=64, show_progress_bar=False, convert_to_numpy=True):
        scores = []
        for q, d in pairs:
            qt = set(q.lower().split())
            dt = d.lower().split()
            shared = sum(1 for w in dt if w in qt)
            scores.append(float(shared) + 0.01 * len(dt))
        return np.asarray(scores, dtype=np.float32)


def _make_reranker():
    r = CrossEncoderReranker(model_name="stub", device="cpu", batch_size=4)
    r._model = _StubCE()
    return r


# --------------------------------------------------------------------------- #
# CrossEncoderReranker
# --------------------------------------------------------------------------- #


def test_score_returns_one_per_pair():
    r = _make_reranker()
    pairs = [
        ("paris france", "the eiffel tower is in paris france"),
        ("paris france", "kangaroos live in australia"),
    ]
    scores = r.score(pairs, show_progress_bar=False)
    assert scores.shape == (2,)
    # First pair shares "paris" and "france"; second shares nothing.
    assert scores[0] > scores[1]


def test_score_empty_returns_empty():
    r = _make_reranker()
    out = r.score([], show_progress_bar=False)
    assert out.shape == (0,)


def test_rerank_batch_resorts_per_query():
    r = _make_reranker()
    queries = ["paris france", "australia kangaroo"]
    candidates = [
        # Query 0: order is intentionally bad — best match is last.
        [
            ("d_irrelevant", "kangaroos live in australia"),
            ("d_partial", "france is a country in europe"),
            ("d_best", "the eiffel tower is in paris france"),
        ],
        [
            ("d_aus_best", "kangaroos hop around australia widely"),
            ("d_unrelated", "paris is the capital of france"),
        ],
    ]
    reranked, info = r.rerank_batch(queries, candidates, show_progress_bar=False)

    assert len(reranked) == 2
    # Query 0: best should now be on top.
    q0 = reranked[0]
    assert q0[0][0] == "d_best"
    # Scores must be sorted descending within each query.
    assert all(q0[i][1] >= q0[i + 1][1] for i in range(len(q0) - 1))

    # Query 1: the kangaroo doc wins over a paris-france doc.
    assert reranked[1][0][0] == "d_aus_best"

    assert info["n_pairs"] == 5
    assert info["score_seconds"] >= 0.0


def test_rerank_batch_preserves_doc_id_count_per_query():
    r = _make_reranker()
    queries = ["q one", "q two"]
    candidates = [
        [("a", "alpha"), ("b", "bravo")],
        [("c", "charlie")],
    ]
    reranked, _ = r.rerank_batch(queries, candidates, show_progress_bar=False)
    assert [len(x) for x in reranked] == [2, 1]
    assert {d for d, _ in reranked[0]} == {"a", "b"}
    assert reranked[1][0][0] == "c"


def test_rerank_batch_handles_empty_candidate_lists():
    r = _make_reranker()
    queries = ["q1", "q2", "q3"]
    candidates = [
        [],
        [("d0", "hello world")],
        [],
    ]
    reranked, info = r.rerank_batch(queries, candidates, show_progress_bar=False)
    # Stub score: 0 shared tokens + 0.01 * 2 doc tokens = 0.02
    assert reranked[0] == [] and reranked[2] == []
    assert reranked[1][0][0] == "d0"
    assert reranked[1][0][1] == pytest.approx(0.02, rel=1e-4)
    assert info["n_pairs"] == 1


def test_rerank_batch_raises_on_mismatched_lengths():
    r = _make_reranker()
    with pytest.raises(ValueError):
        r.rerank_batch(["q"], [], show_progress_bar=False)


# --------------------------------------------------------------------------- #
# io helpers
# --------------------------------------------------------------------------- #


def test_read_run_tsv_roundtrip(tmp_path):
    path = tmp_path / "run.tsv"
    write_run_tsv(
        path,
        qids=["q1", "q2"],
        doc_ids_lists=[["d_a", "d_b", "d_c"], ["d_x"]],
        scores_lists=[[3.0, 2.5, 1.0], [9.9]],
        system_name="test",
    )
    runs = read_run_tsv(path)
    assert list(runs.keys()) == ["q1", "q2"]
    assert runs["q1"] == [("d_a", 3.0), ("d_b", 2.5), ("d_c", 1.0)]
    assert runs["q2"] == [("d_x", 9.9)]


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("q1\tQ0\td1\t1\t1.0\n", "expected 6 tab-separated fields"),
        ("\tQ0\td1\t1\t1.0\tdense\n", "empty query id"),
        ("q1\tQ0\t\t1\t1.0\tdense\n", "empty document id"),
        ("q1\tQ0\td1\t0\t1.0\tdense\n", "rank must be positive"),
        ("q1\tQ0\td1\tNOT_A_RANK\t1.0\tdense\n", "rank is not an integer"),
        ("q1\tQ0\td1\t1\tNOT_A_SCORE\tdense\n", "score is not numeric"),
        ("q1\tQ0\td1\t1\tnan\tdense\n", "score must be finite"),
        ("q1\tQ0\td\ufffd\t1\t1.0\tdense\n", "replacement character"),
    ],
)
def test_read_run_tsv_rejects_malformed_lines(tmp_path, line, message):
    path = tmp_path / "run.tsv"
    path.write_text(line, encoding="utf-8")
    with pytest.raises(RunTsvFormatError, match=message) as excinfo:
        read_run_tsv(path)
    assert excinfo.value.line_number == 1


def test_read_run_tsv_rejects_duplicate_rank_per_query(tmp_path):
    path = tmp_path / "run.tsv"
    path.write_text(
        "q1\tQ0\td1\t1\t2.0\tdense\n"
        "q1\tQ0\td2\t1\t1.0\tdense\n",
        encoding="utf-8",
    )
    with pytest.raises(RunTsvFormatError, match="duplicate rank 1"):
        read_run_tsv(path)


def test_read_run_tsv_rejects_duplicate_doc_per_query(tmp_path):
    path = tmp_path / "run.tsv"
    path.write_text(
        "q1\tQ0\td1\t1\t2.0\tdense\n"
        "q1\tQ0\td1\t2\t1.0\tdense\n",
        encoding="utf-8",
    )
    with pytest.raises(RunTsvFormatError, match="duplicate document id"):
        read_run_tsv(path)


def test_read_run_tsv_rejects_non_utf8_bytes(tmp_path):
    path = tmp_path / "run.tsv"
    path.write_bytes(b"q1\tQ0\td1\t1\t1.0\tdense\n\xff")
    with pytest.raises(RunTsvFormatError, match="not valid UTF-8") as excinfo:
        read_run_tsv(path)
    assert excinfo.value.line_number is None


def test_truncate_top_k():
    runs = {
        "q1": [("a", 5.0), ("b", 4.0), ("c", 3.0), ("d", 2.0)],
        "q2": [("x", 1.0)],
    }
    out = truncate_top_k(runs, 2)
    assert out == {"q1": [("a", 5.0), ("b", 4.0)], "q2": [("x", 1.0)]}
    # Original is not mutated.
    assert len(runs["q1"]) == 4


def test_collect_unique_doc_ids_preserves_first_seen_order():
    runs = {
        "q1": [("a", 1.0), ("b", 1.0)],
        "q2": [("b", 1.0), ("c", 1.0)],
        "q3": [("a", 1.0), ("c", 1.0), ("d", 1.0)],
    }
    assert collect_unique_doc_ids(runs) == ["a", "b", "c", "d"]


def test_write_run_tsv_format(tmp_path):
    path = tmp_path / "run.tsv"
    write_run_tsv(
        path,
        qids=["q1"],
        doc_ids_lists=[["d_a", "d_b"]],
        scores_lists=[[2.0, 1.0]],
        system_name="my_sys",
    )
    lines = path.read_text().strip().splitlines()
    assert lines[0] == "q1\tQ0\td_a\t1\t2.000000\tmy_sys"
    assert lines[1] == "q1\tQ0\td_b\t2\t1.000000\tmy_sys"


# --------------------------------------------------------------------------- #
# Resume helpers: append_run_tsv / read_done_qids / prune_partial_qids
# --------------------------------------------------------------------------- #


def test_append_run_tsv_creates_then_appends(tmp_path):
    """First call creates the file; second call appends without truncating."""
    path = tmp_path / "run.tsv"
    append_run_tsv(
        path,
        qids=["q1"],
        doc_ids_lists=[["d_a", "d_b"]],
        scores_lists=[[2.0, 1.0]],
        system_name="ce",
    )
    append_run_tsv(
        path,
        qids=["q2"],
        doc_ids_lists=[["d_x"]],
        scores_lists=[[9.0]],
        system_name="ce",
    )
    runs = read_run_tsv(path)
    assert list(runs.keys()) == ["q1", "q2"]
    assert runs["q1"] == [("d_a", 2.0), ("d_b", 1.0)]
    assert runs["q2"] == [("d_x", 9.0)]


def test_read_done_qids_complete_only(tmp_path):
    """A qid is 'done' iff its block contains all ranks 1..top_k."""
    path = tmp_path / "run.tsv"
    # q1: full top-3, q2: only ranks 1 and 2 of top-3 (partial — was being
    # flushed when the process died), q3: full top-3.
    path.write_text(
        "q1\tQ0\td_a\t1\t9.0\tce\n"
        "q1\tQ0\td_b\t2\t8.0\tce\n"
        "q1\tQ0\td_c\t3\t7.0\tce\n"
        "q2\tQ0\td_x\t1\t5.0\tce\n"
        "q2\tQ0\td_y\t2\t4.0\tce\n"
        "q3\tQ0\td_p\t1\t3.0\tce\n"
        "q3\tQ0\td_q\t2\t2.0\tce\n"
        "q3\tQ0\td_r\t3\t1.0\tce\n"
    )
    done = read_done_qids(path, top_k=3)
    assert done == {"q1", "q3"}


def test_read_done_qids_handles_missing_file(tmp_path):
    """A nonexistent file returns the empty set — first run with --resume."""
    assert read_done_qids(tmp_path / "does_not_exist.tsv", top_k=3) == set()


def test_read_done_qids_max_rank_mismatch_excluded(tmp_path):
    """If max rank > top_k (shouldn't happen, but guard), the qid is dropped.

    Practical case: a previous run used a different ``top_k`` and we're now
    resuming with a different ``top_k``. Safer to re-score than silently
    accept stale entries.
    """
    path = tmp_path / "run.tsv"
    path.write_text(
        # q1 has 3 entries but max rank is 5 — inconsistent, so NOT done.
        "q1\tQ0\td_a\t1\t9.0\tce\n"
        "q1\tQ0\td_b\t3\t7.0\tce\n"
        "q1\tQ0\td_c\t5\t6.0\tce\n"
    )
    assert read_done_qids(path, top_k=3) == set()


def test_prune_partial_qids_keeps_only_done(tmp_path):
    """Drops lines for qids not in the done set; returns the drop count."""
    path = tmp_path / "run.tsv"
    path.write_text(
        "q1\tQ0\td_a\t1\t9.0\tce\n"
        "q1\tQ0\td_b\t2\t8.0\tce\n"
        "q2\tQ0\td_x\t1\t5.0\tce\n"  # half-written, should be dropped
        "q3\tQ0\td_p\t1\t3.0\tce\n"
    )
    dropped = prune_partial_qids(path, keep_qids={"q1", "q3"})
    assert dropped == 1
    text = path.read_text()
    assert "q2" not in text
    assert text.count("\n") == 3


def test_resume_round_trip_skips_done_appends_new(tmp_path):
    """End-to-end resume flow: do qids 1-2, prune, then append qids 3-4."""
    path = tmp_path / "run.tsv"

    # Phase 1: write q1 and q2 (complete top-2 each), then a partial q3.
    append_run_tsv(
        path,
        qids=["q1", "q2"],
        doc_ids_lists=[["d_a", "d_b"], ["d_c", "d_d"]],
        scores_lists=[[2.0, 1.0], [3.0, 2.0]],
        system_name="ce",
    )
    # Simulate a kill mid-chunk: open the file and append ONE rank-1 line for q3.
    with open(path, "a") as f:
        f.write("q3\tQ0\td_e\t1\t9.0\tce\n")

    # On resume: identify done, prune partials, compute pending.
    done = read_done_qids(path, top_k=2)
    assert done == {"q1", "q2"}
    pending = [q for q in ["q1", "q2", "q3", "q4"] if q not in done]
    assert pending == ["q3", "q4"]
    prune_partial_qids(path, keep_qids=done)

    # Phase 2: append q3 and q4 (the pending ones) with full top-2 blocks.
    append_run_tsv(
        path,
        qids=["q3", "q4"],
        doc_ids_lists=[["d_e_new", "d_f"], ["d_g", "d_h"]],
        scores_lists=[[7.0, 6.0], [5.0, 4.0]],
        system_name="ce",
    )

    # Final file should have all four qids with NO duplicates for q3.
    runs = read_run_tsv(path)
    assert set(runs) == {"q1", "q2", "q3", "q4"}
    assert runs["q3"] == [("d_e_new", 7.0), ("d_f", 6.0)]  # stale d_e dropped
    # And on a third "resume" pass, all four would be reported done.
    assert read_done_qids(path, top_k=2) == {"q1", "q2", "q3", "q4"}


# --------------------------------------------------------------------------- #
# Integration: read → truncate → rerank → write
# --------------------------------------------------------------------------- #


def test_end_to_end_pipeline_with_stub(tmp_path):
    # Synthetic first-stage run with a clearly wrong order.
    in_path = tmp_path / "in.tsv"
    write_run_tsv(
        in_path,
        qids=["q1"],
        doc_ids_lists=[["d_kangaroo", "d_partial", "d_best"]],
        scores_lists=[[5.0, 4.0, 3.0]],
        system_name="dense",
    )

    runs = read_run_tsv(in_path)
    runs = truncate_top_k(runs, 3)

    text_by_id = {
        "d_kangaroo": "kangaroos live in australia",
        "d_partial": "france is a country in europe",
        "d_best": "the eiffel tower is in paris france",
    }
    qids = list(runs.keys())
    queries = ["paris france"]
    candidates = [
        [(d, text_by_id[d]) for d, _ in runs[q]] for q in qids
    ]

    r = _make_reranker()
    reranked, _ = r.rerank_batch(queries, candidates, show_progress_bar=False)

    out_path = tmp_path / "out.tsv"
    doc_ids_lists = [[d for d, _ in row] for row in reranked]
    scores_lists = [[s for _, s in row] for row in reranked]
    write_run_tsv(out_path, qids, doc_ids_lists, scores_lists, "dense+ce")

    out_runs = read_run_tsv(out_path)
    assert out_runs["q1"][0][0] == "d_best"
    assert math.isfinite(out_runs["q1"][0][1])
