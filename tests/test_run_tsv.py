"""Unit tests for TREC-format ``run.tsv`` parsing.

Covers both ``experiments.run_retrieval._read_runs_from_tsv`` (returns
ranks + scores) and ``experiments.run_generation_baseline.load_runs``
(returns ranks only). They diverge in a known way; the tests pin both.
"""

from __future__ import annotations

import textwrap

import pytest

from experiments.run_retrieval import _read_done_qids, _read_runs_from_tsv
from experiments.run_generation_baseline import load_runs


SAMPLE = textwrap.dedent(
    """\
    q1\tQ0\td1\t1\t12.500000\tbm25
    q1\tQ0\td2\t2\t11.200000\tbm25
    q1\tQ0\td3\t3\t10.000000\tbm25
    q2\tQ0\td9\t1\t8.500000\tbm25
    q2\tQ0\td1\t2\t8.000000\tbm25
    """
)


@pytest.fixture
def tsv_path(tmp_path):
    p = tmp_path / "run.tsv"
    p.write_text(SAMPLE)
    return p


# --------------------------------------------------------------------------- #
# _read_runs_from_tsv (with scores)
# --------------------------------------------------------------------------- #

def test_read_runs_from_tsv_basic(tsv_path):
    runs, scores = _read_runs_from_tsv(tsv_path)
    assert runs == {
        "q1": ["d1", "d2", "d3"],
        "q2": ["d9", "d1"],
    }
    assert scores["q1"] == pytest.approx([12.5, 11.2, 10.0])
    assert scores["q2"] == pytest.approx([8.5, 8.0])


def test_read_runs_from_tsv_sorts_by_rank(tmp_path):
    """Out-of-order rank lines must still produce a rank-ordered list."""
    p = tmp_path / "run.tsv"
    p.write_text(
        "q1\tQ0\td3\t3\t1.0\tbm25\n"
        "q1\tQ0\td1\t1\t3.0\tbm25\n"
        "q1\tQ0\td2\t2\t2.0\tbm25\n"
    )
    runs, scores = _read_runs_from_tsv(p)
    assert runs["q1"] == ["d1", "d2", "d3"]
    assert scores["q1"] == pytest.approx([3.0, 2.0, 1.0])


def test_read_runs_from_tsv_skips_malformed(tmp_path):
    p = tmp_path / "run.tsv"
    p.write_text(
        "q1\tQ0\td1\t1\t1.0\tbm25\n"
        "this line is junk\n"
        "\n"
        "q1\tQ0\td2\tNOT_A_RANK\t1.0\tbm25\n"
    )
    runs, _ = _read_runs_from_tsv(p)
    assert runs == {"q1": ["d1"]}


# --------------------------------------------------------------------------- #
# load_runs (W3 helper, ranks only)
# --------------------------------------------------------------------------- #

def test_load_runs_basic(tsv_path):
    runs = load_runs(tsv_path)
    assert runs == {
        "q1": ["d1", "d2", "d3"],
        "q2": ["d9", "d1"],
    }


# --------------------------------------------------------------------------- #
# _read_done_qids (resume support)
# --------------------------------------------------------------------------- #

def test_read_done_qids_complete_only(tmp_path):
    p = tmp_path / "run.tsv"
    # q1: complete top-3. q2: only 2 of expected 3 -> incomplete.
    p.write_text(
        "q1\tQ0\td1\t1\t1.0\tbm25\n"
        "q1\tQ0\td2\t2\t1.0\tbm25\n"
        "q1\tQ0\td3\t3\t1.0\tbm25\n"
        "q2\tQ0\td9\t1\t1.0\tbm25\n"
        "q2\tQ0\td8\t2\t1.0\tbm25\n"
    )
    done = _read_done_qids(p, top_k=3)
    assert done == {"q1"}


def test_read_done_qids_missing_file(tmp_path):
    p = tmp_path / "nope.tsv"
    assert _read_done_qids(p, top_k=10) == set()


def test_read_done_qids_treats_top_rank_mismatch_as_incomplete(tmp_path):
    """If a qid has top_k lines but a max rank below top_k, treat as incomplete.

    This guards against duplicate-line corruption where an interrupted writer
    leaves N entries with ranks {1, 2, 3, 1, 2} after a kill-mid-chunk.
    """
    p = tmp_path / "run.tsv"
    # q1: 3 lines but ranks are {1, 1, 2} -> max=2 != top_k=3, incomplete.
    p.write_text(
        "q1\tQ0\td1\t1\t1.0\tbm25\n"
        "q1\tQ0\td1b\t1\t1.0\tbm25\n"
        "q1\tQ0\td2\t2\t1.0\tbm25\n"
    )
    done = _read_done_qids(p, top_k=3)
    assert done == set()
