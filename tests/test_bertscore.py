"""Unit tests for ``src.evaluation.bertscore.per_query_bertscore_f1``.

``bert_score`` is mocked so the test runs in milliseconds without a
model download. What matters for unit-test coverage is the paired-API
contract: same length in, same length out, max-over-references for
multi-ref queries, ``0.0`` for empty pred/refs, and correct forwarding
of CLI knobs to ``bert_score.score``.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from src.evaluation.bertscore import per_query_bertscore_f1


class _FakeTensor:
    """Stand-in for the F-score tensor that ``bert_score.score`` returns."""

    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


def _install_fake_bert_score(monkeypatch, f_values):
    """Make ``import bert_score`` resolve to a mock module.

    Returns the mock so tests can assert on the recorded call.
    """
    fake = types.ModuleType("bert_score")
    score_mock = MagicMock()
    # bert_score.score returns (P, R, F); we only consume F.
    score_mock.return_value = (None, None, _FakeTensor(f_values))
    fake.score = score_mock
    monkeypatch.setitem(sys.modules, "bert_score", fake)
    return score_mock


def test_per_query_bertscore_single_reference(monkeypatch):
    score_mock = _install_fake_bert_score(monkeypatch, [0.91, 0.82])

    out = per_query_bertscore_f1(
        ["pred one", "pred two"],
        [["ref one"], ["ref two"]],
    )

    assert out == [0.91, 0.82]
    score_mock.assert_called_once()
    kwargs = score_mock.call_args.kwargs
    assert kwargs["cands"] == ["pred one", "pred two"]
    assert kwargs["refs"] == ["ref one", "ref two"]
    assert kwargs["model_type"] == "distilbert-base-uncased"
    assert kwargs["rescale_with_baseline"] is True


def test_per_query_bertscore_takes_max_over_references(monkeypatch):
    # query 0: two refs -> max(0.4, 0.9) = 0.9
    # query 1: one ref  -> 0.5
    _install_fake_bert_score(monkeypatch, [0.4, 0.9, 0.5])

    out = per_query_bertscore_f1(
        ["pred zero", "pred one"],
        [["ref a", "ref b"], ["ref c"]],
    )

    assert out == [0.9, 0.5]


def test_empty_prediction_or_references_scores_zero(monkeypatch):
    # Only the third query (index 2) has both a non-empty pred and a
    # non-empty ref, so bert_score should see exactly one (cand, ref)
    # pair.
    score_mock = _install_fake_bert_score(monkeypatch, [0.77])

    out = per_query_bertscore_f1(
        ["", "   ", "real pred", "another"],
        [["ref a"], ["ref b"], ["ref c"], []],
    )

    assert out == [0.0, 0.0, 0.77, 0.0]
    kwargs = score_mock.call_args.kwargs
    assert kwargs["cands"] == ["real pred"]
    assert kwargs["refs"] == ["ref c"]


def test_returns_empty_for_empty_input(monkeypatch):
    # bert_score must NOT be called when there is nothing to score.
    score_mock = _install_fake_bert_score(monkeypatch, [])
    assert per_query_bertscore_f1([], []) == []
    score_mock.assert_not_called()


def test_all_empty_returns_zeros_without_calling_bertscore(monkeypatch):
    score_mock = _install_fake_bert_score(monkeypatch, [])
    out = per_query_bertscore_f1(["", "  "], [[], [""]])
    assert out == [0.0, 0.0]
    score_mock.assert_not_called()


def test_length_mismatch_raises(monkeypatch):
    _install_fake_bert_score(monkeypatch, [])
    with pytest.raises(ValueError):
        per_query_bertscore_f1(["a", "b"], [["only one ref list"]])


def test_kwargs_forwarded(monkeypatch):
    score_mock = _install_fake_bert_score(monkeypatch, [0.7])
    per_query_bertscore_f1(
        ["p"],
        [["r"]],
        model_type="bert-base-uncased",
        lang="en",
        rescale_with_baseline=False,
        batch_size=16,
        device="cpu",
        verbose=True,
    )
    kwargs = score_mock.call_args.kwargs
    assert kwargs["model_type"] == "bert-base-uncased"
    assert kwargs["rescale_with_baseline"] is False
    assert kwargs["batch_size"] == 16
    assert kwargs["device"] == "cpu"
    assert kwargs["verbose"] is True
