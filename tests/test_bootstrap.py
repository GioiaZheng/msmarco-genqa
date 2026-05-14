"""Unit tests for ``src.evaluation.bootstrap``.

Covers the pure-Python ``paired_bootstrap_diff``. ``per_query_rouge_l`` and
``per_query_bleu`` are exercised in slow / opt-in tests because they pull
in ``rouge_score`` (lightweight) and NLTK (lightweight, but BLEU's
behaviour at very short answer lengths is brittle and not the point of
these unit tests).
"""

from __future__ import annotations

import pytest

from src.evaluation.bootstrap import paired_bootstrap_diff


class TestPairedBootstrapDiff:
    def test_identical_scores_ci_contains_zero(self):
        # Two identical systems: mean delta is 0 and the percentile CI
        # is degenerate at 0 (every resample averages a vector of zeros).
        a = [0.3, 0.5, 0.7, 0.4, 0.6]
        b = list(a)
        out = paired_bootstrap_diff(a, b, n_resamples=1000, seed=0)
        assert out["mean_delta"] == pytest.approx(0.0)
        assert out["ci_low"] == pytest.approx(0.0)
        assert out["ci_high"] == pytest.approx(0.0)
        # p-value for a degenerate distribution at 0 is 2*1 = 2 -> clipped to 1.
        assert out["p_two_sided"] == pytest.approx(1.0)
        assert out["n"] == 5
        assert out["n_resamples"] == 1000

    def test_large_constant_gap_excludes_zero(self):
        # Every query b > a by 0.3 -> mean_delta = 0.3 exactly, and any
        # nonempty resample gives 0.3. CI collapses on the point estimate.
        a = [0.1, 0.2, 0.3, 0.4, 0.5]
        b = [x + 0.3 for x in a]
        out = paired_bootstrap_diff(a, b, n_resamples=2000, seed=0)
        assert out["mean_delta"] == pytest.approx(0.3)
        assert out["ci_low"] == pytest.approx(0.3)
        assert out["ci_high"] == pytest.approx(0.3)
        # CI excludes zero -> the lower bound is strictly positive.
        assert out["ci_low"] > 0

    def test_noisy_positive_gap_ci_strictly_positive(self):
        # Pair b roughly +0.1 above a but with per-query noise. With
        # n=200 paired observations, the bootstrap CI on the mean delta
        # should be tight and strictly positive.
        rng_a = [(i % 7) / 10.0 for i in range(200)]
        rng_b = [rng_a[i] + 0.1 + ((i % 3) - 1) * 0.02 for i in range(200)]
        out = paired_bootstrap_diff(rng_a, rng_b, n_resamples=2000, seed=42)
        assert out["mean_delta"] > 0
        assert out["ci_low"] > 0
        assert out["ci_high"] > out["ci_low"]
        # All resamples gave a positive delta -> p_two_sided ≈ 0.
        assert out["p_two_sided"] < 0.05

    def test_negative_gap_ci_strictly_negative(self):
        a = [0.6, 0.7, 0.8, 0.9, 0.55, 0.65, 0.75]
        b = [x - 0.2 for x in a]
        out = paired_bootstrap_diff(a, b, n_resamples=1000, seed=1)
        assert out["mean_delta"] == pytest.approx(-0.2)
        assert out["ci_high"] < 0

    def test_seed_reproducibility(self):
        a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        b = [0.2, 0.1, 0.5, 0.3, 0.7, 0.5, 0.8, 0.7, 0.9, 1.0]
        out1 = paired_bootstrap_diff(a, b, n_resamples=500, seed=123)
        out2 = paired_bootstrap_diff(a, b, n_resamples=500, seed=123)
        assert out1 == out2
        out3 = paired_bootstrap_diff(a, b, n_resamples=500, seed=124)
        # Different seed -> at least one of the CI endpoints should differ.
        assert (out1["ci_low"], out1["ci_high"]) != (out3["ci_low"], out3["ci_high"])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            paired_bootstrap_diff([0.1, 0.2], [0.1], n_resamples=10)

    def test_empty_inputs_raises(self):
        with pytest.raises(ValueError):
            paired_bootstrap_diff([], [], n_resamples=10)

    def test_bad_ci_raises(self):
        with pytest.raises(ValueError):
            paired_bootstrap_diff([0.1], [0.2], ci=1.5, n_resamples=10)

    def test_bad_n_resamples_raises(self):
        with pytest.raises(ValueError):
            paired_bootstrap_diff([0.1], [0.2], n_resamples=0)

    def test_returns_means_and_n(self):
        a = [0.1, 0.3, 0.5]
        b = [0.2, 0.4, 0.7]
        out = paired_bootstrap_diff(a, b, n_resamples=200, seed=0)
        assert out["mean_a"] == pytest.approx(0.3)
        assert out["mean_b"] == pytest.approx(13.0 / 30.0)
        assert out["n"] == 3


# --------------------------------------------------------------------------- #
# Per-query scorers — optional, run if rouge_score / nltk are importable.
# --------------------------------------------------------------------------- #


def test_per_query_rouge_l_smoke():
    pytest.importorskip("rouge_score")
    from src.evaluation.bootstrap import per_query_rouge_l

    preds = ["Eiffel Tower", "Tokyo"]
    refs = [["the Eiffel Tower"], ["Paris", "City of Light"]]
    scores = per_query_rouge_l(preds, refs)
    assert len(scores) == 2
    assert scores[0] > 0.5  # near-perfect overlap modulo "the"
    assert scores[1] == pytest.approx(0.0)  # no overlap with either ref


def test_per_query_rouge_l_empty_refs():
    pytest.importorskip("rouge_score")
    from src.evaluation.bootstrap import per_query_rouge_l

    scores = per_query_rouge_l(["anything"], [[]])
    assert scores == [0.0]


def test_per_query_bleu_smoke():
    pytest.importorskip("nltk")
    from src.evaluation.bootstrap import per_query_bleu

    preds = ["the cat sat on the mat", "completely unrelated"]
    refs = [["the cat sat on the mat"], ["the cat sat on the mat"]]
    scores = per_query_bleu(preds, refs)
    assert len(scores) == 2
    assert scores[0] > 0.5  # exact match -> high BLEU
    assert scores[1] < scores[0]


def test_per_query_bleu_empty_pred():
    pytest.importorskip("nltk")
    from src.evaluation.bootstrap import per_query_bleu

    scores = per_query_bleu(["", "  "], [["something"], ["something else"]])
    assert scores == [0.0, 0.0]
