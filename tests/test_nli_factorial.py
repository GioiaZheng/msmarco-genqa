"""Tests for the R5 NLI grounding factorial over cached probabilities.

No model is loaded: every test feeds synthetic 3-class probability dicts,
exactly the shape ``per_query_nli_probs`` returns, so the factorial
arithmetic and the sign-reversal flags are exercised deterministically.
"""

from __future__ import annotations

import pytest

from msmarco_genqa.evaluation.nli_factorial import (
    DEFAULT_FORMULAS,
    DEFAULT_THRESHOLDS,
    NLI_BACKBONES,
    binarize,
    continuous_scores,
    factorial_cell,
    run_factorial,
)


def _p(e: float, n: float, c: float) -> dict[str, float]:
    return {"entailment": e, "neutral": n, "contradiction": c}


class TestBackboneRegistry:
    def test_three_backbones_with_pinned_revisions(self):
        assert len(NLI_BACKBONES) == 3
        for spec in NLI_BACKBONES.values():
            assert spec["model_id"]
            assert len(spec["revision"]) == 40  # full HF commit SHA


class TestScoreHelpers:
    def test_continuous_scores_uses_formula(self):
        probs = [_p(0.6, 0.3, 0.1), _p(0.2, 0.2, 0.6)]
        assert continuous_scores(probs, "entailment") == pytest.approx([0.6, 0.2])
        assert continuous_scores(
            probs, "entailment_minus_contradiction"
        ) == pytest.approx([0.5, -0.4])

    def test_binarize_threshold_is_inclusive(self):
        assert binarize([0.49, 0.5, 0.51], 0.5) == [0.0, 1.0, 1.0]


class TestFactorialCell:
    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            factorial_cell([_p(0.6, 0.3, 0.1)], [], formula="entailment")

    def test_detects_grounding_drop_after_rerank(self):
        # Reranked arm has uniformly lower entailment -> negative delta.
        bm25 = [_p(0.9, 0.05, 0.05) for _ in range(40)]
        rerank = [_p(0.2, 0.1, 0.7) for _ in range(40)]
        cell = factorial_cell(
            bm25, rerank, formula="entailment", n_resamples=500, seed=1
        )
        assert cell["bootstrap"]["mean_delta"] < 0
        assert cell["reverses_sign"] is True
        assert cell["raises_grounding"] is False
        assert cell["score_aggregation"] == "mean_score"
        assert cell["threshold"] is None

    def test_detects_grounding_rise_after_rerank(self):
        bm25 = [_p(0.2, 0.1, 0.7) for _ in range(40)]
        rerank = [_p(0.9, 0.05, 0.05) for _ in range(40)]
        cell = factorial_cell(
            bm25, rerank, formula="entailment", n_resamples=500, seed=1
        )
        assert cell["reverses_sign"] is False
        assert cell["raises_grounding"] is True

    def test_threshold_switches_to_grounded_rate(self):
        bm25 = [_p(0.9, 0.05, 0.05) for _ in range(20)]
        rerank = [_p(0.2, 0.1, 0.7) for _ in range(20)]
        cell = factorial_cell(
            bm25, rerank, formula="entailment", threshold=0.5,
            n_resamples=300, seed=1,
        )
        assert cell["score_aggregation"] == "grounded_rate"
        # All bm25 grounded (0.9>=0.5), no rerank grounded (0.2<0.5).
        assert cell["bootstrap"]["mean_a"] == pytest.approx(1.0)
        assert cell["bootstrap"]["mean_b"] == pytest.approx(0.0)


class TestRunFactorial:
    def test_full_grid_shape(self):
        bm25 = [_p(0.6, 0.3, 0.1) for _ in range(10)]
        rerank = [_p(0.5, 0.3, 0.2) for _ in range(10)]
        cells = run_factorial(bm25, rerank, n_resamples=200, seed=1)
        assert len(cells) == len(DEFAULT_FORMULAS) * len(DEFAULT_THRESHOLDS)
        seen = {(c["formula"], c["threshold"]) for c in cells}
        assert len(seen) == len(cells)  # no duplicate cells
