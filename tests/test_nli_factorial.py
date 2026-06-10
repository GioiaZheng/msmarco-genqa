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
    aggregate_backbones,
    binarize,
    continuous_scores,
    factorial_cell,
    run_factorial,
)


def _p(e: float, n: float, c: float) -> dict[str, float]:
    return {"entailment": e, "neutral": n, "contradiction": c}


def _cell(formula: str, threshold, delta: float, ci_low: float, ci_high: float) -> dict:
    return {
        "formula": formula,
        "threshold": threshold,
        "reverses_sign": ci_high < 0.0,
        "raises_grounding": ci_low > 0.0,
        "bootstrap": {"mean_delta": delta, "ci_low": ci_low, "ci_high": ci_high},
    }


def _summary(backbone: str, cells: list[dict]) -> dict:
    return {"backbone": backbone, "cells": cells}


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


class TestAggregateBackbones:
    def _three_backbones(self, ci_highs):
        # Same single baseline cell across 3 backbones, varying CI upper bound.
        return [
            _summary(name, [_cell("entailment", None, -0.1, -0.2, hi)])
            for name, hi in zip(("a", "b", "c"), ci_highs)
        ]

    def test_robust_reversal_when_all_backbones_reverse(self):
        agg = aggregate_backbones(self._three_backbones([-0.05, -0.03, -0.01]))
        assert agg["n_backbones"] == 3
        assert agg["headline"]["baseline_robust_reversal"] is True
        assert agg["headline"]["baseline_n_reverse"] == 3
        assert agg["n_robust_reversal_cells"] == 1

    def test_not_robust_when_one_backbone_ci_straddles_zero(self):
        agg = aggregate_backbones(self._three_backbones([-0.05, +0.02, -0.01]))
        assert agg["headline"]["baseline_robust_reversal"] is False
        assert agg["headline"]["baseline_n_reverse"] == 2
        assert agg["headline"]["baseline_n_backbones"] == 3

    def test_unanimous_rise_flag(self):
        cells = [_summary(n, [_cell("entailment", None, 0.1, 0.05, 0.2)]) for n in "ab"]
        agg = aggregate_backbones(cells)
        cell = agg["cells"][0]
        assert cell["unanimous_rise"] is True
        assert cell["robust_reversal"] is False

    def test_missing_cell_is_reported_not_dropped(self):
        s1 = _summary("a", [_cell("entailment", None, -0.1, -0.2, -0.05),
                            _cell("calibrated", 0.5, -0.1, -0.2, -0.05)])
        s2 = _summary("b", [_cell("entailment", None, -0.1, -0.2, -0.05)])
        agg = aggregate_backbones([s1, s2])
        calibrated = next(c for c in agg["cells"] if c["formula"] == "calibrated")
        assert calibrated["missing_in"] == ["b"]
        assert calibrated["n_backbones"] == 1
