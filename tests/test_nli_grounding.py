"""Edge-case tests for ``msmarco_genqa.evaluation.nli_grounding``.

The forward-pass path is intentionally NOT exercised here — that would
download a ~140 MB model on every test run. The integration test for
the model output happens in the W7-A audit run itself. These tests
cover the deterministic input-validation / edge-case paths that don't
touch the model.
"""

from __future__ import annotations

import pytest

from msmarco_genqa.evaluation.nli_grounding import (
    NLI_LABELS,
    SCORE_FORMULAS,
    nli_score,
    per_query_nli_entailment,
    per_query_nli_probs,
    resolve_label_indices,
)


class TestInputValidation:
    def test_empty_inputs_returns_empty_list(self):
        assert per_query_nli_entailment([], []) == []
        assert per_query_nli_probs([], []) == []

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            per_query_nli_entailment(["a"], [])
        with pytest.raises(ValueError, match="same length"):
            per_query_nli_entailment([], [["x"]])
        with pytest.raises(ValueError, match="same length"):
            per_query_nli_probs(["a"], [])

    def test_unknown_direction_raises(self):
        with pytest.raises(ValueError, match="unknown direction"):
            per_query_nli_probs(["a"], [["x"]], direction="sideways")


class TestScoreFormulas:
    @pytest.fixture
    def probs(self):
        # entailment=0.6, neutral=0.3, contradiction=0.1
        return {"entailment": 0.6, "neutral": 0.3, "contradiction": 0.1}

    def test_entailment_is_identity(self, probs):
        assert nli_score(probs, "entailment") == pytest.approx(0.6)

    def test_entailment_minus_contradiction(self, probs):
        assert nli_score(probs, "entailment_minus_contradiction") == pytest.approx(0.5)

    def test_calibrated(self, probs):
        # 0.6 - 0.3 - 2*0.1 = 0.1
        assert nli_score(probs, "calibrated") == pytest.approx(0.1)

    def test_all_registered_formulas_callable(self, probs):
        for name in SCORE_FORMULAS:
            assert isinstance(nli_score(probs, name), float)

    def test_unknown_formula_raises(self, probs):
        with pytest.raises(ValueError, match="unknown score formula"):
            nli_score(probs, "bogus")

    def test_contradiction_drags_calibrated_below_entailment_only(self):
        confident_wrong = {"entailment": 0.4, "neutral": 0.1, "contradiction": 0.5}
        assert nli_score(confident_wrong, "calibrated") < nli_score(
            confident_wrong, "entailment"
        )


class TestResolveLabelIndices:
    def test_standard_ordering(self):
        id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
        assert resolve_label_indices(id2label) == {
            "contradiction": 0,
            "neutral": 1,
            "entailment": 2,
        }

    def test_reversed_ordering(self):
        id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        assert resolve_label_indices(id2label) == {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2,
        }

    def test_case_insensitive_and_short_form(self):
        id2label = {0: "ENTAILMENT", 1: "Neutral", 2: "CONTRADICT"}
        resolved = resolve_label_indices(id2label)
        assert set(resolved) == set(NLI_LABELS)

    def test_missing_label_raises(self):
        with pytest.raises(ValueError, match="missing label"):
            resolve_label_indices({0: "entailment", 1: "neutral"})
