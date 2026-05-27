"""Unit tests for ``msmarco_genqa.evaluation.generation``.

Covers the pure-Python helpers (``_normalize``, ``exact_match``, ``token_f1``).
``evaluate_generation`` is exercised by an opt-in slow test that requires
HuggingFace ``evaluate``'s ROUGE / BLEU metric scripts to be cached locally
(or network access to fetch them). Run with ``-m slow`` to include it. The
slow test is *skipped*, not failed, when the metric scripts are unavailable.
"""

from __future__ import annotations

import pytest

from msmarco_genqa.evaluation.generation import (
    _normalize,
    exact_match,
    token_f1,
)


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #

class TestNormalize:
    def test_lowercases(self):
        assert _normalize("Canberra") == "canberra"

    def test_strips_articles(self):
        assert _normalize("the capital of australia") == "capital of australia"
        assert _normalize("a horse, an apple") == "horse apple"

    def test_strips_punctuation(self):
        assert _normalize("Hello, world!") == "hello world"

    def test_collapses_whitespace(self):
        assert _normalize("  multiple   spaces \t here  ") == "multiple spaces here"

    def test_handles_none_and_empty(self):
        assert _normalize(None) == ""  # type: ignore[arg-type]
        assert _normalize("") == ""


# --------------------------------------------------------------------------- #
# exact_match (best-of-N)
# --------------------------------------------------------------------------- #

class TestExactMatch:
    def test_match_after_normalisation(self):
        assert exact_match("The Eiffel Tower", ["eiffel tower"]) == 1.0

    def test_no_match(self):
        assert exact_match("Paris", ["Tokyo"]) == 0.0

    def test_best_of_n(self):
        # First ref doesn't match, second does
        assert exact_match("Paris, France", ["Tokyo", "paris france"]) == 1.0

    def test_empty_references(self):
        assert exact_match("Paris", []) == 0.0


# --------------------------------------------------------------------------- #
# token_f1 (best-of-N)
# --------------------------------------------------------------------------- #

class TestTokenF1:
    def test_perfect_overlap(self):
        assert token_f1("eiffel tower", ["eiffel tower"]) == pytest.approx(1.0)

    def test_normalisation_applied(self):
        # The article "the" is stripped; rest matches perfectly.
        assert token_f1("the Eiffel Tower", ["Eiffel Tower"]) == pytest.approx(1.0)

    def test_partial_overlap(self):
        # Use non-article tokens — _normalize strips "a"/"an"/"the".
        # pred = {x, y, z}; ref = {x, y}; common=2; P=2/3, R=1.0; F1 = 0.8
        assert token_f1("x y z", ["x y"]) == pytest.approx(0.8)

    def test_no_overlap(self):
        assert token_f1("foo bar", ["baz qux"]) == 0.0

    def test_best_of_n_keeps_max(self):
        # First ref overlaps poorly, second perfectly
        assert token_f1("eiffel tower", ["paris", "eiffel tower"]) == pytest.approx(1.0)

    def test_empty_prediction_or_refs(self):
        assert token_f1("", ["something"]) == 0.0
        assert token_f1("something", []) == 0.0


# --------------------------------------------------------------------------- #
# evaluate_generation (slow — requires HF metric scripts)
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_evaluate_generation_smoke():
    """End-to-end sanity check: 2 preds, multi-ref, returns the expected keys.

    Requires HF ``evaluate``'s ROUGE/BLEU metric scripts (downloaded on first
    use, cached afterwards). When the cache is missing AND there's no network
    — e.g. a sandboxed CI runner — ``evaluate.load("rouge")`` raises.
    We skip in that case, never fail, since the offline status is a property
    of the environment, not the code under test.
    """
    pytest.importorskip("evaluate")
    pytest.importorskip("rouge_score")
    from msmarco_genqa.evaluation.generation import evaluate_generation

    preds = ["Canberra is the capital", "Paris"]
    refs = [["Canberra"], ["Paris", "City of Light"]]
    try:
        out = evaluate_generation(preds, refs)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"HF metric scripts unavailable: {exc!r}")

    for key in ("rouge-l", "bleu", "exact-match", "token-f1", "n_predictions"):
        assert key in out, f"missing {key} in {out}"
    # exact_match: pred 0 ('Canberra is the capital') vs 'Canberra' -> 0;
    #              pred 1 'Paris' vs 'Paris'/'City of Light' -> 1; avg = 0.5
    assert out["exact-match"] == pytest.approx(0.5)
    assert out["n_predictions"] == 2
