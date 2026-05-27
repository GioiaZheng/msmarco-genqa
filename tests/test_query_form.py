"""Unit tests for ``msmarco_genqa.evaluation.query_form``.

The classifier is a pure-Python rule cascade — every test is
deterministic, microseconds, no model load.
"""

from __future__ import annotations

import pytest

from msmarco_genqa.evaluation.query_form import (
    QUESTION_FORM_CATEGORIES,
    classify_many,
    classify_question_form,
)


class TestWhWords:
    @pytest.mark.parametrize(
        "query,expected",
        [
            ("who invented the telephone", "who"),
            ("Whose book is this", "who"),
            ("whom did he see", "who"),
            ("what is insulin", "what"),
            ("whats the largest planet", "what"),
            ("what's tom cruise's birth name", "what"),
            ("When was Lincoln shot", "when"),
            ("where does real insulin come from", "where"),
            ("why is the sky blue", "why"),
            ("how many bones in the human body", "how"),
            ("How's the weather in Tokyo", "how"),
            ("which country has the highest gdp", "which"),
        ],
    )
    def test_wh_word_classification(self, query: str, expected: str) -> None:
        assert classify_question_form(query) == expected


class TestYesNoAuxiliaries:
    @pytest.mark.parametrize(
        "aux",
        [
            "is", "are", "was", "were",
            "do", "does", "did",
            "can", "could",
            "will", "would",
            "should", "shall",
            "has", "have", "had",
            "am", "may", "might", "must",
        ],
    )
    def test_each_auxiliary_routes_to_yes_no(self, aux: str) -> None:
        assert classify_question_form(f"{aux} water boils at 100 degrees") == "yes_no"

    def test_case_insensitive(self) -> None:
        assert classify_question_form("IS water wet") == "yes_no"


class TestOtherBucket:
    @pytest.mark.parametrize(
        "query",
        [
            "define: precipitous delivery",
            "average rainfall in seattle",
            "cortana what is the apocalypse",
            "weather in whitefish montana",
            "types of insulin",
            "the largest desert in the world",
        ],
    )
    def test_keyword_style_falls_into_other(self, query: str) -> None:
        assert classify_question_form(query) == "other"


class TestEdgeCases:
    def test_empty_string(self) -> None:
        assert classify_question_form("") == "other"

    def test_whitespace_only(self) -> None:
        assert classify_question_form("   \t\n  ") == "other"

    def test_leading_whitespace_does_not_break_match(self) -> None:
        assert classify_question_form("   who are you") == "who"

    def test_trailing_punctuation_ignored_at_dispatch(self) -> None:
        # The classifier looks at the first whitespace token; punctuation
        # attached to it WOULD prevent a match, which is the intended
        # behaviour for the rare "what?" style fragment.
        assert classify_question_form("what,") == "other"

    def test_what_with_apostrophe_variants(self) -> None:
        assert classify_question_form("whats up") == "what"
        assert classify_question_form("what's the time") == "what"


class TestBatchAPI:
    def test_classify_many_preserves_order(self) -> None:
        qs = ["who is bob", "average price of oil", "why is the sky blue"]
        assert classify_many(qs) == ["who", "other", "why"]

    def test_all_categories_are_reachable(self) -> None:
        samples = {
            "who": "who is bob",
            "what": "what is x",
            "when": "when did x happen",
            "where": "where is paris",
            "why": "why is x",
            "how": "how to x",
            "which": "which is bigger",
            "yes_no": "is water wet",
            "other": "average rainfall",
        }
        assert set(QUESTION_FORM_CATEGORIES) == set(samples.keys())
        for expected, q in samples.items():
            assert classify_question_form(q) == expected
