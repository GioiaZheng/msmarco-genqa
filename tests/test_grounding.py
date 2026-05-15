"""Unit tests for ``src.evaluation.grounding``.

The metrics are pure functions over strings with a fixed regex
tokeniser and a frozen stopword set, so every test is deterministic
and runs in milliseconds (no model load).
"""

from __future__ import annotations

import pytest

from src.evaluation.grounding import (
    ENGLISH_STOPWORDS,
    content_tokens,
    is_vacuously_grounded_lex,
    is_vacuously_grounded_ngram,
    lexical_grounding,
    ngram_grounding,
    tokenize,
)


class TestTokenize:
    def test_basic_lowercase_and_split(self):
        assert tokenize("Hello, World!") == ["hello", "world"]

    def test_punctuation_dropped(self):
        assert tokenize("foo. bar; baz!") == ["foo", "bar", "baz"]

    def test_digits_kept(self):
        assert tokenize("the year 2024 was hot") == ["the", "year", "2024", "was", "hot"]

    def test_apostrophes_kept_as_part_of_token(self):
        # "it's" stays as one token; the contraction is then matched against
        # ENGLISH_STOPWORDS, which lists "n't" and "s" but not "it's" itself.
        assert tokenize("it's a cat") == ["it's", "a", "cat"]

    def test_non_ascii_treated_as_break(self):
        # "café" -> 'caf' + 'é' would split, but 'é' is not [a-z0-9'], so
        # the regex skips it entirely. The two halves are returned separately.
        assert tokenize("café au lait") == ["caf", "au", "lait"]

    def test_empty_string(self):
        assert tokenize("") == []


class TestContentTokens:
    def test_stopwords_removed(self):
        # 'the', 'of', 'a' are in the stopword set
        assert content_tokens("the king of a country") == ["king", "country"]

    def test_order_preserved(self):
        assert content_tokens("alpha beta gamma") == ["alpha", "beta", "gamma"]

    def test_duplicates_kept(self):
        # Lexical metric deduplicates internally; we keep order/duplicates here.
        assert content_tokens("dog dog cat dog") == ["dog", "dog", "cat", "dog"]

    def test_all_stopwords_returns_empty(self):
        assert content_tokens("the of an and or") == []


class TestStopwordSet:
    def test_is_frozenset(self):
        assert isinstance(ENGLISH_STOPWORDS, frozenset)

    def test_contains_core_articles(self):
        for w in ("a", "an", "the"):
            assert w in ENGLISH_STOPWORDS

    def test_excludes_high_signal_words(self):
        # Content words must NOT be in the stopword set, otherwise the
        # lexical metric would silently strip real answer content.
        for w in ("capital", "australia", "canberra", "answer", "year"):
            assert w not in ENGLISH_STOPWORDS


class TestLexicalGrounding:
    def test_full_overlap_returns_one(self):
        pred = "Canberra is the capital city of Australia"
        passages = ["Canberra is the capital city of Australia"]
        assert lexical_grounding(pred, passages) == pytest.approx(1.0)

    def test_zero_overlap_returns_zero(self):
        pred = "elephants are mammals"
        passages = ["nothing relevant to the prediction at all here"]
        assert lexical_grounding(pred, passages) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # Content tokens of prediction: {capital, city, australia} (3 unique)
        # 2 of 3 appear in the passage -> 2/3.
        pred = "the capital city of Australia"
        passages = ["Australia is a big country; its capital is hidden"]
        assert lexical_grounding(pred, passages) == pytest.approx(2 / 3)

    def test_stopwords_in_prediction_do_not_count(self):
        # Without stopword stripping this would be 5/6; with stopword
        # stripping the only content token in the pred is 'cat', and it's
        # in the passage -> 1.0.
        pred = "the cat is on the mat"
        passages = ["a cat sat there"]
        # content_tokens(pred) = ['cat', 'mat']; 'cat' is in passage tokens
        # = {'a','cat','sat','there'}; 'mat' is not -> 1/2.
        assert lexical_grounding(pred, passages) == pytest.approx(0.5)

    def test_case_insensitive(self):
        assert (
            lexical_grounding("CANBERRA", ["canberra is the capital"])
            == pytest.approx(1.0)
        )

    def test_passage_union_is_set_union(self):
        # Word appearing in passage[1] but not passage[0] still counts.
        pred = "canberra"
        passages = ["other words only", "canberra appears here"]
        assert lexical_grounding(pred, passages) == pytest.approx(1.0)

    def test_empty_prediction_is_vacuously_one(self):
        assert lexical_grounding("", ["anything"]) == pytest.approx(1.0)

    def test_all_stopword_prediction_is_vacuously_one(self):
        assert lexical_grounding("the of an", ["foo bar"]) == pytest.approx(1.0)

    def test_empty_passages_with_content_is_zero(self):
        assert lexical_grounding("canberra", []) == pytest.approx(0.0)
        assert lexical_grounding("canberra", [""]) == pytest.approx(0.0)

    def test_deduplicates_pred_content_tokens(self):
        # 'cat' counted once even though it appears 3x in prediction.
        pred = "cat cat cat dog"
        passages = ["cat fish"]
        # Unique content tokens: {cat, dog}; 'cat' in passage, 'dog' not -> 1/2.
        assert lexical_grounding(pred, passages) == pytest.approx(0.5)


class TestNgramGrounding:
    def test_full_3gram_overlap(self):
        pred = "the capital of australia"
        # All 3-grams: ('the','capital','of'), ('capital','of','australia') —
        # both present in this passage.
        passages = ["xxx the capital of australia is canberra"]
        assert ngram_grounding(pred, passages, n=3) == pytest.approx(1.0)

    def test_zero_3gram_overlap(self):
        pred = "elephants are mammals from africa"
        passages = ["nothing matching here exists at all really"]
        # No 3-gram from prediction appears in the passage.
        assert ngram_grounding(pred, passages, n=3) == pytest.approx(0.0)

    def test_partial_3gram_overlap(self):
        # Pred has 3 3-grams; 1 of them is in the passage.
        pred = "capital of australia plus extra"
        passages = ["xxx the capital of australia today"]
        # Pred 3-grams: ('capital','of','australia'),
        # ('of','australia','plus'), ('australia','plus','extra')
        # Only the first is in the passage.
        assert ngram_grounding(pred, passages, n=3) == pytest.approx(1 / 3)

    def test_ngrams_do_not_cross_passage_boundary(self):
        # 'a b c' is split across two passages; if boundary-crossing were
        # allowed the 3-gram ('a','b','c') would match. Our spec says no:
        # passages are independent prompt segments.
        pred = "a b c"
        passages = ["a b", "c d"]
        # Pred 3-grams: ('a','b','c'); neither passage alone contains it.
        assert ngram_grounding(pred, passages, n=3) == pytest.approx(0.0)

    def test_unigram_grounding_equivalent_to_token_membership(self):
        pred = "canberra capital city"
        passages = ["canberra is the capital city"]
        # All 3 unigrams of prediction appear in passage -> 1.0
        assert ngram_grounding(pred, passages, n=1) == pytest.approx(1.0)

    def test_short_prediction_is_vacuously_one(self):
        # 2 tokens < n=3 -> vacuous
        assert ngram_grounding("hello world", ["foo bar"], n=3) == pytest.approx(1.0)
        # 1 token, n=3 -> vacuous
        assert ngram_grounding("ok", ["foo bar baz"], n=3) == pytest.approx(1.0)
        # empty prediction -> vacuous
        assert ngram_grounding("", ["foo bar baz"], n=3) == pytest.approx(1.0)

    def test_empty_passages_with_long_pred_is_zero(self):
        assert ngram_grounding("a b c d e", [], n=3) == pytest.approx(0.0)
        assert ngram_grounding("a b c d e", ["", ""], n=3) == pytest.approx(0.0)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            ngram_grounding("a b c", ["a b c"], n=0)


class TestVacuousFlags:
    def test_lex_vacuous_flag(self):
        assert is_vacuously_grounded_lex("") is True
        assert is_vacuously_grounded_lex("the of") is True
        assert is_vacuously_grounded_lex("capital") is False

    def test_ngram_vacuous_flag_default_n(self):
        assert is_vacuously_grounded_ngram("") is True
        assert is_vacuously_grounded_ngram("a b") is True  # 2 tokens < 3
        assert is_vacuously_grounded_ngram("a b c") is False

    def test_ngram_vacuous_flag_custom_n(self):
        # With n=5, "a b c d" (4 tokens) is vacuous; "a b c d e" is not.
        assert is_vacuously_grounded_ngram("a b c d", n=5) is True
        assert is_vacuously_grounded_ngram("a b c d e", n=5) is False
