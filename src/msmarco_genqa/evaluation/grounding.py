"""Lexical and n-gram grounding of RAG generator predictions to their
prompt passages.

These are *grounding* metrics, not faithfulness in the semantic sense.
A heavy paraphrase that is perfectly faithful will score low on both;
a verbatim copy that is also wrong (e.g. extracted from a distractor
passage) will score high. The metrics answer "is the model's output
literally derived from the prompt context" — which is the dominant
question for small extractive Seq2Seq generators like T5-small on the
``question: ... context: ...`` prompt shape used by the generation baseline.

Two metrics:

- ``lexical_grounding(prediction, passages)``
    Fraction of *unique content tokens* in the prediction (lowercase,
    non-stopword) that appear anywhere in the union of passage texts.
    1.0 means every distinct content word is present somewhere in the
    prompt; 0.0 means none are.

- ``ngram_grounding(prediction, passages, n=3)``
    Fraction of the prediction's contiguous n-grams (default 3-grams,
    on the lowercased token sequence with stopwords retained) that
    appear as a contiguous span in at least one passage. Captures
    phrase-level extractiveness that the lexical metric cannot.

Edge cases (documented and stable):

- Empty prediction (after content-token filtering) → ``lexical_grounding``
  returns ``1.0`` (vacuously grounded — nothing to fail).
- Prediction with fewer than ``n`` tokens → ``ngram_grounding`` returns
  ``1.0`` (no n-grams to score). The audit driver separately reports
  the count of such predictions per arm so this convention cannot
  silently inflate the headline number.
- Empty / all-empty passage list → both metrics return ``0.0`` whenever
  the prediction has non-vacuous content to ground.

All tokenisation is deterministic ASCII regex; no model load, no NLTK
or scikit-learn dependency. CPU-only, milliseconds per query.
"""

from __future__ import annotations

import re
from typing import Sequence


# --------------------------------------------------------------------------- #
# Tokenisation + stopwords
# --------------------------------------------------------------------------- #

# Lowercased alphanumerics-plus-apostrophe spans. MS MARCO answers are ASCII
# in practice; non-ASCII (rare) is treated as a token break, which is the
# right behaviour for "does this content word appear in the passage".
_TOKEN_RE = re.compile(r"[a-z0-9']+")


# Compact, deterministic English stopword list. Intentionally vetted to ~85
# items rather than imported from sklearn / NLTK — keeps the metric a pure
# function with no dependency surface and no version-skew risk between
# library releases. Matches roughly the union of "common closed-class
# words" + "auxiliaries / modals" + "high-frequency prepositions / pronouns".
# Apostrophe-bearing contractions are tokenised as single tokens by
# ``_TOKEN_RE`` so both surface forms ("isn't") and the lemma ("is") are
# excluded from content.
ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    {
        # articles / determiners
        "a", "an", "the", "this", "that", "these", "those",
        "any", "some", "all", "each", "every", "no", "none",
        # personal / possessive pronouns
        "i", "me", "my", "mine", "myself",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "we", "us", "our", "ours", "ourselves",
        "they", "them", "their", "theirs", "themselves",
        # be / have / do
        "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having",
        "do", "does", "did", "doing",
        # modals / auxiliaries
        "will", "would", "shall", "should", "can", "could",
        "may", "might", "must", "ought",
        # negations & contractions handled at the lemma level
        "not", "n't",
        # high-frequency prepositions / conjunctions / adverbs
        "of", "in", "on", "at", "by", "for", "with", "to", "from",
        "into", "onto", "out", "over", "under", "up", "down",
        "and", "or", "but", "nor", "so", "yet", "as",
        "if", "then", "than", "because", "while", "until",
        "about", "above", "below", "between", "through",
        # interrogatives / relatives (intentionally included — the *question*
        # words are not the answer's content)
        "what", "which", "who", "whom", "whose", "where", "when",
        "why", "how",
        # generic verbs of high frequency that rarely carry answer content
        "there", "here", "very", "just", "also", "only", "more", "most",
        # explicit short forms that survive the regex
        "s", "t", "d", "ll", "re", "ve", "m",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase, regex-tokenise; deterministic.

    Returns a list of token strings — *order preserved* — so the same
    function feeds both the lexical (set-based) and n-gram (sequence-
    based) metrics without duplicate work.
    """
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokenise then drop stopwords. Order preserved; duplicates kept
    (the lexical metric deduplicates internally; n-gram callers don't
    use this helper)."""
    return [t for t in tokenize(text) if t not in ENGLISH_STOPWORDS]


def _passage_union_tokens(passages: Sequence[str]) -> set[str]:
    """Union of all unique tokens (case-folded) across all passages."""
    out: set[str] = set()
    for p in passages:
        if not p:
            continue
        out.update(tokenize(p))
    return out


def _passage_union_ngrams(passages: Sequence[str], n: int) -> set[tuple[str, ...]]:
    """Union of all unique contiguous n-grams across all passages.

    Each passage's n-grams are computed independently then unioned, so
    a 3-gram crossing the boundary between passage[i] and passage[i+1]
    is *not* present in the union (the passages are independent
    documents in the prompt, even though the prompt concatenates them
    with a space)."""
    out: set[tuple[str, ...]] = set()
    for p in passages:
        if not p:
            continue
        toks = tokenize(p)
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            out.add(tuple(toks[i : i + n]))
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def lexical_grounding(prediction: str, passages: Sequence[str]) -> float:
    """Fraction of unique content tokens from ``prediction`` that appear
    anywhere in the union of ``passages`` (case-folded, stopwords removed
    from the prediction).

    Edge cases
    ----------
    - Prediction has no content tokens (all-stopword or empty after
      tokenisation) → returns ``1.0``. Vacuously grounded; the audit
      script reports the rate of such predictions separately so this
      cannot silently inflate the headline number.
    - Prediction has content tokens but ``passages`` is empty or
      all-empty → returns ``0.0``.
    """
    pred_content = set(content_tokens(prediction))
    if not pred_content:
        return 1.0
    passage_toks = _passage_union_tokens(passages)
    if not passage_toks:
        return 0.0
    hit = sum(1 for t in pred_content if t in passage_toks)
    return hit / len(pred_content)


def ngram_grounding(
    prediction: str,
    passages: Sequence[str],
    n: int = 3,
) -> float:
    """Fraction of the prediction's contiguous n-grams that appear as a
    contiguous span in at least one passage (case-folded; stopwords
    retained, since n-gram order is the signal).

    Edge cases
    ----------
    - Prediction has fewer than ``n`` tokens after tokenisation →
      returns ``1.0`` (no n-grams to score).
    - Prediction has n-grams but ``passages`` is empty or all-empty →
      returns ``0.0``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    pred_toks = tokenize(prediction)
    if len(pred_toks) < n:
        return 1.0
    pred_ngrams: set[tuple[str, ...]] = set()
    for i in range(len(pred_toks) - n + 1):
        pred_ngrams.add(tuple(pred_toks[i : i + n]))
    if not pred_ngrams:
        return 1.0
    passage_ngrams = _passage_union_ngrams(passages, n=n)
    if not passage_ngrams:
        return 0.0
    hit = sum(1 for ng in pred_ngrams if ng in passage_ngrams)
    return hit / len(pred_ngrams)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

def is_vacuously_grounded_lex(prediction: str) -> bool:
    """True iff ``prediction`` has no content tokens. Used by the audit
    script to report the rate of vacuously-1.0 lexical scores per arm."""
    return not content_tokens(prediction)


def is_vacuously_grounded_ngram(prediction: str, n: int = 3) -> bool:
    """True iff ``prediction`` has fewer than ``n`` tokens. Used by the
    audit script to report the rate of vacuously-1.0 n-gram scores per
    arm."""
    return len(tokenize(prediction)) < n
