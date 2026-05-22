"""Question-form classification for MS MARCO dev/small queries.

A *complement* to MS MARCO QA v2.1's native ``query_type`` field, which
classifies queries by *answer type* (DESCRIPTION / NUMERIC / ENTITY /
PERSON / LOCATION). This module instead classifies by *question form* —
the wh-word (or absence of one) that opens the query — which is
orthogonal: a "PERSON" answer can come from a ``who`` question or a
keyword query ("ceo of tesla"); a "DESCRIPTION" answer from ``what``,
``how``, or a keyword query ("define photosynthesis").

The classifier is a deterministic rule cascade on the lowercased
first-token. First match wins; categories are:

- ``who``    — opens with ``who`` / ``whose`` / ``whom``.
- ``what``   — opens with ``what`` / ``whats`` / ``what's``.
- ``when``   — opens with ``when``.
- ``where``  — opens with ``where``.
- ``why``    — opens with ``why``.
- ``how``    — opens with ``how`` / ``hows`` / ``how's`` (covers
               "how much / how many / how to / ..." too).
- ``which``  — opens with ``which``.
- ``yes_no`` — opens with an auxiliary / modal verb that signals a
               polar question (``is``, ``are``, ``was``, ``were``,
               ``do``, ``does``, ``did``, ``can``, ``could``, ``will``,
               ``would``, ``should``, ``has``, ``have``, ``had``,
               ``am``, ``may``, ``might``, ``must``, ``shall``).
- ``other``  — anything else. Empirically ~28 % of MS MARCO dev/small,
               dominated by keyword-style queries ("define X",
               "average Y", "X symptoms").

Empty / whitespace-only queries return ``other``. The classifier is
pure Python, no model load, microseconds per query.
"""

from __future__ import annotations

from typing import Iterable

WH_WHO = frozenset({"who", "whose", "whom"})
WH_WHAT = frozenset({"what", "whats", "what's"})
WH_WHEN = frozenset({"when"})
WH_WHERE = frozenset({"where"})
WH_WHY = frozenset({"why"})
WH_HOW = frozenset({"how", "hows", "how's"})
WH_WHICH = frozenset({"which"})
YES_NO_AUX = frozenset(
    {
        "is", "are", "was", "were",
        "do", "does", "did",
        "can", "could",
        "will", "would",
        "should", "shall",
        "has", "have", "had",
        "am",
        "may", "might", "must",
    }
)

QUESTION_FORM_CATEGORIES: tuple[str, ...] = (
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "which",
    "yes_no",
    "other",
)


def _first_token(query: str) -> str:
    """Return the lowercased first whitespace-delimited token of ``query``.

    Returns an empty string for empty / whitespace-only input.
    """
    if not query:
        return ""
    stripped = query.strip().lower()
    if not stripped:
        return ""
    return stripped.split(None, 1)[0]


def classify_question_form(query: str) -> str:
    """Classify a single query into one of ``QUESTION_FORM_CATEGORIES``.

    Deterministic rule cascade on the lowercased first token. See the
    module docstring for category definitions.
    """
    tok = _first_token(query)
    if not tok:
        return "other"
    if tok in WH_WHO:
        return "who"
    if tok in WH_WHAT:
        return "what"
    if tok in WH_WHEN:
        return "when"
    if tok in WH_WHERE:
        return "where"
    if tok in WH_WHY:
        return "why"
    if tok in WH_HOW:
        return "how"
    if tok in WH_WHICH:
        return "which"
    if tok in YES_NO_AUX:
        return "yes_no"
    return "other"


def classify_many(queries: Iterable[str]) -> list[str]:
    """Vectorised wrapper around ``classify_question_form``."""
    return [classify_question_form(q) for q in queries]
