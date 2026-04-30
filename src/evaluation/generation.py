"""Generation metrics: ROUGE-L, BLEU, Exact Match, Token F1.

ROUGE-L and BLEU are computed via the ``evaluate`` library against the *first*
reference (HuggingFace's APIs do not natively support best-of-N references).
Exact Match and Token F1 use the SQuAD-style "best of references" rule.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Sequence


_PUNCT = set(string.punctuation)


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokens(text: str) -> list[str]:
    return _normalize(text).split()


def exact_match(prediction: str, references: Sequence[str]) -> float:
    if not references:
        return 0.0
    p = _normalize(prediction)
    return float(any(p == _normalize(r) for r in references))


def token_f1(prediction: str, references: Sequence[str]) -> float:
    pred_toks = _tokens(prediction)
    if not pred_toks or not references:
        return 0.0
    best = 0.0
    for ref in references:
        ref_toks = _tokens(ref)
        if not ref_toks:
            continue
        common = Counter(pred_toks) & Counter(ref_toks)
        n_same = sum(common.values())
        if n_same == 0:
            continue
        precision = n_same / len(pred_toks)
        recall = n_same / len(ref_toks)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best:
            best = f1
    return best


def evaluate_generation(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> dict[str, float]:
    """Compute ROUGE-L, BLEU, Exact Match, Token F1.

    Parameters
    ----------
    predictions :
        One generated answer per example.
    references :
        One *list* of reference answers per example. The list may contain
        multiple human-written answers.
    """
    import evaluate

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    primary_refs = [(refs[0] if refs else "") for refs in references]
    rouge_l = rouge.compute(
        predictions=list(predictions),
        references=primary_refs,
    )["rougeL"]
    bleu_score = bleu.compute(
        predictions=list(predictions),
        references=[[r] for r in primary_refs],
    )["bleu"]

    n = len(predictions)
    em = sum(exact_match(p, refs) for p, refs in zip(predictions, references)) / max(n, 1)
    f1 = sum(token_f1(p, refs) for p, refs in zip(predictions, references)) / max(n, 1)

    return {
        "rouge-l": float(rouge_l),
        "bleu": float(bleu_score),
        "exact-match": float(em),
        "token-f1": float(f1),
        "n_predictions": n,
    }
