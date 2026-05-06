"""Generation metrics: ROUGE-L, BLEU, Exact Match, Token F1.

All four metrics are computed in **best-of-N reference** mode:

- ROUGE-L: HF ``evaluate``'s ``rouge`` accepts ``references: List[List[str]]``
  and uses ``rouge_score.score_multi`` internally, which scores against each
  reference and keeps the best.
- BLEU: HF ``evaluate``'s ``bleu`` natively supports ``List[List[str]]`` —
  the underlying NLTK BLEU uses the closest-length / max-overlap reference
  per prediction.
- Exact Match: SQuAD-style normalisation, ``any`` over references.
- Token F1: SQuAD-style token overlap, ``max`` over references.

This is the standard MS MARCO QA / SQuAD evaluation convention. Earlier
versions of this file scored ROUGE/BLEU against only the first reference,
which systematically under-rated paraphrases that match a non-first
human-written answer.
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

    # Best-of-N references: pass list-of-list directly to both metrics.
    # Predictions whose reference list is empty are dropped (they would just
    # contribute zero in any case, but HF metrics can crash on empty inner
    # lists). EM / Token-F1 below handle the empty-refs case explicitly.
    paired = [(p, list(r)) for p, r in zip(predictions, references) if r]
    if paired:
        scoring_preds = [p for p, _ in paired]
        scoring_refs = [r for _, r in paired]
        rouge_l = rouge.compute(
            predictions=scoring_preds,
            references=scoring_refs,
        )["rougeL"]
        bleu_score = bleu.compute(
            predictions=scoring_preds,
            references=scoring_refs,
        )["bleu"]
    else:
        rouge_l = 0.0
        bleu_score = 0.0

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
