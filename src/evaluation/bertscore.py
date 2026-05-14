"""DistilBERT-based BERTScore per-query scorer (low-cost semantic proxy).

This is intentionally *not* the canonical ``roberta-large`` BERTScore
(which is the choice the BERTScore paper recommends for English). On
this project's CPU-only setup, scoring the full 6 980-query dev/small
twice with ``roberta-large`` would take several hours; ``distilbert-
base-uncased`` is ~3x faster and fits a 3 000-paired-qid semantic
sanity check inside a single working session.

The function is intentionally narrow: it scores per query (with the
multi-reference best-of-N convention used by the other scorers in
``src.evaluation.bootstrap``) and returns a plain ``list[float]`` so
the result can be fed directly into ``paired_bootstrap_diff``.

What ``rescale_with_baseline`` does
-----------------------------------
``bert_score`` provides per-model baselines so the typically narrow
``[0.85, 0.95]`` raw score band is rescaled to roughly ``[0, 1]`` and
becomes easier to read. The rescale is a *linear* transform of the raw
score, so it does not flip Δ signs or change the paired bootstrap's
inferential conclusions; it only affects readability of the absolute
levels. Default ``True`` here for consistency with how the project
reports the other generation metrics.

What this is *not*
------------------
A canonical, citation-grade BERTScore for this benchmark. For paired
direction-of-effect and rough magnitude on this corpus the DistilBERT
proxy is fine; cross-paper comparisons should re-run with the
``roberta-large`` default.
"""

from __future__ import annotations

from typing import Sequence


def per_query_bertscore_f1(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    *,
    model_type: str = "distilbert-base-uncased",
    lang: str = "en",
    rescale_with_baseline: bool = True,
    batch_size: int = 32,
    device: str | None = None,
    verbose: bool = False,
) -> list[float]:
    """Per-example BERTScore F1 with best-of-N reference scoring.

    Multi-reference handling mirrors ``per_query_rouge_l`` /
    ``per_query_bleu``: each reference is scored separately and the
    maximum F1 is kept. Examples with an empty reference list (or an
    all-whitespace prediction) score ``0.0`` so the index alignment with
    the paired bootstrap is preserved.

    Parameters
    ----------
    predictions, references :
        Same length. ``references[i]`` is the (non-empty) list of gold
        answers for example ``i``.
    model_type :
        HF model id for the encoder. Default ``distilbert-base-uncased``
        keeps the scorer fast on CPU.
    lang :
        Passed through to ``bert_score.score`` for tokenizer config and
        (when ``rescale_with_baseline=True``) baseline lookup.
    rescale_with_baseline :
        See module docstring. Linear transform; does not affect Δ
        significance.
    batch_size, device, verbose :
        Forwarded to ``bert_score.score``. ``device=None`` lets
        ``bert_score`` autodetect.

    Returns
    -------
    list[float] of length ``len(predictions)``.
    """
    import bert_score

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")
    if not predictions:
        return []

    # Flatten (prediction, reference_j) pairs into one bert_score.score
    # call so we hit the model in a single batched pass, then collapse
    # back to per-query max F1.
    flat_cands: list[str] = []
    flat_refs: list[str] = []
    spans: list[tuple[int, int]] = []
    for pred, refs in zip(predictions, references):
        start = len(flat_cands)
        if not pred.strip() or not refs:
            spans.append((start, start))
            continue
        non_empty = [r for r in refs if r]
        if not non_empty:
            spans.append((start, start))
            continue
        for r in non_empty:
            flat_cands.append(pred)
            flat_refs.append(r)
        spans.append((start, len(flat_cands)))

    if not flat_cands:
        return [0.0] * len(predictions)

    score_kwargs: dict = {
        "cands": flat_cands,
        "refs": flat_refs,
        "model_type": model_type,
        "lang": lang,
        "rescale_with_baseline": rescale_with_baseline,
        "batch_size": batch_size,
        "verbose": verbose,
    }
    if device is not None:
        score_kwargs["device"] = device
    _P, _R, F = bert_score.score(**score_kwargs)
    f_list = [float(x) for x in F.tolist()]

    out: list[float] = []
    for lo, hi in spans:
        if lo == hi:
            out.append(0.0)
        else:
            out.append(max(f_list[lo:hi]))
    return out
