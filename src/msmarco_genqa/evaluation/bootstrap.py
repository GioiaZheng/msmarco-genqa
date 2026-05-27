"""Paired bootstrap confidence intervals for per-query metric deltas.

When comparing two systems (e.g. BM25-vs-reranked retrieval feeding the
same generator on the same query subsample), the *level* numbers depend
on the query sample but the *delta* is a paired quantity: each query
contributes one (a_score, b_score) pair. A paired bootstrap resamples
queries with replacement and recomputes mean(b - a) per resample; the
empirical percentile interval is a non-parametric 95% CI on that delta.

This module also exposes per-example scorers for ROUGE-L and BLEU,
matching the metrics produced corpus-wise by ``evaluate_generation``:

- ``per_query_rouge_l`` uses ``rouge_score.rouge_scorer`` directly with
  the same multi-reference best-of-N convention.
- ``per_query_bleu`` uses NLTK's ``sentence_bleu`` with smoothing method 1
  (avoids 0s when a 4-gram precision is empty, which is common at short
  MS MARCO QA answer lengths).

Token-F1 and Exact-Match per-query scorers already live in
``msmarco_genqa.evaluation.generation`` (``token_f1``, ``exact_match``).
"""

from __future__ import annotations

import math
import random
from typing import Sequence


# --------------------------------------------------------------------------- #
# Per-query metric scorers
# --------------------------------------------------------------------------- #


def per_query_rouge_l(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    *,
    use_stemmer: bool = True,
) -> list[float]:
    """Per-example ROUGE-L F-measure with best-of-N reference scoring.

    Returns a list the same length as ``predictions``. Examples with an
    empty reference list score 0.0 (mirrors ``evaluate_generation`` which
    drops them from the corpus-level computation; here we keep one entry
    per index so bootstrap resampling stays paired).
    """
    from rouge_score import rouge_scorer

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=use_stemmer)
    out: list[float] = []
    for pred, refs in zip(predictions, references):
        if not refs:
            out.append(0.0)
            continue
        best = 0.0
        for ref in refs:
            f = scorer.score(ref, pred)["rougeL"].fmeasure
            if f > best:
                best = f
        out.append(float(best))
    return out


def per_query_bleu(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> list[float]:
    """Per-example sentence-BLEU with best-of-N reference scoring.

    Uses NLTK's ``sentence_bleu`` with smoothing method 1, the standard
    choice for short answers where one of the n-gram precisions is often
    zero. Multi-reference mode: a single ``sentence_bleu`` call already
    takes a list of references and picks the closest-length / max-overlap
    reference internally.
    """
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")
    smoothing = SmoothingFunction().method1
    out: list[float] = []
    for pred, refs in zip(predictions, references):
        if not refs or not pred.strip():
            out.append(0.0)
            continue
        ref_tokens = [r.split() for r in refs if r]
        if not ref_tokens:
            out.append(0.0)
            continue
        pred_tokens = pred.split()
        score = sentence_bleu(
            ref_tokens,
            pred_tokens,
            smoothing_function=smoothing,
        )
        out.append(float(score))
    return out


# --------------------------------------------------------------------------- #
# Paired bootstrap
# --------------------------------------------------------------------------- #


def paired_bootstrap_diff(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Paired-bootstrap 95% CI on mean(b - a).

    Parameters
    ----------
    scores_a, scores_b :
        Per-query metric scores. Must be the same length and the same
        order (index i in both refers to the same query).
    n_resamples :
        Number of bootstrap resamples. 10k is standard for stable
        percentile CIs at this sample size.
    ci :
        Coverage of the percentile interval (default 0.95).
    seed :
        Reproducibility seed for the resample indices.

    Returns
    -------
    dict with:
        mean_a, mean_b, mean_delta :
            Point estimates on the observed sample.
        ci_low, ci_high :
            Percentile interval on the bootstrap distribution of
            mean(b - a).
        p_two_sided :
            Two-sided bootstrap p-value: 2 * min(P(delta* <= 0),
            P(delta* >= 0)), clipped to [0, 1]. Reported alongside the
            CI for readers used to NHST; the CI is the load-bearing
            quantity.
        n :
            Number of paired examples.
        n_resamples, ci :
            Echoed for traceability.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have same length")
    n = len(scores_a)
    if n == 0:
        raise ValueError("need at least one paired observation")
    if not 0 < ci < 1:
        raise ValueError("ci must be in (0, 1)")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")

    a = list(scores_a)
    b = list(scores_b)
    diffs = [b[i] - a[i] for i in range(n)]
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    mean_delta = sum(diffs) / n

    rng = random.Random(seed)
    boot_deltas: list[float] = []
    n_ge_zero = 0
    n_le_zero = 0
    for _ in range(n_resamples):
        s = 0.0
        for _j in range(n):
            s += diffs[rng.randrange(n)]
        d = s / n
        boot_deltas.append(d)
        if d >= 0:
            n_ge_zero += 1
        if d <= 0:
            n_le_zero += 1

    boot_deltas.sort()
    alpha = (1.0 - ci) / 2.0
    lo_idx = int(math.floor(alpha * n_resamples))
    hi_idx = int(math.ceil((1.0 - alpha) * n_resamples)) - 1
    lo_idx = max(0, min(n_resamples - 1, lo_idx))
    hi_idx = max(0, min(n_resamples - 1, hi_idx))
    ci_low = boot_deltas[lo_idx]
    ci_high = boot_deltas[hi_idx]

    p_two_sided = 2.0 * min(n_ge_zero, n_le_zero) / n_resamples
    p_two_sided = max(0.0, min(1.0, p_two_sided))

    return {
        "mean_a": float(mean_a),
        "mean_b": float(mean_b),
        "mean_delta": float(mean_delta),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_two_sided": float(p_two_sided),
        "n": int(n),
        "n_resamples": int(n_resamples),
        "ci": float(ci),
        "seed": int(seed),
    }
