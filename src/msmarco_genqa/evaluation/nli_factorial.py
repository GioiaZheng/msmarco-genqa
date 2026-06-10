"""R5 metric-robustness: NLI grounding factorial over cached probabilities.

The expensive part of the factorial — one NLI forward pass per query per
arm per backbone — is done once by
:func:`msmarco_genqa.evaluation.nli_grounding.per_query_nli_probs`, which
caches the full 3-class softmax. *Given* those cached probabilities, the
remaining factorial axes are free to derive:

- **score formula** (:data:`~msmarco_genqa.evaluation.nli_grounding.SCORE_FORMULAS`)
  collapses the 3-class probs to one grounding score;
- **threshold** binarises that score into grounded / not-grounded, or is
  left ``None`` for the threshold-free continuous mean.

This module turns cached per-arm probability lists into factorial *cells*.
A cell is one ``(formula, threshold)`` combination scored with a paired
bootstrap on the rerank effect ``delta = mean(reranked) - mean(bm25)``.
Everything here is pure and deterministic given a seed, so it is unit
tested without loading a model — the model path lives in the runner.

The headline question R5 answers: does the W7-A grounding sign-reversal
(rerank *lowers* NLI grounding while raising surface metrics) survive
across the formula x threshold grid and across backbones, or is it an
artifact of one ``(formula, threshold, backbone)`` choice?
"""

from __future__ import annotations

from typing import Mapping, Sequence

from msmarco_genqa.evaluation.bootstrap import paired_bootstrap_diff
from msmarco_genqa.evaluation.nli_grounding import SCORE_FORMULAS, nli_score

# Backbone registry for the R5 factorial: 2 small + 1 large, spanning three
# model families (DeBERTa / MiniLM / RoBERTa) so a sign-reversal that holds
# across all three cannot be a single-architecture artifact. Revisions are
# pinned SHAs (resolved via huggingface_hub.HfApi.model_info); bump by
# re-resolving and updating the SHA here. deberta-v3-small matches the
# W7-A default so the baseline cell reproduces the historical number.
NLI_BACKBONES: dict[str, dict[str, str]] = {
    "deberta-v3-small": {
        "model_id": "cross-encoder/nli-deberta-v3-small",
        "revision": "fa2804872c3b4bd748f38c0185cc85775361e735",
    },
    "minilm-l6": {
        "model_id": "cross-encoder/nli-MiniLM2-L6-H768",
        "revision": "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d",
    },
    "roberta-large-mnli": {
        "model_id": "roberta-large-mnli",
        "revision": "2a8f12d27941090092df78e4ba6f0928eb5eac98",
    },
}

# Default factorial axes. Formulas are every registered score; thresholds
# include ``None`` (threshold-free continuous mean, the W7-A regime) plus a
# fixed 0.5 cut. The FPR-equalised threshold is data-dependent (needs a
# labelled grounded set) and is supplied by the caller, not defaulted here.
DEFAULT_FORMULAS: tuple[str, ...] = tuple(SCORE_FORMULAS)
DEFAULT_THRESHOLDS: tuple[float | None, ...] = (None, 0.5)


def continuous_scores(
    probs: Sequence[Mapping[str, float]], formula: str
) -> list[float]:
    """Apply a score formula to a list of 3-class probability dicts."""
    return [nli_score(p, formula) for p in probs]


def binarize(scores: Sequence[float], threshold: float) -> list[float]:
    """Grounded (1.0) iff ``score >= threshold``, else 0.0."""
    return [1.0 if s >= threshold else 0.0 for s in scores]


def factorial_cell(
    probs_bm25: Sequence[Mapping[str, float]],
    probs_reranked: Sequence[Mapping[str, float]],
    *,
    formula: str,
    threshold: float | None = None,
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Score one ``(formula, threshold)`` cell of the factorial.

    The two probability lists must be paired and same-ordered: index ``i``
    in both is the same query, scored from the BM25 arm and the reranked
    arm respectively. The cell reports the paired-bootstrap CI on
    ``delta = mean(reranked) - mean(bm25)`` of the grounding score
    (continuous when ``threshold is None``, else the grounded *rate* after
    binarising at ``threshold``).

    ``reverses_sign`` is the load-bearing flag: ``True`` when the rerank
    effect is strictly negative (CI upper bound below zero), i.e. this cell
    reproduces the W7-A "rerank lowers grounding" reversal against the
    surface-metric lift.
    """
    if len(probs_bm25) != len(probs_reranked):
        raise ValueError("probs_bm25 and probs_reranked must have same length")

    a = continuous_scores(probs_bm25, formula)
    b = continuous_scores(probs_reranked, formula)
    aggregation = "grounded_rate" if threshold is not None else "mean_score"
    if threshold is not None:
        a = binarize(a, threshold)
        b = binarize(b, threshold)

    boot = paired_bootstrap_diff(
        a, b, n_resamples=n_resamples, ci=ci, seed=seed
    )
    return {
        "formula": formula,
        "threshold": threshold,
        "score_aggregation": aggregation,
        "reverses_sign": bool(boot["ci_high"] < 0.0),
        "raises_grounding": bool(boot["ci_low"] > 0.0),
        "bootstrap": boot,
    }


def run_factorial(
    probs_bm25: Sequence[Mapping[str, float]],
    probs_reranked: Sequence[Mapping[str, float]],
    *,
    formulas: Sequence[str] = DEFAULT_FORMULAS,
    thresholds: Sequence[float | None] = DEFAULT_THRESHOLDS,
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> list[dict]:
    """Score the full ``formula x threshold`` grid for one backbone's probs.

    Returns one cell dict per combination, in ``formulas`` x ``thresholds``
    order. The expensive forward pass is assumed already cached into the two
    probability lists; this is pure arithmetic plus bootstrap resampling.
    """
    cells: list[dict] = []
    for formula in formulas:
        for threshold in thresholds:
            cells.append(
                factorial_cell(
                    probs_bm25,
                    probs_reranked,
                    formula=formula,
                    threshold=threshold,
                    n_resamples=n_resamples,
                    ci=ci,
                    seed=seed,
                )
            )
    return cells
