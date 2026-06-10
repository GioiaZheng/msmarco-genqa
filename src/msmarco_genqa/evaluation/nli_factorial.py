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


def _cell_key(cell: Mapping) -> tuple:
    return (cell["formula"], cell["threshold"])


def aggregate_backbones(summaries: Sequence[Mapping]) -> dict:
    """Cross-backbone verdict for the Axis A sign-reversal gate.

    Each input is one backbone's run summary (as written by the factorial
    runner): it must carry ``backbone`` and a ``cells`` list of
    ``factorial_cell`` outputs. Summaries are expected to share the same
    ``(formula, threshold)`` grid; a cell key present in some backbones but
    not others is reported under ``missing_in`` rather than silently dropped.

    Per ``(formula, threshold)`` cell the verdict reports:

    - ``n_reverse`` / ``n_backbones``: how many backbones show a strict
      negative rerank effect (CI upper bound below zero);
    - ``robust_reversal``: every backbone reverses — the cell survives the
      metric-choice attack surface;
    - ``unanimous_rise``: every backbone instead *raises* grounding.

    ``headline`` summarises the baseline cell (``entailment`` formula,
    threshold ``None`` — the W7-A regime). ``baseline_robust_reversal`` true
    across >=3 backbones is the Axis A paper's go condition.
    """
    summaries = list(summaries)
    backbones = [str(s.get("backbone")) for s in summaries]

    keys: list[tuple] = []
    for s in summaries:
        for cell in s.get("cells", []):
            k = _cell_key(cell)
            if k not in keys:
                keys.append(k)

    cell_rows: list[dict] = []
    for formula, threshold in keys:
        per_backbone: list[dict] = []
        missing_in: list[str] = []
        for s in summaries:
            match = next(
                (c for c in s.get("cells", []) if _cell_key(c) == (formula, threshold)),
                None,
            )
            if match is None:
                missing_in.append(str(s.get("backbone")))
                continue
            boot = match["bootstrap"]
            per_backbone.append({
                "backbone": str(s.get("backbone")),
                "delta": boot["mean_delta"],
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
                "reverses_sign": bool(match["reverses_sign"]),
                "raises_grounding": bool(match["raises_grounding"]),
            })
        n_reverse = sum(1 for p in per_backbone if p["reverses_sign"])
        n_rise = sum(1 for p in per_backbone if p["raises_grounding"])
        cell_rows.append({
            "formula": formula,
            "threshold": threshold,
            "n_backbones": len(per_backbone),
            "n_reverse": n_reverse,
            "n_rise": n_rise,
            "robust_reversal": bool(per_backbone) and n_reverse == len(per_backbone),
            "unanimous_rise": bool(per_backbone) and n_rise == len(per_backbone),
            "missing_in": missing_in,
            "per_backbone": per_backbone,
        })

    baseline = next(
        (c for c in cell_rows if c["formula"] == "entailment" and c["threshold"] is None),
        None,
    )
    return {
        "backbones": backbones,
        "n_backbones": len(summaries),
        "n_cells": len(cell_rows),
        "n_robust_reversal_cells": sum(1 for c in cell_rows if c["robust_reversal"]),
        "headline": {
            "baseline_robust_reversal": bool(baseline and baseline["robust_reversal"]),
            "baseline_n_reverse": baseline["n_reverse"] if baseline else None,
            "baseline_n_backbones": baseline["n_backbones"] if baseline else None,
        },
        "cells": cell_rows,
    }
