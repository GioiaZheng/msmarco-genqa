"""NLI-based grounding: entailment probability that a top-K passage
union entails the generator's prediction.

This is the *semantic faithfulness* analogue to the regex-only lexical
and 3-gram grounding metrics in ``msmarco_genqa.evaluation.grounding``. A heavy
paraphrase that is perfectly faithful semantically will (usually)
score high here even though it scores low on lexical / 3-gram
grounding; a verbatim copy from a distractor passage that contradicts
the gold answer will score *low* even though it scores 1.0 on lexical
grounding.

The metric is the model's softmax probability for the "entailment"
label, in [0, 1], on the pair (premise = joined top-K passages,
hypothesis = prediction). One forward pass per query; entailment is
*not* aggregated over individual passages — that would require K
passes per query and is the obvious follow-up.

Default model: ``cross-encoder/nli-deberta-v3-small`` (~140 MB, CPU
inference: ~30 ms / pair on a 6-core MacBook). The function reads
``model.config.id2label`` to locate the entailment index so it works
unchanged across NLI checkpoints that index labels differently.

Edge cases (documented, stable):

- Empty prediction → 0.0 (no hypothesis to entail).
- Empty / all-empty passages → 0.0 (no premise; conservative).
- Predictions shorter than a few tokens still receive a real
  entailment score; we do *not* short-circuit "trivial" outputs
  because that defeats the point of having a semantic check.

The function returns a list aligned with the input order so it can be
plugged straight into the paired bootstrap utilities in
``msmarco_genqa.evaluation.bootstrap``.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

# HF revision pin for the default NLI cross-encoder. Hardcoded (not
# config-driven) because configs/baseline.yaml has no NLI block --
# grounding is a post-hoc audit, not part of the per-run config surface.
# To bump: look up the latest SHA via huggingface_hub.HfApi (see
# scripts/backfill_provenance.py for the lookup pattern) and update
# this constant + any call sites that override ``model_type``.
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
DEFAULT_NLI_REVISION = "fa2804872c3b4bd748f38c0185cc85775361e735"

# The three NLI label keys this module standardises on, independent of how
# any given checkpoint orders them in ``config.id2label``. Every per-query
# probability dict and every score formula speaks in these keys.
NLI_LABELS: tuple[str, str, str] = ("entailment", "neutral", "contradiction")

# Score formulas for the R5 metric-robustness factorial. Each maps the
# three softmax probabilities (entailment, neutral, contradiction) to a
# single grounding score. The factorial sweeps these as one of its axes;
# the chosen name is recorded verbatim in ``extra.nli.score_formula`` on
# the run manifest (see ``util.manifest.PROFILE_REQUIRED_FIELDS``).
#
# - ``entailment``                     : P(entail). The W7-A default; kept
#   as the identity baseline so historical numbers stay reproducible.
# - ``entailment_minus_contradiction`` : P(entail) - P(contradict). Nets out
#   active contradiction so a confident wrong copy scores below neutral text.
# - ``calibrated``                     : P(entail) - P(neutral) - 2*P(contradict).
#   Penalises hedging (neutral) lightly and contradiction heavily; the
#   most opinionated of the three.
SCORE_FORMULAS: dict[str, Callable[[float, float, float], float]] = {
    "entailment": lambda e, n, c: e,
    "entailment_minus_contradiction": lambda e, n, c: e - c,
    "calibrated": lambda e, n, c: e - n - 2.0 * c,
}


def resolve_label_indices(id2label: Mapping[int, str]) -> dict[str, int]:
    """Map each of ``NLI_LABELS`` to its column index in a model's logits.

    NLI checkpoints disagree on label ordering (some emit
    ``contradiction, neutral, entailment``, others the reverse), so the
    only safe key is the label *name* from ``config.id2label`` rather than
    a positional assumption. Matching is case-insensitive and tolerates the
    short-form ``contradict`` some checkpoints use.

    Raises ``ValueError`` listing every label that could not be located, so
    a misindexed checkpoint fails loudly at setup rather than silently
    scoring against the wrong column.
    """
    aliases = {
        "entailment": {"entailment", "entail"},
        "neutral": {"neutral"},
        "contradiction": {"contradiction", "contradict"},
    }
    found: dict[str, int] = {}
    for idx, raw in id2label.items():
        name = str(raw).strip().lower()
        for canonical, names in aliases.items():
            if name in names:
                found[canonical] = int(idx)
    missing = [label for label in NLI_LABELS if label not in found]
    if missing:
        raise ValueError(
            f"NLI model id2label={dict(id2label)} is missing label(s) "
            f"{missing}; cannot map to {NLI_LABELS}."
        )
    return found


def nli_score(probs: Mapping[str, float], formula: str) -> float:
    """Collapse a per-query 3-class probability dict to one score.

    ``probs`` must carry the three ``NLI_LABELS`` keys (as produced by
    :func:`per_query_nli_probs`). ``formula`` selects an entry from
    :data:`SCORE_FORMULAS`; an unknown name raises ``ValueError`` up-front
    so a typo cannot silently degrade to a default.
    """
    if formula not in SCORE_FORMULAS:
        raise ValueError(
            f"unknown score formula {formula!r}. Known formulas: "
            f"{sorted(SCORE_FORMULAS)}."
        )
    return float(
        SCORE_FORMULAS[formula](
            float(probs["entailment"]),
            float(probs["neutral"]),
            float(probs["contradiction"]),
        )
    )


def per_query_nli_probs(
    predictions: Sequence[str],
    passages_lists: Sequence[Sequence[str]],
    *,
    model_type: str = DEFAULT_NLI_MODEL,
    revision: str | None = DEFAULT_NLI_REVISION,
    direction: str = "passages_to_prediction",
    batch_size: int = 16,
    device: str | None = None,
    max_length: int = 512,
) -> list[dict[str, float]]:
    """Per-query 3-class NLI probabilities for ``passages`` vs ``prediction``.

    This is the factorial primitive underneath
    :func:`per_query_nli_entailment`: it returns the full softmax over the
    three :data:`NLI_LABELS` so any of :data:`SCORE_FORMULAS` can be applied
    downstream without re-running the model.

    Parameters
    ----------
    predictions :
        One generated answer per example.
    passages_lists :
        Same length as ``predictions``. ``passages_lists[i]`` is the list of
        top-K passage texts the generator saw for example ``i``; they are
        joined with single spaces into one passage string.
    model_type :
        HF model id for an NLI cross-encoder. Must expose 3-class logits
        whose ``config.id2label`` carries the three :data:`NLI_LABELS`.
    direction :
        Premise/hypothesis assignment, a factorial axis recorded in
        ``extra.nli.premise_hypothesis_direction``:

        - ``"passages_to_prediction"`` (default): premise = passages,
          hypothesis = prediction. "Do the passages entail the answer?"
        - ``"prediction_to_passages"``: the reverse pairing.
    batch_size :
        Forward-pass batch size. Default ``16`` keeps CPU latency bounded on
        the default DeBERTa-v3-small checkpoint.
    device :
        Override device string ("cpu" / "cuda" / "mps"). ``None``
        auto-detects CUDA, otherwise CPU.
    max_length :
        Tokeniser truncation length. The default of 512 fits the
        DeBERTa-v3-small context window.

    Returns
    -------
    list[dict[str, float]] of length ``len(predictions)``. Each dict carries
    the three :data:`NLI_LABELS` keys. Edge cases (empty prediction or
    all-empty passages) return all-zero probabilities, mirroring the
    conservative ``0.0`` entailment of the original audit.
    """
    if len(predictions) != len(passages_lists):
        raise ValueError(
            "predictions and passages_lists must have the same length"
        )
    if direction not in ("passages_to_prediction", "prediction_to_passages"):
        raise ValueError(
            f"unknown direction {direction!r}; expected "
            "'passages_to_prediction' or 'prediction_to_passages'."
        )
    n = len(predictions)
    if n == 0:
        return []

    import torch  # local import: avoid CPU/GPU init at module load time
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_kwargs: dict = {}
    if revision is not None:
        hf_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_type, **hf_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_type, **hf_kwargs
    )
    model = model.to(device).eval()

    label_idx = resolve_label_indices(model.config.id2label)

    # Build premise / hypothesis pairs. Edge cases (empty pred or all-empty
    # passages) get a placeholder pair zeroed out below; we always submit
    # every index so batched alignment stays simple.
    passage_side: list[str] = []
    prediction_side: list[str] = []
    zero_mask: list[bool] = []
    for pred, psgs in zip(predictions, passages_lists):
        hypothesis = (pred or "").strip()
        premise = " ".join(p for p in (psgs or []) if p).strip()
        if not hypothesis or not premise:
            passage_side.append("placeholder")
            prediction_side.append("placeholder")
            zero_mask.append(True)
        else:
            passage_side.append(premise)
            prediction_side.append(hypothesis)
            zero_mask.append(False)

    if direction == "passages_to_prediction":
        first_side, second_side = passage_side, prediction_side
    else:
        first_side, second_side = prediction_side, passage_side

    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            enc = tokenizer(
                first_side[i : i + batch_size],
                second_side[i : i + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            for row in probs:
                rows.append({label: float(row[label_idx[label]]) for label in NLI_LABELS})

    zero_row = {label: 0.0 for label in NLI_LABELS}
    return [dict(zero_row) if z else r for z, r in zip(zero_mask, rows)]


def per_query_nli_entailment(
    predictions: Sequence[str],
    passages_lists: Sequence[Sequence[str]],
    *,
    model_type: str = DEFAULT_NLI_MODEL,
    revision: str | None = DEFAULT_NLI_REVISION,
    batch_size: int = 16,
    device: str | None = None,
    max_length: int = 512,
    verbose: bool = False,
) -> list[float]:
    """Per-query entailment probability for ``passages → prediction``.

    Thin backward-compatible wrapper over :func:`per_query_nli_probs`:
    returns the ``entailment`` column (the ``"entailment"`` score formula),
    preserving the exact numbers the W7-A audit reported.

    Parameters
    ----------
    predictions :
        One generated answer per example.
    passages_lists :
        Same length as ``predictions``. ``passages_lists[i]`` is the
        list of top-K passage texts the generator saw for example ``i``;
        they are joined with single spaces into one premise string.
    model_type :
        HF model id for an NLI cross-encoder. Must expose 3-class
        logits with an ``entailment`` label discoverable via
        ``config.id2label``.
    batch_size :
        Forward-pass batch size. Default ``16`` keeps CPU latency
        bounded on the default DeBERTa-v3-small checkpoint.
    device :
        Override device string ("cpu" / "cuda" / "mps"). ``None``
        auto-detects CUDA, otherwise CPU.
    max_length :
        Tokeniser truncation length. The default of 512 fits the
        DeBERTa-v3-small context window; with K=3 MS MARCO passages
        + a short hypothesis this is enough head-room that truncation
        only fires on the tail of long passages.
    verbose :
        Currently unused; reserved for parity with the other per-query
        scorers in ``msmarco_genqa.evaluation``.

    Returns
    -------
    list[float] of length ``len(predictions)``.
    """
    rows = per_query_nli_probs(
        predictions,
        passages_lists,
        model_type=model_type,
        revision=revision,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
    )
    return [nli_score(row, "entailment") for row in rows]
