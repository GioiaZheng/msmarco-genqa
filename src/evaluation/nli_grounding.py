"""NLI-based grounding: entailment probability that a top-K passage
union entails the generator's prediction.

This is the *semantic faithfulness* analogue to the regex-only lexical
and 3-gram grounding metrics in ``src.evaluation.grounding``. A heavy
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
``src.evaluation.bootstrap``.
"""

from __future__ import annotations

from typing import Sequence

# HF revision pin for the default NLI cross-encoder. Hardcoded (not
# config-driven) because configs/baseline.yaml has no NLI block --
# grounding is a post-hoc audit, not part of the per-run config surface.
# To bump: look up the latest SHA via huggingface_hub.HfApi (see
# scripts/backfill_provenance.py for the lookup pattern) and update
# this constant + any call sites that override ``model_type``.
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
DEFAULT_NLI_REVISION = "fa2804872c3b4bd748f38c0185cc85775361e735"


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
        scorers in ``src.evaluation``.

    Returns
    -------
    list[float] of length ``len(predictions)``.
    """
    if len(predictions) != len(passages_lists):
        raise ValueError(
            "predictions and passages_lists must have the same length"
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

    id2label = model.config.id2label
    entail_idx: int | None = None
    for idx, label in id2label.items():
        if str(label).lower() == "entailment":
            entail_idx = int(idx)
            break
    if entail_idx is None:
        raise ValueError(
            f"Model {model_type} has no 'entailment' label in id2label={id2label}"
        )

    # Build premise / hypothesis pairs. Edge cases (empty pred or
    # all-empty passages) get a placeholder pair scored to 0.0 below;
    # we always submit every index so batched alignment stays simple.
    premises: list[str] = []
    hypotheses: list[str] = []
    zero_mask: list[bool] = []
    for pred, psgs in zip(predictions, passages_lists):
        hypothesis = (pred or "").strip()
        premise = " ".join(p for p in (psgs or []) if p).strip()
        if not hypothesis or not premise:
            premises.append("placeholder")
            hypotheses.append("placeholder")
            zero_mask.append(True)
        else:
            premises.append(premise)
            hypotheses.append(hypothesis)
            zero_mask.append(False)

    scores: list[float] = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_premises = premises[i : i + batch_size]
            batch_hypotheses = hypotheses[i : i + batch_size]
            enc = tokenizer(
                batch_premises,
                batch_hypotheses,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            scores.extend(probs[:, entail_idx].cpu().tolist())

    return [0.0 if z else float(s) for z, s in zip(zero_mask, scores)]
