"""Low-cost semantic evaluation sanity check on 3 000 paired examples.

Computes a DistilBERT-based BERTScore F1 per query for both the BM25
generation predictions and the reranked generation predictions, then
runs the project's existing paired-bootstrap CI on Δ (rerank − BM25).
This complements the surface-form metric CIs already published in the
W3 table and guides whether a full ``roberta-large`` BERTScore pass or
T5-small SFT is warranted before further work.

This is *not* a final, citation-grade BERTScore evaluation. It is a
fast proxy whose purpose is to answer a single question: is the
ROUGE/BLEU/EM/F1 Δ also visible in a semantic-similarity metric?

Inputs (same convention as ``scripts/bootstrap_generation_comparison``):

- ``<bm25-dir>/predictions.jsonl``
- ``<reranked-dir>/predictions.jsonl``

The two prediction files must cover the same qid set in the same order
(the W3 generation runner guarantees this when both runs use the same
``--restrict-to-run`` argument and seed).

Usage::

    python scripts/bertscore_paired_eval.py \\
        --bm25-dir outputs/generation_bm25_full \\
        --reranked-dir outputs/generation_reranked_full \\
        --output-dir outputs/bertscore_proxy \\
        --n-pairs 3000

Outputs:

- ``<output-dir>/bertscore_proxy_ci.json``  — machine-readable summary
  with full provenance (model type, n_pairs, subsample seed, bootstrap
  params, ``rescale_with_baseline`` flag).
- console table — human-readable.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from msmarco_genqa.evaluation.bertscore import per_query_bertscore_f1
from msmarco_genqa.evaluation.bootstrap import paired_bootstrap_diff

logger = logging.getLogger("bertscore_paired_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bm25-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/generation_bm25_full",
    )
    parser.add_argument(
        "--reranked-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/generation_reranked_full",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/bertscore_proxy",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=3000,
        help=(
            "Paired-qid subsample size. Pass 0 to score every shared "
            "qid (a full-dev pass takes hours on CPU for DistilBERT)."
        ),
    )
    parser.add_argument(
        "--subsample-seed",
        type=int,
        default=42,
        help="Seed for the paired-qid subsample.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="distilbert-base-uncased",
    )
    parser.add_argument(
        "--no-rescale-with-baseline",
        action="store_true",
        help="Disable bert_score's baseline rescaling (kept ON by default).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def _relpath(p: Path) -> str:
    """Render `p` relative to PROJECT_ROOT when it lives under the repo,
    else return the absolute path. Matches the convention in
    ``bootstrap_generation_comparison.py``; tolerates relative CLI input
    like ``--bm25-dir outputs/...`` without raising."""
    p = p.resolve()
    if p.is_relative_to(PROJECT_ROOT):
        return str(p.relative_to(PROJECT_ROOT))
    return str(p)


def load_predictions_ordered(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    bm25 = load_predictions_ordered(args.bm25_dir / "predictions.jsonl")
    rerank = load_predictions_ordered(args.reranked_dir / "predictions.jsonl")
    if len(bm25) != len(rerank):
        raise SystemExit(
            f"Prediction files have different lengths: "
            f"bm25={len(bm25)} reranked={len(rerank)}."
        )
    bm25_qids = [str(r["query_id"]) for r in bm25]
    rerank_qids = [str(r["query_id"]) for r in rerank]
    if bm25_qids != rerank_qids:
        if set(bm25_qids) == set(rerank_qids):
            raise SystemExit(
                "Prediction files cover the same qids but in different order."
            )
        raise SystemExit(
            "Prediction files cover different qid sets — pairing impossible."
        )

    # ---- paired subsample (by index, since the two lists are already
    # aligned index-wise) ----
    n_total = len(bm25)
    if args.n_pairs <= 0 or args.n_pairs >= n_total:
        sampled_indices = list(range(n_total))
        n_sampled = n_total
        subsample_label = "full"
    else:
        rng = random.Random(args.subsample_seed)
        sampled_indices = sorted(rng.sample(range(n_total), args.n_pairs))
        n_sampled = args.n_pairs
        subsample_label = f"random seed={args.subsample_seed}"
    logger.info(
        "Paired subsample: %d of %d shared qids (%s).",
        n_sampled,
        n_total,
        subsample_label,
    )

    bm25_preds = [bm25[i]["prediction"] for i in sampled_indices]
    rerank_preds = [rerank[i]["prediction"] for i in sampled_indices]
    references = [list(bm25[i]["references"]) for i in sampled_indices]

    rescale = not args.no_rescale_with_baseline

    # ---- score (the slow step) ----
    logger.info(
        "Scoring DistilBERT BERTScore-F1: model=%s  rescale=%s  batch_size=%d",
        args.model_type,
        rescale,
        args.batch_size,
    )
    t0 = time.time()
    logger.info("  ... BM25 system ...")
    bm25_scores = per_query_bertscore_f1(
        bm25_preds,
        references,
        model_type=args.model_type,
        lang="en",
        rescale_with_baseline=rescale,
        batch_size=args.batch_size,
        device=args.device,
        verbose=False,
    )
    bm25_secs = time.time() - t0
    logger.info("  BM25 scoring done in %.1f s.", bm25_secs)

    t1 = time.time()
    logger.info("  ... reranked system ...")
    rerank_scores = per_query_bertscore_f1(
        rerank_preds,
        references,
        model_type=args.model_type,
        lang="en",
        rescale_with_baseline=rescale,
        batch_size=args.batch_size,
        device=args.device,
        verbose=False,
    )
    rerank_secs = time.time() - t1
    logger.info("  Reranked scoring done in %.1f s.", rerank_secs)

    # ---- bootstrap ----
    logger.info(
        "Paired bootstrap: n_resamples=%d, ci=%.2f, seed=%d",
        args.n_resamples,
        args.ci,
        args.bootstrap_seed,
    )
    result = paired_bootstrap_diff(
        bm25_scores,
        rerank_scores,
        n_resamples=args.n_resamples,
        ci=args.ci,
        seed=args.bootstrap_seed,
    )
    win_rate = (
        sum(1 for a, b in zip(bm25_scores, rerank_scores) if b > a)
        / max(n_sampled, 1)
    )
    tie_rate = (
        sum(1 for a, b in zip(bm25_scores, rerank_scores) if b == a)
        / max(n_sampled, 1)
    )
    loss_rate = 1.0 - win_rate - tie_rate

    # ---- persist ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "bertscore_proxy_paired_bootstrap",
        "label": (
            "DistilBERT-based BERTScore — semantic proxy / subsample sanity "
            "check, NOT the final full-dev semantic evaluation. Purpose: "
            "complement surface-form metric CIs and guide whether a full "
            "roberta-large pass or T5-small SFT is warranted."
        ),
        "inputs": {
            "bm25_predictions": _relpath(args.bm25_dir / "predictions.jsonl"),
            "reranked_predictions": _relpath(args.reranked_dir / "predictions.jsonl"),
        },
        "subsample": {
            "n_total_shared_qids": n_total,
            "n_sampled": n_sampled,
            "subsample_seed": args.subsample_seed,
            "policy": subsample_label,
        },
        "bertscore": {
            "model_type": args.model_type,
            "lang": "en",
            "rescale_with_baseline": rescale,
            "batch_size": args.batch_size,
            "device": args.device,
            "bm25_scoring_seconds": round(bm25_secs, 2),
            "rerank_scoring_seconds": round(rerank_secs, 2),
        },
        "bootstrap": result,
        "win_rate_rerank_strictly_better": float(win_rate),
        "tie_rate": float(tie_rate),
        "loss_rate_bm25_strictly_better": float(loss_rate),
        "notes": (
            "Per-query BERTScore F1 with best-of-N reference scoring. "
            "Multi-reference: max F1 across references. Pairing is by "
            "query_id and preserved by index alignment. The bootstrap "
            "resamples paired (a, b) tuples; the percentile interval is "
            "the reported CI."
        ),
    }
    out_path = args.output_dir / "bertscore_proxy_ci.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s", out_path)

    # ---- console table ----
    r = result
    ci_str = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
    print()
    print("=== BERTScore proxy (DistilBERT) — paired bootstrap on subsample ===")
    print(
        f"n paired qids: {n_sampled} / {n_total}   "
        f"model: {args.model_type}   rescale_with_baseline: {rescale}"
    )
    print(
        f"  {'metric':18s}  {'bm25':>8s}  {'rerank':>8s}  {'Δ':>9s}  "
        f"{'95% CI':>22s}  {'p2':>8s}"
    )
    print(
        f"  {'bertscore-f1':18s}  {r['mean_a']:>8.4f}  {r['mean_b']:>8.4f}  "
        f"{r['mean_delta']:>+9.4f}  {ci_str:>22s}  {r['p_two_sided']:>8.4f}"
    )
    print()
    print(
        f"  per-query: win {win_rate:.3f}   tie {tie_rate:.3f}   "
        f"loss {loss_rate:.3f}   (rerank − BM25)"
    )
    print()
    print(
        "Reminder: semantic proxy on a "
        f"{n_sampled}-paired-qid subsample, not full-dev. "
        "See bertscore_proxy_ci.json 'label' for context."
    )


if __name__ == "__main__":
    main()
