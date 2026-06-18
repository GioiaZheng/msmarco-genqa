"""Paired-bootstrap 95% CIs on the BM25-vs-reranked generation comparison.

Inputs (both produced by ``experiments/run_generation_baseline.py`` with
different ``--input-run`` upstreams):

- ``<bm25-dir>/predictions.jsonl``
- ``<reranked-dir>/predictions.jsonl``

The two prediction files MUST cover the same set of query_ids in the same
order (the W3 generation runner guarantees this when both runs share the
same ``--restrict-to-run`` argument and seed). The script enforces this
and exits with a clear error otherwise.

For each of token-F1, exact-match, ROUGE-L and sentence-BLEU, compute:

- per-query scores for each system,
- the mean delta (rerank − BM25) on the observed sample,
- a percentile-based 95% paired-bootstrap CI on that delta,
- a two-sided bootstrap p-value.

Outputs:

- ``<output-dir>/bootstrap_ci.json``  — machine-readable summary.
- console table — human-readable summary.

Usage::

    python scripts/bootstrap_generation_comparison.py \\
        --bm25-dir outputs/generation_bm25 \\
        --reranked-dir outputs/generation_reranked \\
        --output-dir outputs/generation_bootstrap

The defaults point at the 200-query W3 comparison directories.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from msmarco_genqa.evaluation.bootstrap import (
    paired_bootstrap_diff,
    per_query_bleu,
    per_query_rouge_l,
)
from msmarco_genqa.evaluation.generation import exact_match, token_f1

logger = logging.getLogger("bootstrap_generation_comparison")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bm25-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/generation_bm25",
    )
    parser.add_argument(
        "--reranked-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/generation_reranked",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/generation_bootstrap",
    )
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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
            f"bm25={len(bm25)} reranked={len(rerank)}. "
            "Re-run both generators with the same --restrict-to-run and seed."
        )
    bm25_qids = [str(r["query_id"]) for r in bm25]
    rerank_qids = [str(r["query_id"]) for r in rerank]
    if bm25_qids != rerank_qids:
        if set(bm25_qids) == set(rerank_qids):
            raise SystemExit(
                "Prediction files cover the same qids but in different order. "
                "The bootstrap is paired by index, so order must match. Sort both "
                "files by query_id and re-run, or fix the generation runner."
            )
        raise SystemExit(
            "Prediction files cover different qid sets. The bootstrap is paired; "
            "the two systems must be evaluated on the same query subsample."
        )
    n = len(bm25)
    logger.info("Paired bootstrap on %d shared qids.", n)

    bm25_preds = [r["prediction"] for r in bm25]
    rerank_preds = [r["prediction"] for r in rerank]
    # References come from MS MARCO QA v2.1; they're identical across the two
    # generation runs by construction. Use the BM25-side copy.
    references = [list(r["references"]) for r in bm25]

    # ---- per-query scores ----
    logger.info("Scoring token-F1 per query...")
    bm25_tf1 = [token_f1(p, refs) for p, refs in zip(bm25_preds, references)]
    rerank_tf1 = [token_f1(p, refs) for p, refs in zip(rerank_preds, references)]

    logger.info("Scoring exact-match per query...")
    bm25_em = [exact_match(p, refs) for p, refs in zip(bm25_preds, references)]
    rerank_em = [exact_match(p, refs) for p, refs in zip(rerank_preds, references)]

    logger.info("Scoring ROUGE-L per query (rouge_score)...")
    bm25_rl = per_query_rouge_l(bm25_preds, references)
    rerank_rl = per_query_rouge_l(rerank_preds, references)

    logger.info("Scoring sentence-BLEU per query (NLTK, smoothing method1)...")
    bm25_bleu = per_query_bleu(bm25_preds, references)
    rerank_bleu = per_query_bleu(rerank_preds, references)

    # ---- bootstrap ----
    logger.info(
        "Paired bootstrap: n_resamples=%d, ci=%.2f, seed=%d",
        args.n_resamples,
        args.ci,
        args.seed,
    )
    results: dict[str, dict[str, float]] = {}
    for name, a, b in [
        ("rouge-l", bm25_rl, rerank_rl),
        ("bleu", bm25_bleu, rerank_bleu),
        ("exact-match", bm25_em, rerank_em),
        ("token-f1", bm25_tf1, rerank_tf1),
    ]:
        results[name] = paired_bootstrap_diff(
            a,
            b,
            n_resamples=args.n_resamples,
            ci=args.ci,
            seed=args.seed,
        )

    # ---- persist ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bm25_dir_rel = (
        str(args.bm25_dir.relative_to(PROJECT_ROOT))
        if args.bm25_dir.is_relative_to(PROJECT_ROOT)
        else str(args.bm25_dir)
    )
    reranked_dir_rel = (
        str(args.reranked_dir.relative_to(PROJECT_ROOT))
        if args.reranked_dir.is_relative_to(PROJECT_ROOT)
        else str(args.reranked_dir)
    )
    payload = {
        "task": "generation_paired_bootstrap",
        "inputs": {
            "bm25_predictions": f"{bm25_dir_rel}/predictions.jsonl",
            "reranked_predictions": f"{reranked_dir_rel}/predictions.jsonl",
        },
        "n_shared_qids": n,
        "n_resamples": args.n_resamples,
        "ci": args.ci,
        "seed": args.seed,
        "metrics": results,
        "notes": (
            "Per-query scores were computed in-script and then resampled "
            "with replacement to form 'n_resamples' bootstrap samples of "
            "mean(rerank - bm25). The reported interval is the percentile "
            "interval of that bootstrap distribution. 'p_two_sided' is "
            "2 * min(P(delta* <= 0), P(delta* >= 0))."
        ),
    }
    out_path = args.output_dir / "bootstrap_ci.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s", out_path)

    # ---- friendly console table ----
    print("\n=== Paired-bootstrap 95% CI: rerank − BM25 ===")
    print(f"shared qids: {n}   n_resamples: {args.n_resamples}   seed: {args.seed}")
    print(f"  {'metric':12s}  {'bm25':>8s}  {'rerank':>8s}  {'Δ':>8s}"
          f"  {'95% CI':>22s}  {'p2':>8s}")
    for name in ("rouge-l", "bleu", "exact-match", "token-f1"):
        r = results[name]
        ci_str = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        print(
            f"  {name:12s}  {r['mean_a']:>8.4f}  {r['mean_b']:>8.4f}  "
            f"{r['mean_delta']:>+8.4f}  {ci_str:>22s}  {r['p_two_sided']:>8.4f}"
        )


if __name__ == "__main__":
    main()
