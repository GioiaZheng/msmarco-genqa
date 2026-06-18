"""W7 grounding audit on existing W3 paired predictions.

For each shared query_id between the BM25-fed and reranked-fed
generation runs, compute two grounding scores on the *exact* top-K
passages the generator saw at prompt time (they are persisted inside
``predictions.jsonl`` alongside the model output):

- lexical content-token grounding (``msmarco_genqa.evaluation.grounding.lexical_grounding``)
- 3-gram grounding (``msmarco_genqa.evaluation.grounding.ngram_grounding``)

Then run the project's existing paired bootstrap CI on Δ (rerank −
BM25) for each metric. No new model, no generation, no NLI; CPU-only,
minutes.

Inputs (same convention as ``scripts/bootstrap_generation_comparison.py``):

- ``<bm25-dir>/predictions.jsonl``
- ``<reranked-dir>/predictions.jsonl``

The two files MUST cover the same set of query_ids in the same order
(the W3 generation runner guarantees this when both runs share the
same ``--restrict-to-run`` argument and seed). The script enforces
this and exits with a clear error otherwise.

Outputs (all under ``<output-dir>``, gitignored):

- ``per_query_grounding.jsonl``  — one row per qid with both arms.
- ``summary.json``               — means + paired-bootstrap CIs +
                                   win/tie/loss rates + edge-case
                                   diagnostics + full provenance.
- console table                  — human-readable summary.

Usage::

    python scripts/grounding_audit.py \\
        --bm25-dir outputs/generation_bm25_full \\
        --reranked-dir outputs/generation_reranked_full \\
        --output-dir outputs/grounding

The defaults point at the canonical full-dev W3 outputs.
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

from msmarco_genqa.evaluation.bootstrap import paired_bootstrap_diff
from msmarco_genqa.evaluation.grounding import (
    is_vacuously_grounded_lex,
    is_vacuously_grounded_ngram,
    lexical_grounding,
    ngram_grounding,
)

logger = logging.getLogger("grounding_audit")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

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
        default=PROJECT_ROOT / "outputs/grounding",
    )
    parser.add_argument(
        "--ngram-n",
        type=int,
        default=3,
        help="n for the n-gram grounding metric (default 3).",
    )
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--nli-n-pairs",
        type=int,
        default=0,
        help=(
            "Paired subsample size for the optional NLI-entailment "
            "grounding pass. ``0`` (default) skips NLI; pass e.g. "
            "``3000`` to mirror the W6 BERTScore-proxy convention. "
            "Pass a value ``>= n_shared_qids`` to score every paired "
            "qid (slow but exhaustive)."
        ),
    )
    parser.add_argument(
        "--nli-model",
        type=str,
        default="cross-encoder/nli-deberta-v3-small",
        help="HF model id for the NLI cross-encoder.",
    )
    parser.add_argument("--nli-batch-size", type=int, default=16)
    parser.add_argument("--nli-max-length", type=int, default=512)
    parser.add_argument(
        "--nli-device",
        type=str,
        default=None,
        help="Override device for NLI scoring; auto-detect when omitted.",
    )
    parser.add_argument(
        "--nli-subsample-seed",
        type=int,
        default=42,
        help="Seed for the paired NLI subsample; independent of --bootstrap-seed.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

def load_predictions_ordered(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _assert_paired(bm25: list[dict], rerank: list[dict]) -> None:
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


# --------------------------------------------------------------------------- #
# Per-query scoring
# --------------------------------------------------------------------------- #

def score_arm(
    rows: list[dict],
    n: int,
) -> tuple[list[float], list[float], int, int]:
    """Return ``(lex_scores, ngram_scores, n_lex_vacuous, n_ngram_vacuous)``.

    Vacuous counts are reported alongside the means so the audit's
    "no content tokens" / "<n tokens" edge cases cannot silently inflate
    the headline number — see the metric docstrings for the convention.
    """
    lex: list[float] = []
    ngram: list[float] = []
    n_lex_vacuous = 0
    n_ngram_vacuous = 0
    for row in rows:
        pred = row.get("prediction") or ""
        passages = row.get("passages") or []
        lex.append(lexical_grounding(pred, passages))
        ngram.append(ngram_grounding(pred, passages, n=n))
        if is_vacuously_grounded_lex(pred):
            n_lex_vacuous += 1
        if is_vacuously_grounded_ngram(pred, n=n):
            n_ngram_vacuous += 1
    return lex, ngram, n_lex_vacuous, n_ngram_vacuous


# --------------------------------------------------------------------------- #
# Win/tie/loss helper
# --------------------------------------------------------------------------- #

def _win_tie_loss(a: list[float], b: list[float]) -> tuple[float, float, float]:
    n = max(len(a), 1)
    win = sum(1 for x, y in zip(a, b) if y > x) / n
    tie = sum(1 for x, y in zip(a, b) if y == x) / n
    loss = 1.0 - win - tie
    return win, tie, loss


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    bm25 = load_predictions_ordered(args.bm25_dir / "predictions.jsonl")
    rerank = load_predictions_ordered(args.reranked_dir / "predictions.jsonl")
    _assert_paired(bm25, rerank)
    n_shared = len(bm25)
    logger.info("Paired predictions: %d shared qids.", n_shared)

    # ---- per-query scoring (lex + ngram on full pool) ----
    logger.info("Scoring lexical + %d-gram grounding ...", args.ngram_n)
    t0 = time.time()
    bm25_lex, bm25_ngram, bm25_lex_vac, bm25_ngram_vac = score_arm(bm25, args.ngram_n)
    rerank_lex, rerank_ngram, rerank_lex_vac, rerank_ngram_vac = score_arm(
        rerank, args.ngram_n
    )
    scoring_secs = time.time() - t0
    logger.info("  scoring done in %.1f s.", scoring_secs)

    # ---- NLI grounding on a paired subsample ----
    nli_block: dict[str, Any] | None = None
    bm25_nli_per_qid: dict[str, float] = {}
    rerank_nli_per_qid: dict[str, float] = {}
    if args.nli_n_pairs and args.nli_n_pairs > 0:
        n_sub = min(int(args.nli_n_pairs), n_shared)
        if n_sub >= n_shared:
            sampled_indices = list(range(n_shared))
            subsample_label = "full"
        else:
            rng = random.Random(args.nli_subsample_seed)
            sampled_indices = sorted(rng.sample(range(n_shared), n_sub))
            subsample_label = f"random seed={args.nli_subsample_seed}"
        logger.info(
            "NLI paired subsample: %d of %d qids (%s); model=%s",
            n_sub, n_shared, subsample_label, args.nli_model,
        )

        from msmarco_genqa.evaluation.nli_grounding import per_query_nli_entailment

        bm25_preds = [bm25[i].get("prediction") or "" for i in sampled_indices]
        bm25_psgs = [list(bm25[i].get("passages") or []) for i in sampled_indices]
        rerank_preds = [rerank[i].get("prediction") or "" for i in sampled_indices]
        rerank_psgs = [list(rerank[i].get("passages") or []) for i in sampled_indices]

        t_nli = time.time()
        logger.info("  ... BM25 arm NLI scoring ...")
        bm25_nli = per_query_nli_entailment(
            bm25_preds,
            bm25_psgs,
            model_type=args.nli_model,
            batch_size=args.nli_batch_size,
            device=args.nli_device,
            max_length=args.nli_max_length,
        )
        logger.info("  BM25 arm done in %.1f s.", time.time() - t_nli)
        t_nli2 = time.time()
        logger.info("  ... reranked arm NLI scoring ...")
        rerank_nli = per_query_nli_entailment(
            rerank_preds,
            rerank_psgs,
            model_type=args.nli_model,
            batch_size=args.nli_batch_size,
            device=args.nli_device,
            max_length=args.nli_max_length,
        )
        logger.info("  reranked arm done in %.1f s.", time.time() - t_nli2)

        # Paired bootstrap on the NLI delta (rerank − bm25) over the subsample.
        nli_boot = paired_bootstrap_diff(
            bm25_nli,
            rerank_nli,
            n_resamples=args.n_resamples,
            ci=args.ci,
            seed=args.bootstrap_seed,
        )
        nli_win, nli_tie, nli_loss = _win_tie_loss(bm25_nli, rerank_nli)

        # Index per-qid scores so the per_query_grounding.jsonl can carry
        # them on the sampled rows (the rest get null).
        for i, idx in enumerate(sampled_indices):
            qid = str(bm25[idx]["query_id"])
            bm25_nli_per_qid[qid] = float(bm25_nli[i])
            rerank_nli_per_qid[qid] = float(rerank_nli[i])

        nli_block = {
            "model": args.nli_model,
            "n_subsample": n_sub,
            "subsample_label": subsample_label,
            "subsample_seed": int(args.nli_subsample_seed),
            "mean_bm25": nli_boot["mean_a"],
            "mean_rerank": nli_boot["mean_b"],
            "delta_mean": nli_boot["mean_delta"],
            "ci_low": nli_boot["ci_low"],
            "ci_high": nli_boot["ci_high"],
            "p_two_sided": nli_boot["p_two_sided"],
            "win_rate_rerank_strictly_better": float(nli_win),
            "tie_rate": float(nli_tie),
            "loss_rate_bm25_strictly_better": float(nli_loss),
            "bootstrap": {
                "n": nli_boot["n"],
                "n_resamples": nli_boot["n_resamples"],
                "ci": nli_boot["ci"],
                "seed": nli_boot["seed"],
            },
        }

    # ---- per-query JSONL ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_query_path = args.output_dir / "per_query_grounding.jsonl"
    with open(per_query_path, "w") as f:
        for i in range(n_shared):
            qid = str(bm25[i]["query_id"])
            row: dict[str, Any] = {
                "query_id": qid,
                "lex_bm25": float(bm25_lex[i]),
                "lex_rerank": float(rerank_lex[i]),
                "lex_delta": float(rerank_lex[i] - bm25_lex[i]),
                "ngram_bm25": float(bm25_ngram[i]),
                "ngram_rerank": float(rerank_ngram[i]),
                "ngram_delta": float(rerank_ngram[i] - bm25_ngram[i]),
            }
            if nli_block is not None:
                nli_b = bm25_nli_per_qid.get(qid)
                nli_r = rerank_nli_per_qid.get(qid)
                row["nli_bm25"] = nli_b
                row["nli_rerank"] = nli_r
                row["nli_delta"] = (
                    None if (nli_b is None or nli_r is None) else nli_r - nli_b
                )
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %s", per_query_path)

    # ---- bootstrap CIs ----
    logger.info(
        "Paired bootstrap: n_resamples=%d, ci=%.2f, seed=%d",
        args.n_resamples, args.ci, args.bootstrap_seed,
    )
    lex_boot = paired_bootstrap_diff(
        bm25_lex, rerank_lex,
        n_resamples=args.n_resamples, ci=args.ci, seed=args.bootstrap_seed,
    )
    ngram_boot = paired_bootstrap_diff(
        bm25_ngram, rerank_ngram,
        n_resamples=args.n_resamples, ci=args.ci, seed=args.bootstrap_seed,
    )

    # ---- win/tie/loss per metric ----
    lex_win, lex_tie, lex_loss = _win_tie_loss(bm25_lex, rerank_lex)
    ngram_win, ngram_tie, ngram_loss = _win_tie_loss(bm25_ngram, rerank_ngram)

    # ---- summary ----
    summary: dict[str, Any] = {
        "task": "w7_grounding_audit",
        "label": (
            "Lexical content-token + n-gram grounding of W3 generator "
            "predictions to their prompt passages; optional NLI-entailment "
            "grounding on a paired subsample (semantic check). See report "
            "§6 Limitations for the extractiveness-vs-faithfulness split."
        ),
        "inputs": {
            "bm25_predictions": str(
                (args.bm25_dir / "predictions.jsonl").resolve().relative_to(PROJECT_ROOT)
            ),
            "reranked_predictions": str(
                (args.reranked_dir / "predictions.jsonl").resolve().relative_to(PROJECT_ROOT)
            ),
        },
        "n_shared_qids": int(n_shared),
        "ngram_n": int(args.ngram_n),
        "scoring_seconds": round(scoring_secs, 2),
        "metrics": {
            "lexical_content_token_grounding": {
                "mean_bm25": lex_boot["mean_a"],
                "mean_rerank": lex_boot["mean_b"],
                "delta_mean": lex_boot["mean_delta"],
                "ci_low": lex_boot["ci_low"],
                "ci_high": lex_boot["ci_high"],
                "p_two_sided": lex_boot["p_two_sided"],
                "win_rate_rerank_strictly_better": float(lex_win),
                "tie_rate": float(lex_tie),
                "loss_rate_bm25_strictly_better": float(lex_loss),
                "bootstrap": {
                    "n": lex_boot["n"],
                    "n_resamples": lex_boot["n_resamples"],
                    "ci": lex_boot["ci"],
                    "seed": lex_boot["seed"],
                },
            },
            "ngram_grounding": {
                "n": int(args.ngram_n),
                "mean_bm25": ngram_boot["mean_a"],
                "mean_rerank": ngram_boot["mean_b"],
                "delta_mean": ngram_boot["mean_delta"],
                "ci_low": ngram_boot["ci_low"],
                "ci_high": ngram_boot["ci_high"],
                "p_two_sided": ngram_boot["p_two_sided"],
                "win_rate_rerank_strictly_better": float(ngram_win),
                "tie_rate": float(ngram_tie),
                "loss_rate_bm25_strictly_better": float(ngram_loss),
                "bootstrap": {
                    "n": ngram_boot["n"],
                    "n_resamples": ngram_boot["n_resamples"],
                    "ci": ngram_boot["ci"],
                    "seed": ngram_boot["seed"],
                },
            },
            "nli_entailment_grounding": nli_block,
        },
        "edge_cases": {
            # Counts of predictions that fall into the documented "vacuous"
            # path of each metric. The audit reports rates so a reader can
            # judge whether the headline is materially inflated by them.
            "bm25_n_lex_vacuous": int(bm25_lex_vac),
            "bm25_n_ngram_vacuous": int(bm25_ngram_vac),
            "rerank_n_lex_vacuous": int(rerank_lex_vac),
            "rerank_n_ngram_vacuous": int(rerank_ngram_vac),
            "policy": (
                "Vacuous predictions (no content tokens for the lexical "
                "metric, or <n tokens for the n-gram metric) score 1.0 "
                "by convention. See msmarco_genqa.evaluation.grounding docstrings."
            ),
        },
        "notes": (
            "Both metrics are deterministic CPU pure functions over the "
            "lowercased, regex-tokenised prediction and passage texts. "
            "The paired bootstrap resamples (lex_bm25, lex_rerank) and "
            "(ngram_bm25, ngram_rerank) tuples respectively; reported "
            "intervals are percentile intervals."
        ),
    }
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", summary_path)

    # ---- console table ----
    print()
    print("=== W7 grounding audit — paired bootstrap ===")
    print(
        f"shared qids: {n_shared}   "
        f"n_resamples: {args.n_resamples}   seed: {args.bootstrap_seed}"
    )
    print(
        f"  {'metric':28s}  {'bm25':>8s}  {'rerank':>8s}  "
        f"{'Δ':>9s}  {'95% CI':>22s}  {'p2':>8s}"
    )
    for label, boot in (
        ("lexical_content_token", lex_boot),
        (f"ngram (n={args.ngram_n})", ngram_boot),
    ):
        ci_str = f"[{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]"
        print(
            f"  {label:28s}  {boot['mean_a']:>8.4f}  {boot['mean_b']:>8.4f}  "
            f"{boot['mean_delta']:>+9.4f}  {ci_str:>22s}  {boot['p_two_sided']:>8.4f}"
        )
    if nli_block is not None:
        ci_str = f"[{nli_block['ci_low']:+.4f}, {nli_block['ci_high']:+.4f}]"
        label = f"nli (n={nli_block['n_subsample']})"
        print(
            f"  {label:28s}  {nli_block['mean_bm25']:>8.4f}  {nli_block['mean_rerank']:>8.4f}  "
            f"{nli_block['delta_mean']:>+9.4f}  {ci_str:>22s}  {nli_block['p_two_sided']:>8.4f}"
        )
    print()
    print(
        f"  lex   win/tie/loss: {lex_win:.3f} / {lex_tie:.3f} / {lex_loss:.3f}   (rerank − BM25)"
    )
    print(
        f"  ngram win/tie/loss: {ngram_win:.3f} / {ngram_tie:.3f} / {ngram_loss:.3f}   (rerank − BM25)"
    )
    print()
    print(
        f"  vacuous predictions   bm25 lex={bm25_lex_vac}  bm25 ngram={bm25_ngram_vac}  "
        f"rerank lex={rerank_lex_vac}  rerank ngram={rerank_ngram_vac}"
    )
    print()
    print(
        "Reminder: this measures extractiveness, not semantic "
        "faithfulness. See report §6 Limitations."
    )


if __name__ == "__main__":
    main()
