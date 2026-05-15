"""W7 grounding audit on existing W3 paired predictions.

For each shared query_id between the BM25-fed and reranked-fed
generation runs, compute two grounding scores on the *exact* top-K
passages the generator saw at prompt time (they are persisted inside
``predictions.jsonl`` alongside the model output):

- lexical content-token grounding (``src.evaluation.grounding.lexical_grounding``)
- 3-gram grounding (``src.evaluation.grounding.ngram_grounding``)

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
        --bm25-dir outputs/week03_generation_bm25_full \\
        --reranked-dir outputs/week03_generation_reranked_full \\
        --output-dir outputs/week07_grounding

The defaults point at the canonical full-dev W3 outputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.bootstrap import paired_bootstrap_diff  # noqa: E402
from src.evaluation.grounding import (  # noqa: E402
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
        default=PROJECT_ROOT / "outputs/week03_generation_bm25_full",
    )
    parser.add_argument(
        "--reranked-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/week03_generation_reranked_full",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/week07_grounding",
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
            "Reserved for the optional NLI-proxy subsample. The first W7 "
            "commit ships lexical + n-gram only; passing a non-zero value "
            "here is currently a no-op and a warning is logged."
        ),
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

    if args.nli_n_pairs:
        logger.warning(
            "--nli-n-pairs=%d ignored: NLI proxy is not in this commit. "
            "See reports/templates/week07_grounding.md §7 Next.",
            args.nli_n_pairs,
        )

    bm25 = load_predictions_ordered(args.bm25_dir / "predictions.jsonl")
    rerank = load_predictions_ordered(args.reranked_dir / "predictions.jsonl")
    _assert_paired(bm25, rerank)
    n_shared = len(bm25)
    logger.info("Paired predictions: %d shared qids.", n_shared)

    # ---- per-query scoring ----
    logger.info("Scoring lexical + %d-gram grounding ...", args.ngram_n)
    t0 = time.time()
    bm25_lex, bm25_ngram, bm25_lex_vac, bm25_ngram_vac = score_arm(bm25, args.ngram_n)
    rerank_lex, rerank_ngram, rerank_lex_vac, rerank_ngram_vac = score_arm(
        rerank, args.ngram_n
    )
    scoring_secs = time.time() - t0
    logger.info("  scoring done in %.1f s.", scoring_secs)

    # ---- per-query JSONL ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_query_path = args.output_dir / "per_query_grounding.jsonl"
    with open(per_query_path, "w") as f:
        for i in range(n_shared):
            qid = str(bm25[i]["query_id"])
            row = {
                "query_id": qid,
                "lex_bm25": float(bm25_lex[i]),
                "lex_rerank": float(rerank_lex[i]),
                "lex_delta": float(rerank_lex[i] - bm25_lex[i]),
                "ngram_bm25": float(bm25_ngram[i]),
                "ngram_rerank": float(rerank_ngram[i]),
                "ngram_delta": float(rerank_ngram[i] - bm25_ngram[i]),
            }
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
            "predictions to their prompt passages. Measures extractiveness, "
            "not semantic faithfulness; see report §6 Limitations."
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
                "by convention. See src.evaluation.grounding docstrings."
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
