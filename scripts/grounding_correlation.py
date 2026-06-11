"""Correlate W7 per-query grounding scores with downstream Token-F1
and BERTScore — answer "does higher grounding go with a higher metric?"

W7 established that both arms sit at the lexical / 3-gram grounding
*ceiling* — 97 % of queries score lex = 1.0; 89-92 % score 3-gram = 1.0.
The teacher's follow-up question for W7-C is whether, on the residual
non-ceiling mass, *higher grounding* tracks *higher downstream metric*
at the per-query level. Two cuts:

1. Rank / linear correlations (Spearman ρ, Pearson r) between each
   per-query grounding score and each per-query downstream metric,
   computed **within each arm** (BM25 / rerank).

2. Bin comparison (grounding ≥ 0.9 vs < 0.9): mean downstream by bin,
   diff, Mann-Whitney p, rank-biserial r effect size. This is the more
   robust read for lexical grounding, where the rank correlation is
   dominated by ties at 1.0.

Inputs are all already on disk except BERTScore-F1, which is computed
here (DistilBERT-base, ``rescale_with_baseline=True``, matching the W6
BERTScore-proxy convention) and cached to disk so subsequent runs are
instant.

Outputs (gitignored under ``outputs/W7_grounding_correlation/``):

- ``cache_bertscore_full.jsonl`` — per-qid BERTScore-F1 for both arms;
                                    cache key for repeat runs and W7-A
                                    later. Auto-skipped if present.
- ``per_query_joined.jsonl``    — per-qid grounding + Token-F1 +
                                    BERTScore-F1 (both arms).
- ``summary.json``              — correlations + binned comparisons.
- ``summary.md``                — human-readable table.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("grounding_correlation")


# Tuples drive every table: (grounding_metric_field, downstream_metric_field,
# label_for_report).
GROUNDING_METRICS: tuple[tuple[str, str], ...] = (
    ("lex", "Lexical content-token grounding"),
    ("ngram", "3-gram grounding"),
    ("nli", "NLI entailment grounding"),
)
DOWNSTREAM_METRICS: tuple[tuple[str, str], ...] = (
    ("token_f1", "Token-F1"),
    ("bertscore_f1", "BERTScore-F1"),
)
ARMS: tuple[tuple[str, str], ...] = (
    ("bm25", "BM25"),
    ("rerank", "Reranked"),
)
HIGH_THRESHOLD = 0.9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--per-query-grounding",
        type=Path,
        default=PROJECT_ROOT / "outputs/W7_grounding/per_query_grounding.jsonl",
    )
    p.add_argument(
        "--per-query-metrics",
        type=Path,
        default=PROJECT_ROOT / "outputs/W6_analysis/per_query_metrics.jsonl",
    )
    p.add_argument(
        "--bm25-predictions",
        type=Path,
        default=PROJECT_ROOT / "outputs/W3_generation_bm25_full/predictions.jsonl",
    )
    p.add_argument(
        "--rerank-predictions",
        type=Path,
        default=PROJECT_ROOT / "outputs/W3_generation_reranked_full/predictions.jsonl",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/W7_grounding_correlation",
    )
    p.add_argument(
        "--bertscore-model",
        type=str,
        default="distilbert-base-uncased",
        help="HF model id for BERTScore (matches W6 proxy default).",
    )
    p.add_argument(
        "--bertscore-batch-size",
        type=int,
        default=32,
    )
    p.add_argument(
        "--no-rescale-with-baseline",
        action="store_true",
        help="Disable BERTScore baseline rescaling (default: rescale on).",
    )
    p.add_argument(
        "--force-bertscore",
        action="store_true",
        help="Ignore on-disk BERTScore cache and rescore.",
    )
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def predictions_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(r["query_id"]): {
            "prediction": r.get("prediction") or "",
            "references": list(r.get("references") or []),
        }
        for r in rows
    }


def compute_bertscore_or_load_cache(
    *,
    cache_path: Path,
    qids: list[str],
    bm25_idx: dict[str, dict[str, Any]],
    rerank_idx: dict[str, dict[str, Any]],
    model_type: str,
    batch_size: int,
    rescale_with_baseline: bool,
    force: bool,
) -> dict[str, dict[str, float]]:
    if cache_path.exists() and not force:
        logger.info("Using BERTScore cache at %s (pass --force-bertscore to override).", cache_path)
        out: dict[str, dict[str, float]] = {}
        for r in load_jsonl(cache_path):
            out[str(r["query_id"])] = {
                "bm25_f1": float(r["bm25_f1"]),
                "rerank_f1": float(r["rerank_f1"]),
            }
        if len(out) != len(qids):
            logger.warning(
                "Cache has %d qids but %d expected — falling back to rescore.",
                len(out), len(qids),
            )
        else:
            return out

    from msmarco_genqa.evaluation.bertscore import per_query_bertscore_f1

    bm25_preds = [bm25_idx[q]["prediction"] for q in qids]
    rerank_preds = [rerank_idx[q]["prediction"] for q in qids]
    refs = [bm25_idx[q]["references"] for q in qids]
    logger.info(
        "Scoring per-query BERTScore-F1 on %d qids × 2 arms (model=%s, rescale=%s) ...",
        len(qids), model_type, rescale_with_baseline,
    )
    bm25_scores = per_query_bertscore_f1(
        bm25_preds,
        refs,
        model_type=model_type,
        rescale_with_baseline=rescale_with_baseline,
        batch_size=batch_size,
        verbose=False,
    )
    rerank_scores = per_query_bertscore_f1(
        rerank_preds,
        refs,
        model_type=model_type,
        rescale_with_baseline=rescale_with_baseline,
        batch_size=batch_size,
        verbose=False,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        for qid, b, r in zip(qids, bm25_scores, rerank_scores):
            f.write(json.dumps({"query_id": qid, "bm25_f1": float(b), "rerank_f1": float(r)}) + "\n")
    logger.info("Wrote BERTScore cache to %s (%d rows).", cache_path, len(qids))

    return {
        qid: {"bm25_f1": float(b), "rerank_f1": float(r)}
        for qid, b, r in zip(qids, bm25_scores, rerank_scores)
    }


def correlations(
    grounding: Sequence[float],
    downstream: Sequence[float],
) -> dict[str, float]:
    """Spearman ρ + Pearson r on two paired arrays."""
    from scipy.stats import pearsonr, spearmanr

    n = len(grounding)
    if n < 3:
        return {
            "n": n,
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
        }
    sp = spearmanr(grounding, downstream)
    pr = pearsonr(grounding, downstream)
    return {
        "n": n,
        "spearman_rho": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
    }


def bin_compare(
    grounding: Sequence[float],
    downstream: Sequence[float],
    *,
    threshold: float,
) -> dict[str, float]:
    """Split downstream by grounding ≥ threshold vs < threshold; return
    per-bin means and a two-sided Mann-Whitney U + rank-biserial r.
    """
    from scipy.stats import mannwhitneyu

    high = [d for g, d in zip(grounding, downstream) if g >= threshold]
    low = [d for g, d in zip(grounding, downstream) if g < threshold]
    n_high, n_low = len(high), len(low)

    def safe_mean(xs: Sequence[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    mean_high = safe_mean(high)
    mean_low = safe_mean(low)
    if n_high < 1 or n_low < 1:
        return {
            "threshold": threshold,
            "n_high": n_high,
            "n_low": n_low,
            "mean_high": mean_high,
            "mean_low": mean_low,
            "mean_diff_high_minus_low": mean_high - mean_low,
            "mannwhitney_U": 0.0,
            "mannwhitney_p_two_sided": 1.0,
            "rank_biserial_r": 0.0,
        }
    res = mannwhitneyu(high, low, alternative="two-sided")
    U = float(res.statistic)
    p = float(res.pvalue)
    r = 2.0 * U / (n_high * n_low) - 1.0
    return {
        "threshold": threshold,
        "n_high": n_high,
        "n_low": n_low,
        "mean_high": mean_high,
        "mean_low": mean_low,
        "mean_diff_high_minus_low": mean_high - mean_low,
        "mannwhitney_U": U,
        "mannwhitney_p_two_sided": p,
        "rank_biserial_r": float(r),
    }


def render_markdown(
    *,
    n_total: int,
    summary: dict[str, Any],
    bertscore_cache: Path,
    bertscore_model: str,
) -> str:
    lines: list[str] = []
    lines.append("# Grounding vs downstream metrics — W7-C")
    lines.append("")
    lines.append(
        f"Per-query correlation of W7 grounding scores against W6 Token-F1 "
        f"and per-query BERTScore-F1, computed on {n_total} paired qids. "
        "Two cuts: rank/linear correlation and binned mean (grounding "
        f"≥{HIGH_THRESHOLD:.1f} vs <{HIGH_THRESHOLD:.1f}). BERTScore: "
        f"`{bertscore_model}`, rescale_with_baseline=True (cache at "
        f"`{bertscore_cache.relative_to(PROJECT_ROOT)}`)."
    )
    lines.append("")
    lines.append("## 1. Correlations within each arm")
    lines.append("")
    lines.append(
        "Spearman ρ on the heavily-tied grounding distributions (lex: "
        "~97 % of queries score 1.0; 3-gram: ~90 %). |ρ| < 0.05 is "
        "essentially the ceiling effect; the binned table in §2 is the "
        "load-bearing read."
    )
    lines.append("")
    lines.append("| arm | grounding | downstream | n | Spearman ρ | ρ p | Pearson r | r p |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for arm_key, arm_label in ARMS:
        for g_key, g_label in GROUNDING_METRICS:
            for d_key, d_label in DOWNSTREAM_METRICS:
                cell = summary["correlations"][arm_key][g_key][d_key]
                if cell["n"] == 0:
                    lines.append(
                        f"| {arm_label} | {g_label} | {d_label} | "
                        "0 | — | — | — | — |"
                    )
                else:
                    lines.append(
                        f"| {arm_label} | {g_label} | {d_label} | "
                        f"{cell['n']} | "
                        f"{cell['spearman_rho']:+.3f} | "
                        f"{cell['spearman_p']:.3g} | "
                        f"{cell['pearson_r']:+.3f} | "
                        f"{cell['pearson_p']:.3g} |"
                    )
    lines.append("")

    lines.append(f"## 2. Binned mean — grounding ≥ {HIGH_THRESHOLD:.1f} vs < {HIGH_THRESHOLD:.1f}")
    lines.append("")
    lines.append(
        "| arm | grounding | downstream | n high | n low | mean high | mean low | "
        "Δ (high−low) | p | r |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    notable: list[tuple[str, str, str, dict[str, float]]] = []
    for arm_key, arm_label in ARMS:
        for g_key, g_label in GROUNDING_METRICS:
            for d_key, d_label in DOWNSTREAM_METRICS:
                cell = summary["binned"][arm_key][g_key][d_key]
                if cell["n_high"] == 0 and cell["n_low"] == 0:
                    lines.append(
                        f"| {arm_label} | {g_label} | {d_label} | "
                        "0 | 0 | — | — | — | — | — |"
                    )
                    continue
                lines.append(
                    f"| {arm_label} | {g_label} | {d_label} | "
                    f"{cell['n_high']} | {cell['n_low']} | "
                    f"{cell['mean_high']:.4f} | {cell['mean_low']:.4f} | "
                    f"**{cell['mean_diff_high_minus_low']:+.4f}** | "
                    f"{cell['mannwhitney_p_two_sided']:.3g} | "
                    f"{cell['rank_biserial_r']:+.3f} |"
                )
                if (
                    cell["n_high"] >= 30
                    and cell["n_low"] >= 30
                    and cell["mannwhitney_p_two_sided"] < 0.05
                ):
                    notable.append((arm_label, g_label, d_label, cell))
    lines.append("")

    lines.append("**Headline read.**")
    lines.append("")
    if notable:
        for arm_label, g_label, d_label, cell in notable:
            direction = "higher" if cell["mean_diff_high_minus_low"] > 0 else "lower"
            lines.append(
                f"- {arm_label} arm: {d_label} is significantly **{direction}** "
                f"on queries with {g_label} ≥{HIGH_THRESHOLD:.1f} "
                f"(Δ = {cell['mean_diff_high_minus_low']:+.4f}, "
                f"p = {cell['mannwhitney_p_two_sided']:.3g}, r = "
                f"{cell['rank_biserial_r']:+.3f}; n_high={cell['n_high']}, "
                f"n_low={cell['n_low']})."
            )
    else:
        lines.append(
            "- No (arm × grounding × downstream) cell crosses p < 0.05 with "
            "both bins ≥ 30 queries — at this dataset's grounding ceiling, "
            "the residual low-grounding mass is too small / too noisy to "
            "produce a statistically detectable downstream lift."
        )
    lines.append("")

    lines.append("## 3. Caveats")
    lines.append("")
    lines.append(
        "- Lex / 3-gram grounding scores are heavily tied at 1.0 (97 % "
        "lex, 89-92 % 3-gram). Spearman ρ on a near-constant variable is "
        "structurally small; the bin comparison is the right read."
    )
    lines.append(
        "- Lexical grounding's low-bin has only ~40 queries per arm; "
        "Mann-Whitney p is wide and the result is suggestive, not "
        "conclusive."
    )
    lines.append(
        "- NLI grounding (W7-A) lives on a 3,000-paired-qid subsample, "
        "so the per-arm correlation is on n=3,000 (not 6,980). The "
        "≥0.9 bin is *flipped* in cardinality vs lex / 3-gram — NLI "
        "scores concentrate well below 0.9 for T5-small on this prompt "
        "format, so n_high is small and n_low is large."
    )
    lines.append(
        "- BERTScore here is the W6-proxy DistilBERT setup, not the "
        "citation-grade roberta-large. The correlation pattern should "
        "transfer, the absolute level will shift."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    logger.info("Loading per-query grounding from %s ...", args.per_query_grounding)
    grounding_rows = load_jsonl(args.per_query_grounding)
    grounding_of = {str(r["query_id"]): r for r in grounding_rows}

    logger.info("Loading per-query metrics from %s ...", args.per_query_metrics)
    metrics_rows = load_jsonl(args.per_query_metrics)
    metrics_of = {str(r["query_id"]): r for r in metrics_rows}

    logger.info("Loading W3 predictions (BM25 + reranked) ...")
    bm25_idx = predictions_index(load_jsonl(args.bm25_predictions))
    rerank_idx = predictions_index(load_jsonl(args.rerank_predictions))

    # Restrict to qids that appear in all three sources.
    qids = sorted(
        set(grounding_of) & set(metrics_of) & set(bm25_idx) & set(rerank_idx),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    if not qids:
        raise SystemExit("No qids common to grounding, metrics, and predictions.")
    logger.info("Joining on %d common qids.", len(qids))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "cache_bertscore_full.jsonl"
    bertscore_of = compute_bertscore_or_load_cache(
        cache_path=cache_path,
        qids=qids,
        bm25_idx=bm25_idx,
        rerank_idx=rerank_idx,
        model_type=args.bertscore_model,
        batch_size=args.bertscore_batch_size,
        rescale_with_baseline=not args.no_rescale_with_baseline,
        force=args.force_bertscore,
    )

    # Build the per-qid frame (long-ish; one row per qid). NLI columns are
    # nullable: when the W7-A audit was run with --nli-n-pairs < n_shared,
    # only the sampled qids carry NLI values; the rest are None and are
    # filtered out per-(grounding × downstream) cell below.
    def _opt_float(x: Any) -> float | None:
        return None if x is None else float(x)

    rows_out: list[dict[str, Any]] = []
    for qid in qids:
        g = grounding_of[qid]
        m = metrics_of[qid]
        b = bertscore_of[qid]
        rows_out.append(
            {
                "query_id": qid,
                "lex_bm25": float(g["lex_bm25"]),
                "lex_rerank": float(g["lex_rerank"]),
                "ngram_bm25": float(g["ngram_bm25"]),
                "ngram_rerank": float(g["ngram_rerank"]),
                "nli_bm25": _opt_float(g.get("nli_bm25")),
                "nli_rerank": _opt_float(g.get("nli_rerank")),
                "token_f1_bm25": float(m["bm25_token_f1"]),
                "token_f1_rerank": float(m["rerank_token_f1"]),
                "bertscore_f1_bm25": float(b["bm25_f1"]),
                "bertscore_f1_rerank": float(b["rerank_f1"]),
            }
        )

    out_jsonl = args.output_dir / "per_query_joined.jsonl"
    with open(out_jsonl, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    logger.info("Wrote %s (%d rows)", out_jsonl, len(rows_out))

    # ---- correlations + binned ----
    # NLI columns may be None on rows outside the W7-A subsample; for the
    # NLI grounding metric we filter to qids with both NLI and the
    # downstream metric present. Lex / ngram are always present on every
    # qid so their slice is the full 6,980.
    by_arm_corr: dict[str, Any] = {}
    by_arm_bins: dict[str, Any] = {}
    for arm_key, _arm_label in ARMS:
        by_arm_corr[arm_key] = {}
        by_arm_bins[arm_key] = {}
        for g_key, _ in GROUNDING_METRICS:
            by_arm_corr[arm_key][g_key] = {}
            by_arm_bins[arm_key][g_key] = {}
            for d_key, _ in DOWNSTREAM_METRICS:
                paired = [
                    (r[f"{g_key}_{arm_key}"], r[f"{d_key}_{arm_key}"])
                    for r in rows_out
                    if r.get(f"{g_key}_{arm_key}") is not None
                    and r.get(f"{d_key}_{arm_key}") is not None
                ]
                if paired:
                    grounding_vec = [p[0] for p in paired]
                    ds_vec = [p[1] for p in paired]
                    by_arm_corr[arm_key][g_key][d_key] = correlations(grounding_vec, ds_vec)
                    by_arm_bins[arm_key][g_key][d_key] = bin_compare(
                        grounding_vec, ds_vec, threshold=HIGH_THRESHOLD,
                    )
                else:
                    by_arm_corr[arm_key][g_key][d_key] = {
                        "n": 0,
                        "spearman_rho": float("nan"),
                        "spearman_p": float("nan"),
                        "pearson_r": float("nan"),
                        "pearson_p": float("nan"),
                    }
                    by_arm_bins[arm_key][g_key][d_key] = {
                        "threshold": HIGH_THRESHOLD,
                        "n_high": 0,
                        "n_low": 0,
                        "mean_high": 0.0,
                        "mean_low": 0.0,
                        "mean_diff_high_minus_low": 0.0,
                        "mannwhitney_U": 0.0,
                        "mannwhitney_p_two_sided": 1.0,
                        "rank_biserial_r": 0.0,
                    }

    # Coverage of the low-grounding bin per (arm, grounding) — informative
    # for the caveats section. Null values (NLI on un-sampled qids) are
    # skipped, so the lex / ngram counts are over all 6,980 qids and the
    # NLI count is over the sampled subset only.
    low_counts: dict[str, dict[str, int]] = {
        arm_key: {
            g_key: sum(
                1
                for r in rows_out
                if r.get(f"{g_key}_{arm_key}") is not None
                and r[f"{g_key}_{arm_key}"] < HIGH_THRESHOLD
            )
            for g_key, _ in GROUNDING_METRICS
        }
        for arm_key, _ in ARMS
    }

    summary = {
        "task": "grounding_correlation",
        "n_total": len(rows_out),
        "high_threshold": HIGH_THRESHOLD,
        "bertscore_model": args.bertscore_model,
        "bertscore_rescale_with_baseline": not args.no_rescale_with_baseline,
        "correlations": by_arm_corr,
        "binned": by_arm_bins,
        "low_bin_counts": low_counts,
        "notes": (
            "Spearman / Pearson on heavily tied grounding distributions "
            "are dominated by the ceiling effect; the binned ≥0.9 vs <0.9 "
            "Mann-Whitney is the load-bearing read. Lex bins are very "
            "imbalanced (~40 low-grounding queries per arm); 3-gram bins "
            "are more usable (~200-290 low-grounding queries per arm)."
        ),
    }
    summary_json = args.output_dir / "summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", summary_json)

    md = render_markdown(
        n_total=len(rows_out),
        summary=summary,
        bertscore_cache=cache_path,
        bertscore_model=args.bertscore_model,
    )
    md_path = args.output_dir / "summary.md"
    md_path.write_text(md)
    logger.info("Wrote %s", md_path)

    # ---- console summary ----
    print()
    print(f"=== Grounding ↔ downstream (n={len(rows_out)}) — binned mean diff ===")
    print(f"  {'arm':8s}  {'grounding':10s}  {'downstream':14s}  {'n_high':>6s}  {'n_low':>5s}  {'Δ':>8s}  {'p':>8s}  {'r':>7s}")
    for arm_key, arm_label in ARMS:
        for g_key, _ in GROUNDING_METRICS:
            for d_key, d_label in DOWNSTREAM_METRICS:
                c = by_arm_bins[arm_key][g_key][d_key]
                print(
                    f"  {arm_label:8s}  {g_key:10s}  {d_label:14s}  "
                    f"{c['n_high']:>6d}  {c['n_low']:>5d}  "
                    f"{c['mean_diff_high_minus_low']:>+8.4f}  "
                    f"{c['mannwhitney_p_two_sided']:>8.3g}  "
                    f"{c['rank_biserial_r']:>+7.3f}"
                )


if __name__ == "__main__":
    main()
