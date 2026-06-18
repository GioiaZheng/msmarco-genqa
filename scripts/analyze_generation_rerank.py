"""Compare BM25-driven vs reranked-driven RAG generation on dev/small.

Inputs (both produced by ``experiments/run_generation_baseline.py`` with
different ``--input-run`` upstreams):

- ``outputs/generation_bm25_full/predictions.jsonl``
- ``outputs/generation_reranked_full/predictions.jsonl``

The two prediction files MUST cover the same set of query_ids (this is
enforced by running both generators on the same eligible-query pool —
see ``--restrict-to-run`` in the generation runner).

Outputs (under ``outputs/generation_analysis/``):

- ``per_query_metrics.jsonl``   — one row per qid with both sides' metrics
- ``summary.json``              — headline metric table + category breakdown
- ``qualitative_examples.json`` — N representative examples per bucket
- ``report.md``                 — human-readable analysis report

Buckets, in priority order (a qid lands in the first one that fits):

1. ``rerank_fixed_generation_improved``
   Reranking pushed a qrel-relevant passage into top-3 that BM25 missed,
   AND the generation token-F1 improved by a meaningful margin.
2. ``rerank_fixed_generation_still_failing``
   Reranking pushed a qrel-relevant passage into top-3 that BM25 missed,
   but generation token-F1 is still poor on both sides.
3. ``retrieval_equivalent_generation_differs``
   Both sides have at least one qrel-relevant passage in top-3 (or neither
   does), but generation token-F1 differs by a meaningful margin.
4. ``regression``
   Reranked generation token-F1 is noticeably WORSE than BM25 — the
   pathological case we should look at.
5. ``no_signal``
   Everything else: both sides similar, or retrieval signal unavailable.

Usage::

    python scripts/analyze_generation_rerank.py \\
        --bm25-dir outputs/generation_bm25_full \\
        --reranked-dir outputs/generation_reranked_full \\
        --output-dir outputs/generation_analysis

The analysis is pure-Python on the prediction files — no model load,
no docs_store, no network beyond ``ms_marco`` for the ``query_type``
metadata (which is cached after the first generation run).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from msmarco_genqa.data.msmarco import load_msmarco_passage
from msmarco_genqa.evaluation.bootstrap import paired_bootstrap_diff
from msmarco_genqa.evaluation.generation import (
    _normalize,
    exact_match,
    token_f1,
)

logger = logging.getLogger("analyze_generation_rerank")


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
        default=PROJECT_ROOT / "outputs/generation_analysis",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data/raw",
    )
    parser.add_argument(
        "--n-qualitative-per-bucket",
        type=int,
        default=5,
        help="Examples to dump per bucket in qualitative_examples.json.",
    )
    parser.add_argument(
        "--f1-improve-threshold",
        type=float,
        default=0.20,
        help="Token-F1 delta to consider a meaningful generation change.",
    )
    parser.add_argument(
        "--f1-failing-threshold",
        type=float,
        default=0.30,
        help="Token-F1 below this is considered 'still failing'.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10000,
        help="Paired-bootstrap resamples for CI on Δ token-F1 / EM. 0 disables.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Reproducibility seed for the paired bootstrap.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Loading + per-query metrics
# --------------------------------------------------------------------------- #


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """Index a predictions.jsonl by query_id."""
    out: dict[str, dict[str, Any]] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            out[str(row["query_id"])] = row
    return out


def per_query_metrics(pred: str, refs: list[str]) -> dict[str, float]:
    """Cheap per-query metrics (no HuggingFace ROUGE/BLEU — those are corpus-level)."""
    return {
        "token_f1": token_f1(pred, refs),
        "exact_match": exact_match(pred, refs),
        "n_pred_tokens": len(_normalize(pred).split()),
        "n_ref_tokens_max": max((len(_normalize(r).split()) for r in refs), default=0),
    }


def has_relevant_in_top3(top_doc_ids: list[str], qrel_set: set[str]) -> bool:
    return any(d in qrel_set for d in top_doc_ids[:3])


# --------------------------------------------------------------------------- #
# Bucketing
# --------------------------------------------------------------------------- #


def assign_bucket(
    bm25_f1: float,
    rerank_f1: float,
    bm25_retrieved: bool,
    rerank_retrieved: bool,
    f1_improve_threshold: float,
    f1_failing_threshold: float,
) -> str:
    """Return the bucket name for a single qid.

    The retrieval flags are best-effort: they require a qrel set for the
    qid. If the qrel is unavailable we fall through to the generation-only
    buckets (regression / equivalent / no_signal).
    """
    delta = rerank_f1 - bm25_f1
    rerank_fixed = (
        bm25_retrieved is False
        and rerank_retrieved is True
    )
    big_improve = delta >= f1_improve_threshold
    big_regress = delta <= -f1_improve_threshold

    if rerank_fixed and big_improve:
        return "rerank_fixed_generation_improved"
    if rerank_fixed and rerank_f1 < f1_failing_threshold:
        return "rerank_fixed_generation_still_failing"
    if big_regress:
        return "regression"
    if abs(delta) >= f1_improve_threshold:
        return "retrieval_equivalent_generation_differs"
    return "no_signal"


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def aggregate_by_key(
    rows: list[dict[str, Any]],
    key_field: str,
    metric_field: str,
) -> dict[str, dict[str, float]]:
    """Compute mean(metric) per distinct value of key_field, plus count."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        k = r.get(key_field, "UNKNOWN")
        if k is None:
            k = "UNKNOWN"
        grouped[k].append(r[metric_field])
    return {
        k: {"mean": _mean(v), "count": len(v)}
        for k, v in grouped.items()
    }


# --------------------------------------------------------------------------- #
# QA query_type
# --------------------------------------------------------------------------- #


def load_query_types(cache_dir: Path) -> dict[str, str]:
    """Map qid -> query_type (DESCRIPTION/NUMERIC/ENTITY/PERSON/LOCATION).

    Pulled from the same MS MARCO QA v2.1 validation split used by the
    generation runner, so this is offline-cached after the first run.
    """
    from datasets import load_dataset  # local import: keeps the script importable for unit tests

    logger.info("Loading MS MARCO QA v2.1 query_type metadata...")
    ds = load_dataset("ms_marco", "v2.1", split="validation")
    out: dict[str, str] = {}
    for row in ds:
        qt = row.get("query_type") or "UNKNOWN"
        out[str(row["query_id"])] = qt
    return out


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _truncate(s: str, n: int = 220) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _render_examples_section(examples: list[dict[str, Any]], bucket: str) -> list[str]:
    if not examples:
        return ["  *(no examples in this bucket)*", ""]
    out: list[str] = []
    for ex in examples:
        out.append(
            f"- **qid {ex['query_id']}** &nbsp; "
            f"*{ex['query_type']}* &nbsp; "
            f"Δtoken-F1 = `{ex['delta_token_f1']:+.3f}` "
            f"(BM25 `{ex['bm25_token_f1']:.3f}` → Rerank `{ex['rerank_token_f1']:.3f}`)  \n"
            f"  Q: *{_truncate(ex['query'], 140)}*  \n"
            f"  REF: *{_truncate((ex['references'] or [''])[0], 140)}*  \n"
            f"  BM25 → *{_truncate(ex['bm25_prediction'], 140)}*  \n"
            f"  Rerank → *{_truncate(ex['rerank_prediction'], 140)}*  \n"
            f"  Retrieval: BM25 top-3 has-rel={ex['bm25_relevant_in_top3']}, "
            f"Rerank top-3 has-rel={ex['rerank_relevant_in_top3']}"
        )
    out.append("")
    return out


def _render_markdown_report(
    *,
    summary: dict[str, Any],
    examples_by_bucket: dict[str, list[dict[str, Any]]],
    bm25_dir: Path,
    reranked_dir: Path,
) -> str:
    import datetime as dt

    headline = summary["headline"]
    n = summary["n_shared_qids"]
    bt, rt = headline["bm25_mean_token_f1"], headline["rerank_mean_token_f1"]
    bem, rem = headline["bm25_mean_exact_match"], headline["rerank_mean_exact_match"]
    bm25_retr = summary["retrieval_flags"]["bm25_relevant_in_top3_rate"]
    rerank_retr = summary["retrieval_flags"]["rerank_relevant_in_top3_rate"]

    lines: list[str] = []
    lines.append("# Generation × Retrieval Source — Full dev/small analysis")
    lines.append("")
    lines.append(
        f"*Auto-generated {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"by `scripts/analyze_generation_rerank.py`.*"
    )
    lines.append("")
    lines.append("## 1. Setup")
    lines.append("")
    lines.append(
        f"- **Eval set**: {n} queries (apples-to-apples — same qids on both "
        "sides, full eligible dev/small pool)."
    )
    lines.append(
        f"- **BM25 generation**: `{bm25_dir.relative_to(PROJECT_ROOT) if bm25_dir.is_relative_to(PROJECT_ROOT) else bm25_dir}/predictions.jsonl`."
    )
    lines.append(
        f"- **Reranked generation**: `{reranked_dir.relative_to(PROJECT_ROOT) if reranked_dir.is_relative_to(PROJECT_ROOT) else reranked_dir}/predictions.jsonl`."
    )
    lines.append(
        "- **Generator**: `t5-small`, no fine-tuning, top-3 passages, "
        "best-of-N reference scoring."
    )
    lines.append(
        "- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` over the W4 "
        "dense top-100."
    )
    lines.append("")

    # ---- Headline table ----
    lines.append("## 2. Headline metrics")
    lines.append("")
    lines.append("| Retrieval source &rarr; T5-small | Token-F1 | Exact-Match | top-3 has-relevant rate |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| BM25     | {bt:.4f} | {bem:.4f} | {_fmt_pct(bm25_retr)} |")
    lines.append(f"| Reranked | **{rt:.4f}** | **{rem:.4f}** | **{_fmt_pct(rerank_retr)}** |")
    lines.append(
        f"| **Δ (rerank − BM25)** | **{rt - bt:+.4f}** | "
        f"**{rem - bem:+.4f}** | **{_fmt_pct(rerank_retr - bm25_retr)}** |"
    )
    lines.append("")
    lines.append(
        f"Strict per-query improvements / regressions / ties: "
        f"**{headline['n_strict_improvements']} / "
        f"{headline['n_strict_regressions']} / "
        f"{headline['n_ties']}** "
        f"(mean Δtoken-F1 = **{headline['mean_delta_token_f1']:+.4f}**)."
    )
    lines.append("")

    boot = summary.get("bootstrap_ci")
    if boot:
        lines.append(
            "**Paired-bootstrap 95% CI on Δ** (rerank − BM25), "
            f"{boot['token_f1']['n_resamples']} resamples, seed "
            f"{boot['token_f1']['seed']}:"
        )
        lines.append("")
        lines.append("| Metric | Δ (per-query mean) | 95% CI | p₂ |")
        lines.append("|---|---:|---:|---:|")
        for name, key in (("Token-F1", "token_f1"), ("Exact-Match", "exact_match")):
            r = boot[key]
            lines.append(
                f"| {name} | {r['mean_delta']:+.4f} | "
                f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
                f"{r['p_two_sided']:.4f} |"
            )
        lines.append("")
        lines.append(
            "*ROUGE-L / BLEU CIs require per-query scorers from "
            "`rouge_score` / NLTK; produce them with "
            "`python scripts/bootstrap_generation_comparison.py`.*"
        )
        lines.append("")

    # ---- By query_type ----
    lines.append("## 3. By MS MARCO QA `query_type`")
    lines.append("")
    lines.append("| query_type | n | BM25 token-F1 | Rerank token-F1 | Δ |")
    lines.append("|---|---:|---:|---:|---:|")
    bm_by = summary["by_query_type"]["bm25_token_f1"]
    rr_by = summary["by_query_type"]["rerank_token_f1"]
    d_by = summary["by_query_type"]["delta_token_f1"]
    for t in sorted(bm_by.keys()):
        lines.append(
            f"| {t} | {bm_by[t]['count']} | {bm_by[t]['mean']:.4f} | "
            f"{rr_by[t]['mean']:.4f} | {d_by[t]['mean']:+.4f} |"
        )
    lines.append("")

    # ---- Buckets ----
    lines.append("## 4. Per-query buckets")
    lines.append("")
    bucket_descriptions = {
        "rerank_fixed_generation_improved":
            "Reranking pushed a qrel-relevant passage into top-3 that BM25 "
            "missed, AND token-F1 improved by ≥{}.".format(summary["thresholds"]["f1_improve"]),
        "rerank_fixed_generation_still_failing":
            "Reranking pushed a qrel-relevant passage into top-3 that BM25 "
            "missed, but the generator's token-F1 is still poor "
            "(<{}). Retrieval was fixed; generation is the bottleneck."
            .format(summary["thresholds"]["f1_failing"]),
        "regression":
            "Reranked generation token-F1 is meaningfully WORSE than BM25 — "
            "the cross-encoder pushed a worse passage into top-3, or "
            "swapped one acceptable passage for another that confuses T5.",
        "retrieval_equivalent_generation_differs":
            "Both sides have similar retrieval signal in top-3, but "
            "generation token-F1 differs by ≥{}. The reranker still "
            "perturbed the order (or the actual passages) enough to "
            "change what T5 attends to.".format(summary["thresholds"]["f1_improve"]),
        "no_signal":
            "Small delta everywhere — no meaningful signal in either direction.",
    }
    lines.append("| Bucket | n | % of eval | Description |")
    lines.append("|---|---:|---:|---|")
    total = summary["n_shared_qids"]
    for bucket, count in sorted(summary["buckets"].items(), key=lambda kv: -kv[1]):
        pct = count / total * 100 if total else 0.0
        desc = bucket_descriptions.get(bucket, "")
        lines.append(f"| `{bucket}` | {count} | {pct:.1f}% | {desc} |")
    lines.append("")

    # ---- Qualitative examples ----
    lines.append("## 5. Qualitative examples")
    lines.append("")
    bucket_titles = {
        "rerank_fixed_generation_improved":
            "5.1 Reranking fixed retrieval *and* lifted generation (success cases)",
        "rerank_fixed_generation_still_failing":
            "5.2 Reranking fixed retrieval but generation still fails (T5-small ceiling)",
        "regression":
            "5.3 Regressions (reranking hurt generation)",
        "retrieval_equivalent_generation_differs":
            "5.4 Retrieval equivalent, generation differs (passage-order sensitivity)",
        "no_signal":
            "5.5 No-signal sample",
    }
    for bucket, title in bucket_titles.items():
        lines.append(f"### {title}")
        lines.append("")
        lines.extend(_render_examples_section(examples_by_bucket.get(bucket, []), bucket))

    # ---- Caveats ----
    lines.append("## 6. Caveats")
    lines.append("")
    lines.append(
        "- **CPU-only local execution.** The reranker run is the binding "
        "cost — ~5–6 h on a 6-core MacBook to score 6,980 queries × top-100. "
        "Resume is supported via `--resume`; a kill costs at most one chunk "
        "(`reranker.chunk_size` queries)."
    )
    lines.append(
        "- **T5-small is frozen.** No fine-tuning; metrics are bounded by "
        "how much pretrained T5-small can extract from short MS MARCO "
        "passages. The `rerank_fixed_generation_still_failing` bucket is "
        "where supervised fine-tuning would help most."
    )
    lines.append(
        "- **`relevant_in_top3` is a coarse retrieval flag.** It checks "
        "whether any of the top-3 passages appears in the qrels — it "
        "doesn't measure passage quality beyond that binary signal."
    )
    lines.append(
        "- **`exact_match` is harsh.** MS MARCO QA reference answers are "
        "free-form sentences; a generator that emits the correct entity "
        "but in a different surrounding sentence still scores 0 on EM. "
        "Token-F1 is the more informative comparator."
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load predictions ----
    bm25 = load_predictions(args.bm25_dir / "predictions.jsonl")
    rerank = load_predictions(args.reranked_dir / "predictions.jsonl")
    shared = sorted(set(bm25) & set(rerank))
    if not shared:
        raise SystemExit("No shared qids — the two prediction files cover disjoint queries.")
    bm25_only = set(bm25) - set(rerank)
    rerank_only = set(rerank) - set(bm25)
    if bm25_only or rerank_only:
        logger.warning(
            "Prediction-file qid sets differ: bm25-only=%d, rerank-only=%d; "
            "restricting analysis to the %d shared qids.",
            len(bm25_only),
            len(rerank_only),
            len(shared),
        )
    logger.info("Analysing %d shared qids.", len(shared))

    # ---- 2. Qrels + query_type ----
    data = load_msmarco_passage(cache_dir=args.cache_dir, load_corpus=False)
    qrels = data.qrels
    query_types = load_query_types(args.cache_dir)

    # ---- 3. Per-query rows ----
    rows: list[dict[str, Any]] = []
    for qid in shared:
        b = bm25[qid]
        r = rerank[qid]
        refs = b["references"]  # same source on both sides; pick one.
        bm = per_query_metrics(b["prediction"], refs)
        rm = per_query_metrics(r["prediction"], refs)
        qrel_set = qrels.get(qid, set())
        bm25_retrieved = has_relevant_in_top3(b["top_doc_ids"], qrel_set) if qrel_set else None
        rerank_retrieved = has_relevant_in_top3(r["top_doc_ids"], qrel_set) if qrel_set else None
        rows.append({
            "query_id": qid,
            "query": b["query"],
            "query_type": query_types.get(qid, "UNKNOWN"),
            "references": refs,
            "bm25_prediction": b["prediction"],
            "rerank_prediction": r["prediction"],
            "bm25_top_doc_ids": b["top_doc_ids"],
            "rerank_top_doc_ids": r["top_doc_ids"],
            "bm25_token_f1": bm["token_f1"],
            "rerank_token_f1": rm["token_f1"],
            "delta_token_f1": rm["token_f1"] - bm["token_f1"],
            "bm25_exact_match": bm["exact_match"],
            "rerank_exact_match": rm["exact_match"],
            "bm25_relevant_in_top3": bm25_retrieved,
            "rerank_relevant_in_top3": rerank_retrieved,
            "n_qrels": len(qrel_set),
        })

    # ---- 4. Bucketing ----
    for r in rows:
        r["bucket"] = assign_bucket(
            bm25_f1=r["bm25_token_f1"],
            rerank_f1=r["rerank_token_f1"],
            bm25_retrieved=bool(r["bm25_relevant_in_top3"]),
            rerank_retrieved=bool(r["rerank_relevant_in_top3"]),
            f1_improve_threshold=args.f1_improve_threshold,
            f1_failing_threshold=args.f1_failing_threshold,
        )

    # ---- 5. Write per_query_metrics.jsonl + buckets.jsonl ----
    per_query_path = args.output_dir / "per_query_metrics.jsonl"
    with open(per_query_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows to %s", len(rows), per_query_path)

    # ---- 6. Aggregate metrics ----
    summary: dict[str, Any] = {
        "n_shared_qids": len(shared),
        "n_bm25_only": len(bm25_only),
        "n_rerank_only": len(rerank_only),
        "thresholds": {
            "f1_improve": args.f1_improve_threshold,
            "f1_failing": args.f1_failing_threshold,
        },
        "headline": {
            "bm25_mean_token_f1": _mean([r["bm25_token_f1"] for r in rows]),
            "rerank_mean_token_f1": _mean([r["rerank_token_f1"] for r in rows]),
            "bm25_mean_exact_match": _mean([r["bm25_exact_match"] for r in rows]),
            "rerank_mean_exact_match": _mean([r["rerank_exact_match"] for r in rows]),
            "mean_delta_token_f1": _mean([r["delta_token_f1"] for r in rows]),
            "n_strict_improvements": sum(1 for r in rows if r["delta_token_f1"] > 0),
            "n_strict_regressions": sum(1 for r in rows if r["delta_token_f1"] < 0),
            "n_ties": sum(1 for r in rows if r["delta_token_f1"] == 0),
        },
        "by_query_type": {
            "bm25_token_f1": aggregate_by_key(rows, "query_type", "bm25_token_f1"),
            "rerank_token_f1": aggregate_by_key(rows, "query_type", "rerank_token_f1"),
            "delta_token_f1": aggregate_by_key(rows, "query_type", "delta_token_f1"),
        },
        "buckets": dict(Counter(r["bucket"] for r in rows)),
        "retrieval_flags": {
            "bm25_relevant_in_top3_rate": _mean(
                [1.0 if r["bm25_relevant_in_top3"] else 0.0 for r in rows if r["bm25_relevant_in_top3"] is not None]
            ),
            "rerank_relevant_in_top3_rate": _mean(
                [1.0 if r["rerank_relevant_in_top3"] else 0.0 for r in rows if r["rerank_relevant_in_top3"] is not None]
            ),
            "n_with_qrel": sum(1 for r in rows if r["bm25_relevant_in_top3"] is not None),
        },
    }

    # ---- 6b. Paired bootstrap CI on Δ token-F1 / EM (paired by index) ----
    if args.bootstrap_resamples > 0 and rows:
        logger.info(
            "Paired-bootstrap CI on Δ token-F1 / Δ exact-match (n_resamples=%d, seed=%d)...",
            args.bootstrap_resamples,
            args.bootstrap_seed,
        )
        bm25_tf1 = [r["bm25_token_f1"] for r in rows]
        rerank_tf1 = [r["rerank_token_f1"] for r in rows]
        bm25_em = [r["bm25_exact_match"] for r in rows]
        rerank_em = [r["rerank_exact_match"] for r in rows]
        summary["bootstrap_ci"] = {
            "token_f1": paired_bootstrap_diff(
                bm25_tf1,
                rerank_tf1,
                n_resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed,
            ),
            "exact_match": paired_bootstrap_diff(
                bm25_em,
                rerank_em,
                n_resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed,
            ),
            "notes": (
                "Paired bootstrap on per-query (bm25, rerank) Token-F1 and "
                "Exact-Match. ROUGE-L / BLEU CIs are produced by the "
                "standalone scripts/bootstrap_generation_comparison.py script, "
                "which adds the rouge_score and NLTK per-query scorers."
            ),
        }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Wrote summary to %s", args.output_dir / "summary.json")

    # ---- 7. Qualitative examples per bucket ----
    examples_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    # Sort within each bucket for stability + representativeness:
    # rerank-improved buckets sort by largest +delta; failing buckets sort by lowest rerank F1;
    # regressions sort by most-negative delta; "differs" by largest |delta|.
    bucket_sort_key = {
        "rerank_fixed_generation_improved": lambda r: -r["delta_token_f1"],
        "rerank_fixed_generation_still_failing": lambda r: r["rerank_token_f1"],
        "retrieval_equivalent_generation_differs": lambda r: -abs(r["delta_token_f1"]),
        "regression": lambda r: r["delta_token_f1"],
        "no_signal": lambda r: -abs(r["delta_token_f1"]),
    }
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_bucket[r["bucket"]].append(r)
    for bucket, items in by_bucket.items():
        items.sort(key=bucket_sort_key.get(bucket, lambda r: 0))
        examples_by_bucket[bucket] = items[: args.n_qualitative_per_bucket]

    examples_path = args.output_dir / "qualitative_examples.json"
    with open(examples_path, "w") as f:
        json.dump(examples_by_bucket, f, indent=2, ensure_ascii=False)
    logger.info("Wrote qualitative examples to %s", examples_path)

    # ---- 8. Markdown report ----
    report_md = _render_markdown_report(
        summary=summary,
        examples_by_bucket=examples_by_bucket,
        bm25_dir=args.bm25_dir,
        reranked_dir=args.reranked_dir,
    )
    report_path = args.output_dir / "report.md"
    report_path.write_text(report_md)
    logger.info("Wrote markdown report to %s", report_path)

    # ---- 9. Friendly console summary ----
    print("\n=== Generation × retrieval-source comparison ===")
    print(f"shared qids:                  {len(shared)}")
    print(f"BM25  mean token-F1:          {summary['headline']['bm25_mean_token_f1']:.4f}")
    print(f"Rerank mean token-F1:         {summary['headline']['rerank_mean_token_f1']:.4f}")
    print(f"Δ (mean token-F1):            "
          f"{summary['headline']['mean_delta_token_f1']:+.4f}")
    print(
        f"strict improvements / regressions / ties: "
        f"{summary['headline']['n_strict_improvements']} / "
        f"{summary['headline']['n_strict_regressions']} / "
        f"{summary['headline']['n_ties']}"
    )
    print()
    print("by query_type (mean token-F1):")
    types = sorted(summary["by_query_type"]["bm25_token_f1"].keys())
    print(f"  {'type':12s}  {'n':>5s}  {'bm25':>8s}  {'rerank':>8s}  {'Δ':>8s}")
    for t in types:
        n = summary["by_query_type"]["bm25_token_f1"][t]["count"]
        b = summary["by_query_type"]["bm25_token_f1"][t]["mean"]
        r = summary["by_query_type"]["rerank_token_f1"][t]["mean"]
        d = summary["by_query_type"]["delta_token_f1"][t]["mean"]
        print(f"  {t:12s}  {n:5d}  {b:>8.4f}  {r:>8.4f}  {d:>+8.4f}")
    print()
    print("buckets:")
    for bucket, n in sorted(summary["buckets"].items(), key=lambda kv: -kv[1]):
        print(f"  {bucket:50s}  {n:6d}")
    print()
    print(
        f"top-3 'has relevant qrel' rate: "
        f"BM25 {summary['retrieval_flags']['bm25_relevant_in_top3_rate']:.3f} → "
        f"Rerank {summary['retrieval_flags']['rerank_relevant_in_top3_rate']:.3f}"
    )
    print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
