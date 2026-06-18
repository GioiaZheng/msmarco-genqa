"""Compare cross-encoder rerank Δ across two first stages — W5-A.

Tests the teacher's hypothesis: *does the reranker recover more from
a weaker first stage?* The W5 dense+rerank result already showed
rerank lifting MRR@10 from 0.883 → 0.930 on the qrels-anchored 50k
sample. W5-A reruns the same cross-encoder on BM25 top-100 (full
8.8M corpus) — a much weaker first stage (baseline MRR@10 ≈ 0.170) —
and compares Δ side-by-side.

Inputs (all produced by ``experiments/run_reranker.py``):

- ``outputs/cross_encoder_rerank_full/metrics.json``       — dense+rerank
                                                        (W4 sampled).
- ``outputs/cross_encoder_rerank_bm25_full/metrics.json``  — BM25+rerank
                                                        (W2 full-corpus).
- ``outputs/bm25_baseline/metrics.json``                — BM25 baseline
                                                        full-corpus
                                                        (for cross-check).

Two cuts:

- **Absolute Δ** (rerank − first stage) on MRR@10 / nDCG@10 /
  Recall@100. Same scale as the W5 headline table.
- **Recovery rate** Δ / (1 − first_stage). The fraction of the gap
  to a perfect 1.0 that the reranker closes. Puts the "weaker first
  stage has more room" claim on a fair footing.

The two arms aren't strictly apples-to-apples because of the sampling
asymmetry (dense on 50k qrels-anchored sample vs BM25 on full 8.8M
corpus) — the script documents this in the markdown caveats. The
within-arm Δ rerank − first-stage is still well-defined inside each
arm because both use the same first-stage and same reranker.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("compare_rerank_first_stages")


METRICS: tuple[tuple[str, str], ...] = (
    ("mrr@10", "MRR@10"),
    ("ndcg@10", "nDCG@10"),
    ("recall@100", "Recall@100"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bm25-rerank-metrics",
        type=Path,
        default=PROJECT_ROOT / "outputs/cross_encoder_rerank_bm25_full/metrics.json",
    )
    p.add_argument(
        "--dense-rerank-metrics",
        type=Path,
        default=PROJECT_ROOT / "outputs/cross_encoder_rerank_full/metrics.json",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/rerank_first_stage_compare",
    )
    return p.parse_args()


def load_metrics_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(
            f"Missing metrics file: {path}\n"
            "Run the corresponding experiments/run_reranker.py first."
        )
    with open(path) as f:
        return json.load(f)


def extract_arm(
    rerank_metrics: dict[str, Any],
    first_stage_label: str,
) -> dict[str, Any]:
    """Pull (first_stage, rerank) per-metric pairs out of a metrics.json.

    The run_reranker.py output has the shape::

        {"metrics": {"<first_stage>": {"mrr@10": ..., "ndcg@10": ...,
                                       "recall@100": ...},
                     "rerank":        {...same keys...}}}

    The first-stage key is set by ``run_reranker.py`` and is currently
    hard-coded as ``"dense"`` regardless of the actual input run. We
    therefore accept either the *semantic* label (``"bm25"`` /
    ``"dense"``) or the literal ``"dense"`` placeholder: if the
    semantic key is missing we fall back to ``"dense"`` (the only
    other non-rerank key the runner writes).
    """
    block = rerank_metrics.get("metrics", {})
    rr = block.get("rerank")
    fs = block.get(first_stage_label)
    if fs is None:
        # Fall back to whichever non-rerank key the runner wrote.
        for k, v in block.items():
            if k != "rerank":
                fs = v
                break
    if fs is None or rr is None:
        raise SystemExit(
            f"Metrics file is missing '{first_stage_label}' or 'rerank' "
            f"block. Keys seen: {list(block.keys())}"
        )
    out: dict[str, Any] = {"first_stage_label": first_stage_label}
    # Constrained ceiling: the reranker can only re-order what the first
    # stage retrieved into top-K, so the achievable rerank MRR@10 / nDCG@10
    # is upper-bounded by first_stage Recall@100 (not 1.0). On the BM25
    # full-corpus arm Recall@100 ≈ 0.62 — 38 % of queries have no relevant
    # doc to promote — so a "naive" recovery rate against a 1.0 ceiling
    # severely under-credits BM25's headroom.
    recall_ceiling = float(fs.get("recall@100", 1.0))
    for key, _label in METRICS:
        first_val = float(fs[key])
        rerank_val = float(rr[key])
        naive_gap = max(1.0 - first_val, 1e-9)
        constrained_gap = max(recall_ceiling - first_val, 1e-9)
        out[key] = {
            "first_stage": first_val,
            "rerank": rerank_val,
            "delta": rerank_val - first_val,
            "recovery_rate_naive": (rerank_val - first_val) / naive_gap,
            "recovery_rate_constrained": (rerank_val - first_val) / constrained_gap,
        }
    out["recall_ceiling"] = recall_ceiling
    return out


def wall_clock_seconds(rerank_metrics: dict[str, Any]) -> float | None:
    wc = rerank_metrics.get("wall_clock_seconds", {})
    v = wc.get("rerank") or wc.get("score_only")
    return float(v) if v is not None else None


def render_markdown(
    *,
    bm25_arm: dict[str, Any],
    dense_arm: dict[str, Any],
    bm25_wall: float | None,
    dense_wall: float | None,
) -> str:
    lines: list[str] = []
    lines.append("# W5-A — rerank Δ across BM25 vs dense first stages")
    lines.append("")
    lines.append(
        "Cross-encoder MS-MARCO-MiniLM-L-6-v2 reranking applied to two "
        "different first stages: BM25 on the full 8.8 M corpus (W2 "
        "baseline) and dense on the 50 k qrels-anchored sample (W4). "
        "Reports the rerank Δ side-by-side, plus the recovery rate "
        "Δ / (1 − first_stage) which puts the two arms on a fairer "
        "footing — a weaker first stage has more room to recover."
    )
    lines.append("")

    lines.append("## 1. Absolute Δ (rerank − first stage)")
    lines.append("")
    lines.append("| metric | BM25 first stage | BM25 + rerank | Δ_BM25 | Dense first stage | Dense + rerank | Δ_Dense |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for key, label in METRICS:
        b = bm25_arm[key]
        d = dense_arm[key]
        lines.append(
            f"| {label} | "
            f"{b['first_stage']:.4f} | {b['rerank']:.4f} | **{b['delta']:+.4f}** | "
            f"{d['first_stage']:.4f} | {d['rerank']:.4f} | **{d['delta']:+.4f}** |"
        )
    lines.append("")

    lines.append("## 2. Recovery rate — two cuts")
    lines.append("")
    lines.append(
        "**Naive recovery** = Δ / (1 − first_stage). Sanity-checks how "
        "much of the gap to a perfect 1.0 the reranker closes."
    )
    lines.append("")
    lines.append(
        "**Constrained recovery** = Δ / (Recall@100 − first_stage). The "
        "reranker can only re-order what the first stage retrieved into "
        "top-100, so the actually-achievable rerank metric is upper-bounded "
        "by Recall@100, not 1.0. On the BM25 full-corpus arm Recall@100 = "
        f"{bm25_arm['recall_ceiling']:.3f} (i.e. "
        f"{(1 - bm25_arm['recall_ceiling']) * 100:.0f} % of queries have "
        "no relevant doc to promote at all). The constrained recovery is "
        "the apples-to-apples reading."
    )
    lines.append("")
    lines.append(
        "| metric | BM25 naive | Dense naive | BM25 constrained | Dense constrained | larger on (constrained) |"
    )
    lines.append("|---|---:|---:|---:|---:|---|")
    for key, label in METRICS:
        b = bm25_arm[key]
        d = dense_arm[key]
        larger = (
            "BM25"
            if b["recovery_rate_constrained"] > d["recovery_rate_constrained"]
            else "Dense"
        )
        if abs(b["recovery_rate_constrained"] - d["recovery_rate_constrained"]) < 0.01:
            larger = "≈ tie"
        lines.append(
            f"| {label} | {b['recovery_rate_naive']:.1%} | "
            f"{d['recovery_rate_naive']:.1%} | "
            f"**{b['recovery_rate_constrained']:.1%}** | "
            f"**{d['recovery_rate_constrained']:.1%}** | {larger} |"
        )
    lines.append("")

    if bm25_wall is not None and dense_wall is not None:
        lines.append("## 3. Wall-clock")
        lines.append("")
        lines.append(
            f"- BM25 + rerank: **{bm25_wall:.0f} s** "
            f"({bm25_wall / 3600:.2f} h) of reranker forward passes."
        )
        lines.append(
            f"- Dense + rerank: **{dense_wall:.0f} s** "
            f"({dense_wall / 3600:.2f} h) of reranker forward passes."
        )
        lines.append("")
        lines.append(
            "The two arms score the same 6,980 queries × top-100 pairs, "
            "so per-pair throughput is the load-bearing comparison; absolute "
            "differences here are dominated by machine state, not workload."
        )
        lines.append("")

    lines.append("## 4. Caveats")
    lines.append("")
    lines.append(
        "- **Sampling asymmetry.** Dense first stage is W4 on the 50 k "
        "qrels-anchored sample; BM25 first stage is W2 on the full 8.8 M "
        "corpus. Absolute metric levels are not directly comparable "
        "across arms (the W4 sample inflates dense's absolute scores); "
        "the within-arm Δ and the recovery rate *are* meaningful."
    )
    lines.append(
        "- **Recall@100 is fixed by construction.** Cross-encoder reranking "
        "only re-orders the top-100 candidates the first stage already "
        "returned; recall numbers should be identical between first stage "
        "and rerank within each arm. Reported for completeness."
    )
    lines.append(
        "- **One reranker checkpoint, one depth.** Same `ms-marco-MiniLM-"
        "L-6-v2` model, same K=100 on both arms. The next axis (W5-B) "
        "varies K ∈ {50, 100, 200} on both first stages to draw the full "
        "performance–latency Pareto."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    logger.info("Loading dense+rerank metrics from %s ...", args.dense_rerank_metrics)
    dense_metrics = load_metrics_json(args.dense_rerank_metrics)
    dense_arm = extract_arm(dense_metrics, "dense")

    logger.info("Loading BM25+rerank metrics from %s ...", args.bm25_rerank_metrics)
    bm25_metrics = load_metrics_json(args.bm25_rerank_metrics)
    bm25_arm = extract_arm(bm25_metrics, "bm25")

    bm25_wall = wall_clock_seconds(bm25_metrics)
    dense_wall = wall_clock_seconds(dense_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "task": "rerank_first_stage_compare",
        "metrics": [k for k, _ in METRICS],
        "bm25_arm": bm25_arm,
        "dense_arm": dense_arm,
        "wall_clock_seconds": {
            "bm25_rerank": bm25_wall,
            "dense_rerank": dense_wall,
        },
        "notes": (
            "Δ = rerank − first_stage. Recovery rate = Δ / (1 − "
            "first_stage). The sampling asymmetry (dense on W4 sample, "
            "BM25 on full corpus) makes absolute metric levels "
            "incomparable across arms; within-arm Δ and recovery rate "
            "remain meaningful."
        ),
    }
    summary_json = args.output_dir / "summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", summary_json)

    md = render_markdown(
        bm25_arm=bm25_arm,
        dense_arm=dense_arm,
        bm25_wall=bm25_wall,
        dense_wall=dense_wall,
    )
    md_path = args.output_dir / "summary.md"
    md_path.write_text(md)
    logger.info("Wrote %s", md_path)

    # ---- console summary ----
    print()
    print("=== W5-A — rerank Δ across BM25 vs dense first stages ===")
    print(f"  {'metric':14s}  {'Δ_BM25':>10s}  {'Δ_Dense':>10s}  "
          f"{'naive B':>8s}  {'naive D':>8s}  {'cnstr B':>8s}  {'cnstr D':>8s}")
    for key, label in METRICS:
        b = bm25_arm[key]
        d = dense_arm[key]
        print(
            f"  {label:14s}  "
            f"{b['delta']:>+10.4f}  {d['delta']:>+10.4f}  "
            f"{b['recovery_rate_naive']:>8.1%}  {d['recovery_rate_naive']:>8.1%}  "
            f"{b['recovery_rate_constrained']:>8.1%}  {d['recovery_rate_constrained']:>8.1%}"
        )


if __name__ == "__main__":
    main()
