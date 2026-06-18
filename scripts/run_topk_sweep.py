"""Rerank-depth top-k sweep on BM25 and dense first stages.

Drives four ``experiments/run_reranker.py`` invocations and aggregates
the results into a single perf–latency Pareto table + figure.

Design choices (per the project ddl plan; see the
``project_ddl_2026_05_22_scope_b`` memory):

- **K ∈ {50, 100, 200}** on both BM25 (W2 full-corpus run.tsv) and
  dense (W4 sampled run.tsv) first stages.
- **K = 50 and K = 200 are scored on a 1,000-query random
  subsample** (the reranker's ``--num-eval-queries 1000`` flag, seeded
  internally by ``cfg['seed'] = 42``). K = 100 reuses the existing
  full-dev runs (W5 dense+rerank, W5-A BM25+rerank) — that's where
  the absolute headline numbers live; the K = 50 / 200 cells are only
  there to draw the *shape* of the perf–latency Pareto.

Output: ``outputs/rerank_k_sweep/`` (gitignored) with::

    bm25_k50/, bm25_k200/, dense_k50/, dense_k200/    # raw rerank runs
    summary.json
    summary.md
    pareto.png                                         # tracked under figures/

The two K=100 cells are read from their existing locations:
``outputs/cross_encoder_rerank_full/metrics.json`` (dense+rerank) and
``outputs/cross_encoder_rerank_bm25_full/metrics.json`` (W5-A BM25+rerank).

If those cells are missing the script fails fast with a clear error;
this avoids a partial Pareto that could be misread.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("run_topk_sweep")


# (label, first_stage_key_in_metrics, input_run_relpath, input_stage, K, output_subdir, num_eval_queries)
# K=100 cells point at the existing full-dev runs (no new compute).
CELLS: list[dict[str, Any]] = [
    {
        "label": "BM25 K=50",
        "first_stage": "bm25",
        "K": 50,
        "input_run": "outputs/bm25_baseline/run.tsv",
        "input_stage": "bm25_baseline",
        "output_dir": "outputs/rerank_k_sweep/bm25_k50",
        "num_eval_queries": 1000,
        "reuse_existing": None,
    },
    {
        "label": "BM25 K=100",
        "first_stage": "bm25",
        "K": 100,
        "input_run": "outputs/bm25_baseline/run.tsv",
        "input_stage": "bm25_baseline",
        "output_dir": "outputs/cross_encoder_rerank_bm25_full",
        "num_eval_queries": None,
        "reuse_existing": "outputs/cross_encoder_rerank_bm25_full/metrics.json",
    },
    {
        "label": "BM25 K=200",
        "first_stage": "bm25",
        "K": 200,
        "input_run": "outputs/bm25_baseline/run.tsv",
        "input_stage": "bm25_baseline",
        "output_dir": "outputs/rerank_k_sweep/bm25_k200",
        "num_eval_queries": 1000,
        "reuse_existing": None,
    },
    {
        "label": "Dense K=50",
        "first_stage": "dense",
        "K": 50,
        "input_run": "outputs/dense_retrieval/run.tsv",
        "input_stage": "dense_retrieval",
        "output_dir": "outputs/rerank_k_sweep/dense_k50",
        "num_eval_queries": 1000,
        "reuse_existing": None,
    },
    {
        "label": "Dense K=100",
        "first_stage": "dense",
        "K": 100,
        "input_run": "outputs/dense_retrieval/run.tsv",
        "input_stage": "dense_retrieval",
        "output_dir": "outputs/cross_encoder_rerank_full",
        "num_eval_queries": None,
        "reuse_existing": "outputs/cross_encoder_rerank_full/metrics.json",
    },
    {
        "label": "Dense K=200",
        "first_stage": "dense",
        "K": 200,
        "input_run": "outputs/dense_retrieval/run.tsv",
        "input_stage": "dense_retrieval",
        "output_dir": "outputs/rerank_k_sweep/dense_k200",
        "num_eval_queries": 1000,
        "reuse_existing": None,
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--skip-runs",
        action="store_true",
        help=(
            "Skip the four new reranker invocations and only aggregate from "
            "metrics.json files already on disk. Useful when re-rendering "
            "the summary without re-running."
        ),
    )
    p.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Restrict to cells whose K is in this list. Cells outside the "
            "filter are dropped from BOTH the run loop and the aggregation. "
            "Use `--k-values 50 100` to score today's cheap subset and defer "
            "the expensive K=200 cells; rerun later with the default to get "
            "the full Pareto. Aggregation only reads metrics.json files that "
            "already exist for cells in the filter, so a partial sweep is "
            "honoured."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/rerank_k_sweep",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "figures",
    )
    return p.parse_args()


def run_one_cell(cell: dict[str, Any]) -> None:
    if cell["reuse_existing"]:
        logger.info("[%s] reusing existing run at %s", cell["label"], cell["reuse_existing"])
        return
    out_dir = PROJECT_ROOT / cell["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = [
        sys.executable,
        str(PROJECT_ROOT / "experiments/run_reranker.py"),
        "--input-run", cell["input_run"],
        "--input-stage", cell["input_stage"],
        "--output-dir", cell["output_dir"],
        "--rerank-top-k", str(cell["K"]),
        "--resume",
    ]
    if cell["num_eval_queries"] is not None:
        cmd.extend(["--num-eval-queries", str(cell["num_eval_queries"])])
    logger.info("[%s] running: %s", cell["label"], " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True)
    logger.info("[%s] done in %.1f min", cell["label"], (time.time() - t0) / 60)


def load_metrics(cell: dict[str, Any]) -> dict[str, Any]:
    src = cell["reuse_existing"] or f"{cell['output_dir']}/metrics.json"
    path = PROJECT_ROOT / src
    if not path.exists():
        raise SystemExit(
            f"Missing metrics.json for cell {cell['label']!r} at {path}."
        )
    with open(path) as f:
        return json.load(f)


def _first_stage_block(
    block: dict[str, Any], first_stage_label: str
) -> dict[str, Any]:
    """Return the first-stage metrics block, with a fallback for the
    reranker's hard-coded ``"dense"`` placeholder.

    ``run_reranker.py`` writes the first-stage block under the key
    ``"dense"`` regardless of whether the actual input run was BM25 or
    dense. The fallback mirrors the convention already used in
    ``compare_rerank_first_stages.py``.
    """
    fs = block.get(first_stage_label)
    if fs is not None:
        return fs
    for k, v in block.items():
        if k != "rerank":
            return v
    return {}


def aggregate(cells: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        m = load_metrics(cell)
        block = m.get("metrics", {})
        rerank = block.get("rerank", {})
        first = _first_stage_block(block, cell["first_stage"])
        wc = m.get("wall_clock_seconds", {})
        n_q = m.get("config", {}).get("eval_retrieval", {}).get("n_eval_queries")
        if n_q is None:
            # Fall back to the count of qids in the run.tsv if metrics didn't
            # record it. Reranker writes one block per qid × K so any qid
            # appears K times; cheaper to use the existing eval_n if present.
            n_q = m.get("n_eval_queries") or m.get("config", {}).get("n_eval_queries")
        rerank_seconds = float(wc.get("rerank") or wc.get("score_only") or 0.0)
        pairs = (
            float(cell["K"]) * (cell["num_eval_queries"] or 6980)
            if rerank_seconds > 0 else 0.0
        )
        pairs_per_sec = pairs / rerank_seconds if rerank_seconds else 0.0
        rows.append(
            {
                "label": cell["label"],
                "first_stage": cell["first_stage"],
                "K": cell["K"],
                "n_eval_queries": cell["num_eval_queries"] or n_q or 6980,
                "first_stage_mrr_at_10": float(first.get("mrr@10", 0.0)),
                "first_stage_ndcg_at_10": float(first.get("ndcg@10", 0.0)),
                "rerank_mrr_at_10": float(rerank.get("mrr@10", 0.0)),
                "rerank_ndcg_at_10": float(rerank.get("ndcg@10", 0.0)),
                "delta_mrr_at_10": float(rerank.get("mrr@10", 0.0))
                                    - float(first.get("mrr@10", 0.0)),
                "delta_ndcg_at_10": float(rerank.get("ndcg@10", 0.0))
                                     - float(first.get("ndcg@10", 0.0)),
                "rerank_seconds": rerank_seconds,
                "pairs_per_second": pairs_per_sec,
            }
        )
    return {"cells": rows}


def render_markdown(agg: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# W5-B — rerank depth top-k sweep on BM25 vs dense first stages")
    lines.append("")
    lines.append(
        "Cross-encoder MS-MARCO-MiniLM-L-6-v2 reranking applied at "
        "K ∈ {50, 100, 200} on both first stages. K = 100 cells reuse the "
        "existing full-dev runs (W5 dense+rerank, W5-A BM25+rerank). "
        "K = 50 / 200 cells were re-scored on a deterministic 1 000-query "
        "subsample of dev/small (seed 42); their absolute level reflects "
        "the subsample, but the *shape* of the perf–latency Pareto curve "
        "(MRR@10 vs wall-clock) is what this table answers."
    )
    lines.append("")
    lines.append("## 1. Per-cell metrics + wall-clock")
    lines.append("")
    lines.append(
        "| cell | n_q | first MRR@10 | rerank MRR@10 | Δ MRR@10 | rerank nDCG@10 | "
        "Δ nDCG@10 | rerank sec | pairs/s |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in agg["cells"]:
        lines.append(
            f"| {r['label']} | {r['n_eval_queries']} | "
            f"{r['first_stage_mrr_at_10']:.4f} | {r['rerank_mrr_at_10']:.4f} | "
            f"**{r['delta_mrr_at_10']:+.4f}** | "
            f"{r['rerank_ndcg_at_10']:.4f} | **{r['delta_ndcg_at_10']:+.4f}** | "
            f"{r['rerank_seconds']:.0f} | {r['pairs_per_second']:.1f} |"
        )
    lines.append("")
    lines.append("## 2. Caveats")
    lines.append("")
    lines.append(
        "- K = 50 / 200 cells score 1,000 paired qids; K = 100 cells score "
        "the full 6,980. Absolute MRR@10 levels at K = 50 / 200 are not "
        "directly comparable to the K = 100 numbers because the eval set "
        "is different; ΔMRR@10 within each first-stage arm IS comparable "
        "because both first-stage and rerank are evaluated on the same set."
    )
    lines.append(
        "- Pairs/s is the cross-encoder forward-pass throughput; it "
        "reflects machine state at run time (CPU contention, MPS/CUDA "
        "availability) and should be read as an order-of-magnitude cost."
    )
    lines.append(
        "- W5-B at the full 6,980 dev/small is deferred (project_ddl memory)."
    )
    lines.append("")
    return "\n".join(lines)


def plot_pareto(agg: dict[str, Any], figures_dir: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    by_stage: dict[str, list[dict[str, Any]]] = {"bm25": [], "dense": []}
    for r in agg["cells"]:
        by_stage[r["first_stage"]].append(r)
    for stage, rows in by_stage.items():
        rows_sorted = sorted(rows, key=lambda x: x["K"])
        xs = [r["rerank_seconds"] / 60.0 for r in rows_sorted]
        ys = [r["rerank_mrr_at_10"] for r in rows_sorted]
        ks = [r["K"] for r in rows_sorted]
        ax.plot(xs, ys, marker="o", label=stage.upper())
        for x, y, k in zip(xs, ys, ks):
            ax.annotate(f"K={k}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Rerank wall-clock (min)")
    ax.set_ylabel("Rerank MRR@10")
    ax.set_title("W5-B — perf–latency Pareto by first stage and K")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    out_path = figures_dir / "w5b_rerank_k_sweep_pareto.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{out_path.name}"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    cells = (
        [c for c in CELLS if c["K"] in set(args.k_values)]
        if args.k_values else CELLS
    )
    if args.k_values:
        logger.info(
            "Filter --k-values=%s → %d / %d cells in scope.",
            args.k_values, len(cells), len(CELLS),
        )

    if not args.skip_runs:
        for cell in cells:
            run_one_cell(cell)

    agg = aggregate(cells)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_rel = plot_pareto(agg, args.figures_dir)

    summary = {
        "task": "topk_sweep",
        "scope": "K in {50, 100, 200}; K=50,200 on 1000-q subsample (seed 42); K=100 reuses W5 / W5-A full-dev runs",
        "cells": agg["cells"],
        "figure": fig_rel,
        "notes": (
            "Pareto curve drawn over (wall-clock minutes, rerank MRR@10) "
            "per first stage. Absolute MRR@10 at K=50/200 reflects the "
            "1000-q subsample and is not directly comparable to the K=100 "
            "level; ΔMRR@10 within each arm is meaningful."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "summary.md").write_text(render_markdown(agg))
    logger.info("Wrote %s/summary.{json,md} and %s",
                args.output_dir.relative_to(PROJECT_ROOT), fig_rel)

    # ---- console summary ----
    print()
    print("=== W5-B — rerank top-k Pareto ===")
    print(f"  {'cell':16s}  {'n_q':>5s}  {'rerank MRR@10':>14s}  {'Δ MRR@10':>9s}  {'sec':>6s}")
    for r in agg["cells"]:
        print(
            f"  {r['label']:16s}  {r['n_eval_queries']:>5d}  "
            f"{r['rerank_mrr_at_10']:>14.4f}  "
            f"{r['delta_mrr_at_10']:>+9.4f}  "
            f"{r['rerank_seconds']:>6.0f}"
        )


if __name__ == "__main__":
    main()
