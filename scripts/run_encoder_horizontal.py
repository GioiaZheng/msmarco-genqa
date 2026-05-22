"""Same-tier dense encoder horizontal on the W4 sampled corpus.

Drives ``experiments/run_dense_retrieval.py`` for two new encoders
(`BAAI/bge-small-en-v1.5` and `sentence-transformers/all-MiniLM-L12-v2`)
on the **same 50 k qrels-anchored sample** as the existing W4 baseline
(`sentence-transformers/all-MiniLM-L6-v2`), then aggregates all three
metrics.json files into a head-to-head table.

The two new runs reuse the W4 sample selection (`qrels-anchored,
seed = 42`) so the comparison is apples-to-apples. The
``--model-name`` override (added in this session) auto-keys the cached
FAISS index dir on the model id so the new runs don't overwrite the
W4 baseline index.

Output: ``outputs/week04_encoder_horizontal/`` with::

    summary.json
    summary.md

Encoder runs themselves land under ``outputs/week04_dense_<model_safe>/``.
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

logger = logging.getLogger("run_encoder_horizontal")


# (label, model_name, output_subdir, reuse_existing)
ENCODERS: list[dict[str, Any]] = [
    {
        "label": "all-MiniLM-L6-v2 (W4 baseline)",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "output_dir": "outputs/week04_dense",
        "reuse_existing": "outputs/week04_dense/metrics.json",
    },
    {
        "label": "all-MiniLM-L12-v2",
        "model_name": "sentence-transformers/all-MiniLM-L12-v2",
        "output_dir": "outputs/week04_dense_minilm_l12",
        "reuse_existing": None,
    },
    {
        "label": "bge-small-en-v1.5",
        "model_name": "BAAI/bge-small-en-v1.5",
        "output_dir": "outputs/week04_dense_bge_small_en_v1_5",
        "reuse_existing": None,
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--skip-runs",
        action="store_true",
        help="Only aggregate from existing metrics.json files; skip re-encoding.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/week04_encoder_horizontal",
    )
    return p.parse_args()


def run_one_encoder(enc: dict[str, Any]) -> None:
    if enc["reuse_existing"]:
        logger.info("[%s] reusing %s", enc["label"], enc["reuse_existing"])
        return
    cmd: list[str] = [
        sys.executable,
        str(PROJECT_ROOT / "experiments/run_dense_retrieval.py"),
        "--model-name", enc["model_name"],
        "--output-dir", enc["output_dir"],
    ]
    logger.info("[%s] running: %s", enc["label"], " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True)
    logger.info("[%s] done in %.1f min", enc["label"], (time.time() - t0) / 60)


def load_metrics(enc: dict[str, Any]) -> dict[str, Any]:
    src = enc["reuse_existing"] or f"{enc['output_dir']}/metrics.json"
    path = PROJECT_ROOT / src
    if not path.exists():
        raise SystemExit(
            f"Missing metrics.json for encoder {enc['label']!r} at {path}."
        )
    with open(path) as f:
        return json.load(f)


def extract(enc: dict[str, Any]) -> dict[str, Any]:
    m = load_metrics(enc)
    dense = m.get("metrics", {}).get("dense", {})
    bm25_sample = m.get("metrics", {}).get("bm25_sample", {})
    wc = m.get("wall_clock_seconds", {})
    encode_sec = float(wc.get("encode_corpus") or wc.get("encode") or 0.0)
    n_passages = m.get("config", {}).get("dense", {}).get("sample_size") or 50000
    ms_per_passage = (1000.0 * encode_sec / n_passages) if encode_sec else 0.0
    return {
        "label": enc["label"],
        "model_name": enc["model_name"],
        "dense_mrr_at_10": float(dense.get("mrr@10", 0.0)),
        "dense_ndcg_at_10": float(dense.get("ndcg@10", 0.0)),
        "dense_recall_at_100": float(dense.get("recall@100", 0.0)),
        "dense_recall_at_1000": float(dense.get("recall@1000", 0.0)),
        "bm25_sample_mrr_at_10": float(bm25_sample.get("mrr@10", 0.0))
        if bm25_sample else None,
        "encode_seconds": encode_sec,
        "encode_ms_per_passage": ms_per_passage,
        "n_passages": int(n_passages),
    }


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# W4-B — same-tier encoder horizontal on the W4 sampled corpus")
    lines.append("")
    lines.append(
        "Three encoders evaluated head-to-head on the same 50 k qrels-anchored "
        "passage sample. The W4 sample selection and BM25-on-sample baseline "
        "are shared across all three rows, so the comparison is apples-to-apples."
    )
    lines.append("")
    lines.append("## 1. Retrieval quality + encoding cost")
    lines.append("")
    lines.append(
        "| encoder | MRR@10 | nDCG@10 | Recall@100 | Recall@1000 | encode s | ms/passage |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| `{r['model_name']}` | {r['dense_mrr_at_10']:.4f} | "
            f"{r['dense_ndcg_at_10']:.4f} | {r['dense_recall_at_100']:.4f} | "
            f"{r['dense_recall_at_1000']:.4f} | "
            f"{r['encode_seconds']:.0f} | {r['encode_ms_per_passage']:.2f} |"
        )
    lines.append("")
    # Pick a winner for use by W4-A (highest MRR@10 wins).
    winner = max(rows, key=lambda r: r["dense_mrr_at_10"])
    lines.append(
        f"**Best small encoder (by MRR@10): `{winner['model_name']}` "
        f"at {winner['dense_mrr_at_10']:.4f}.** W4-A density sweep uses "
        "this encoder."
    )
    lines.append("")
    lines.append("## 2. Caveats")
    lines.append("")
    lines.append(
        "- All three encoders are evaluated against the W4 sample's "
        "BM25-on-sample baseline (same row in `bm25_sample` of every "
        "metrics.json). Absolute level is inflated by the qrels-anchored "
        "sample; the within-row dense MRR@10 *gap* between encoders is "
        "the load-bearing read."
    )
    lines.append(
        "- Encoding throughput depends on hardware (CPU contention, MPS / "
        "CUDA availability). The ms/passage column is reported as an "
        "order-of-magnitude cost, not a benchmark."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    if not args.skip_runs:
        for enc in ENCODERS:
            run_one_encoder(enc)

    rows = [extract(enc) for enc in ENCODERS]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    winner = max(rows, key=lambda r: r["dense_mrr_at_10"])

    summary = {
        "task": "encoder_horizontal",
        "scope": "3 encoders × same W4 50k qrels-anchored sample",
        "rows": rows,
        "best_encoder_by_mrr_at_10": winner["model_name"],
        "best_encoder_label": winner["label"],
        "notes": (
            "All three encoders evaluated on the same qrels-anchored 50 k "
            "sample. Pick `best_encoder_by_mrr_at_10` for W4-A density sweep."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "summary.md").write_text(render_markdown(rows))
    logger.info("Wrote %s/summary.{json,md}",
                args.output_dir.relative_to(PROJECT_ROOT))

    # ---- console summary ----
    print()
    print("=== W4-B — encoder horizontal ===")
    print(f"  {'encoder':40s}  {'MRR@10':>8s}  {'nDCG@10':>8s}  {'R@100':>7s}  {'ms/p':>6s}")
    for r in rows:
        print(
            f"  {r['model_name']:40s}  "
            f"{r['dense_mrr_at_10']:>8.4f}  "
            f"{r['dense_ndcg_at_10']:>8.4f}  "
            f"{r['dense_recall_at_100']:>7.4f}  "
            f"{r['encode_ms_per_passage']:>6.2f}"
        )
    print(f"\n  Best: {winner['model_name']} (MRR@10 = {winner['dense_mrr_at_10']:.4f})")


if __name__ == "__main__":
    main()
