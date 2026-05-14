"""Lightweight failure taxonomy on a sample of W3 regression queries.

A *regression* (per ``scripts/analyze_generation_rerank.py``) is a query
where the reranker brings the relevant passage into top-3, yet the
generator produces a *worse* token-F1 answer than under BM25. There are
233 such queries in the full-dev W3 comparison; this script samples a
seeded subset, applies a handful of deterministic heuristic rules to
attach a coarse failure-mode label, and dumps:

- ``regression_taxonomy.json``  — machine-readable aggregate + per-row.
- ``regression_taxonomy.md``    — human-readable inspection report.

The taxonomy categories are intentionally simple and rule-based; this
is a *triage* over the regression bucket, not a final classification.
Categories applied in order (first match wins):

1. ``truncation_short``       — rerank prediction has ≤ 3 tokens.
2. ``truncation_midword``     — rerank prediction ends without terminal
                                 punctuation on an alphabetic char
                                 (e.g. cut by ``max_new_tokens=64``).
3. ``topic_drift``            — rerank prediction shares < 20 % tokens
                                 with any reference and < 30 % with the
                                 BM25 prediction (semantically far from
                                 both ground truth and the previous
                                 winner).
4. ``extractive_passage_bias`` — rerank prediction is a (case-insensitive)
                                 substring of one of the rerank top-3
                                 passages, suggesting the generator
                                 verbatim-copied a fragment.
5. ``semantic_mismatch``       — anything else (paraphrase the metric
                                 missed, plausible-but-wrong answer,
                                 etc.). The residual catch-all.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("regression_failure_taxonomy")

PUNCT_END = set(".!?:;\"')")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-query-metrics",
        type=Path,
        default=PROJECT_ROOT / "outputs/week06_analysis/per_query_metrics.jsonl",
    )
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
        default=PROJECT_ROOT / "outputs/week06_analysis",
    )
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--passage-snippet-chars", type=int, default=240)
    return parser.parse_args()


def _tokens(text: str) -> set[str]:
    return {t for t in text.lower().split() if t}


def classify(row: dict[str, Any], rerank_passages: list[str]) -> str:
    pred = row["rerank_prediction"] or ""
    pred_stripped = pred.strip()
    bm25_pred = row["bm25_prediction"] or ""
    refs = row.get("references") or []

    pred_tokens = pred_stripped.split()
    if len(pred_tokens) <= 3:
        return "truncation_short"
    last = pred_stripped[-1]
    if last.isalpha() and last not in PUNCT_END:
        return "truncation_midword"

    pred_t = _tokens(pred_stripped)
    ref_t = set().union(*(_tokens(r) for r in refs)) if refs else set()
    bm25_t = _tokens(bm25_pred)
    if pred_t and ref_t:
        ref_overlap = len(pred_t & ref_t) / max(len(pred_t), 1)
        bm25_overlap = len(pred_t & bm25_t) / max(len(pred_t), 1)
        if ref_overlap < 0.2 and bm25_overlap < 0.3:
            return "topic_drift"

    pred_lower = pred_stripped.lower()
    for psg in rerank_passages:
        if pred_lower and pred_lower in (psg or "").lower():
            return "extractive_passage_bias"

    return "semantic_mismatch"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_qid_to_passages(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {str(r["query_id"]): list(r.get("passages") or []) for r in rows}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    per_query = load_jsonl(args.per_query_metrics)
    regressions = [r for r in per_query if r.get("bucket") == "regression"]
    logger.info(
        "Loaded %d per-query rows; %d in 'regression' bucket.",
        len(per_query),
        len(regressions),
    )
    if not regressions:
        raise SystemExit("No regression rows found — check --per-query-metrics path.")

    rng = random.Random(args.seed)
    sample = regressions if len(regressions) <= args.sample_size else rng.sample(
        regressions, args.sample_size
    )
    sample.sort(key=lambda r: r["delta_token_f1"])  # worst regressions first
    logger.info("Sampled %d regressions (seed=%d).", len(sample), args.seed)

    bm25_preds = load_jsonl(args.bm25_dir / "predictions.jsonl")
    rerank_preds = load_jsonl(args.reranked_dir / "predictions.jsonl")
    bm25_passages = build_qid_to_passages(bm25_preds)
    rerank_passages = build_qid_to_passages(rerank_preds)

    rows_out: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for r in sample:
        qid = str(r["query_id"])
        passages = rerank_passages.get(qid) or []
        label = classify(r, passages)
        counts[label] = counts.get(label, 0) + 1
        rows_out.append(
            {
                "query_id": qid,
                "query": r["query"],
                "query_type": r.get("query_type"),
                "references": r.get("references") or [],
                "bm25_prediction": r["bm25_prediction"],
                "rerank_prediction": r["rerank_prediction"],
                "bm25_token_f1": r["bm25_token_f1"],
                "rerank_token_f1": r["rerank_token_f1"],
                "delta_token_f1": r["delta_token_f1"],
                "bm25_relevant_in_top3": r["bm25_relevant_in_top3"],
                "rerank_relevant_in_top3": r["rerank_relevant_in_top3"],
                "label": label,
                "bm25_top1_passage_snippet": (
                    (bm25_passages.get(qid) or [""])[0][: args.passage_snippet_chars]
                ),
                "rerank_top1_passage_snippet": (
                    (passages or [""])[0][: args.passage_snippet_chars]
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "regression_failure_taxonomy",
        "n_regression_bucket_total": len(regressions),
        "n_sampled": len(sample),
        "seed": args.seed,
        "categories": [
            "truncation_short",
            "truncation_midword",
            "topic_drift",
            "extractive_passage_bias",
            "semantic_mismatch",
        ],
        "counts": counts,
        "rows": rows_out,
        "notes": (
            "Heuristic, deterministic triage of the 'regression' bucket "
            "produced by analyze_generation_rerank.py. Categories are "
            "applied in declaration order; first match wins. 'semantic_"
            "mismatch' is the residual catch-all. This is not a final "
            "human-curated taxonomy."
        ),
    }
    out_json = args.output_dir / "regression_taxonomy.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s", out_json)

    # ---- markdown report ----
    lines: list[str] = []
    lines.append("# Regression failure taxonomy — W3 full-dev")
    lines.append("")
    lines.append(
        f"Sampled **{len(sample)}** of **{len(regressions)}** regression-bucket "
        f"queries (seed = {args.seed}). Heuristic deterministic labels."
    )
    lines.append("")
    lines.append("## Aggregate counts")
    lines.append("")
    lines.append("| label | n | share |")
    lines.append("|---|---:|---:|")
    for k in payload["categories"]:
        n = counts.get(k, 0)
        share = n / max(len(sample), 1)
        lines.append(f"| `{k}` | {n} | {share:.1%} |")
    lines.append(f"| **total** | **{len(sample)}** | 100 % |")
    lines.append("")
    lines.append("## Per-example detail (sorted by Δ token-F1 ascending)")
    lines.append("")
    for i, row in enumerate(rows_out, 1):
        refs_str = " · ".join(f"*{r}*" for r in row["references"][:2]) or "—"
        lines.append(
            f"### {i}. qid={row['query_id']}  ·  "
            f"`{row['label']}`  ·  Δ token-F1 {row['delta_token_f1']:+.3f}  "
            f"({row['bm25_token_f1']:.3f} → {row['rerank_token_f1']:.3f})"
        )
        lines.append("")
        lines.append(f"- **Query** ({row['query_type'] or '—'}): {row['query']}")
        lines.append(f"- **Reference(s)**: {refs_str}")
        lines.append(f"- **BM25 pred**: *{row['bm25_prediction']}*")
        lines.append(f"- **Rerank pred**: *{row['rerank_prediction']}*")
        lines.append(
            f"- **BM25 top-1 passage**: {row['bm25_top1_passage_snippet']}…"
        )
        lines.append(
            f"- **Rerank top-1 passage**: {row['rerank_top1_passage_snippet']}…"
        )
        lines.append("")
    out_md = args.output_dir / "regression_taxonomy.md"
    out_md.write_text("\n".join(lines))
    logger.info("Wrote %s", out_md)

    # ---- console summary ----
    print()
    print(f"=== Regression failure taxonomy (n={len(sample)}/{len(regressions)}) ===")
    print(f"  {'label':30s}  {'n':>3s}  {'share':>6s}")
    for k in payload["categories"]:
        n = counts.get(k, 0)
        print(f"  {k:30s}  {n:>3d}  {n/max(len(sample),1):>6.1%}")


if __name__ == "__main__":
    main()
