"""Deep-dive case study of low-grounding queries on the rerank arm.

The grounding-vs-downstream-metrics analysis confirmed that on the
small low-grounding tail (~3 % lex, ~3-5 % 3-gram per arm) downstream
Token-F1 / BERTScore drops by ~0.05-0.10 in mean. This case study
zooms in: pulls a seeded 30-case sample from
the 197 rerank-arm queries with ``lex_rerank < 0.9`` OR ``ngram_rerank
< 0.9``, dumps the full (query, top-3 passages, prediction, gold)
record per case, and attaches a coarse rule-cascade failure-mode
label so the categories surface in aggregate.

Categories (first-match-wins, all on the rerank arm):

- ``prediction_too_short``    — prediction ≤2 tokens (3-gram vacuous
                                regime; commonly a passage title or
                                a single noun).
- ``paraphrase_reorder``      — ``lex_rerank ≥ 0.9`` AND
                                ``ngram_rerank < 0.9``: the content
                                words are present in the prompt but
                                the order / phrasing differs (the
                                lex metric is blind to order).
- ``parametric_or_external``  — ``lex_rerank < 0.5``: most of the
                                prediction's content tokens are NOT
                                in the top-3 passages. Either the
                                model fell back on its parametric
                                memory or it hallucinated.
- ``partial_external``        — ``0.5 ≤ lex_rerank < 0.9``: some
                                content from the prompt, some from
                                elsewhere.
- ``residual``                — catch-all (should be empty under the
                                preceding rules; reported defensively).

The labels are intentionally coarse — this is *triage* over the
low-grounding bucket, not a final classifier. The value is in the
distribution + the per-case inspection rows, not in any single
attribution.

Outputs (gitignored under ``outputs/low_grounding_cases/``):

- ``cases.jsonl``   — per-case row with all the inspection fields.
- ``cases.md``      — human-readable inspection report (one block per
                      case, sorted by min(lex, ngram) ascending).
- ``summary.json``  — aggregate counts per label + pool diagnostics.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("low_grounding_case_study")


# Match the grounding-audit tokeniser: keep [a-z0-9'] tokens, lowercase.
TOKEN_RE = re.compile(r"[a-z0-9']+")
HIGH_THRESHOLD = 0.9
LOW_LEX_PARAMETRIC = 0.5
SHORT_PREDICTION_TOKENS = 2

CATEGORIES: tuple[str, ...] = (
    "prediction_too_short",
    "paraphrase_reorder",
    "parametric_or_external",
    "partial_external",
    "residual",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--per-query-grounding",
        type=Path,
        default=PROJECT_ROOT / "outputs/grounding/per_query_grounding.jsonl",
    )
    p.add_argument(
        "--rerank-predictions",
        type=Path,
        default=PROJECT_ROOT / "outputs/generation_reranked_full/predictions.jsonl",
    )
    p.add_argument(
        "--bm25-predictions",
        type=Path,
        default=PROJECT_ROOT / "outputs/generation_bm25_full/predictions.jsonl",
        help="Only used to surface the BM25 prediction side-by-side per case.",
    )
    p.add_argument(
        "--per-query-metrics",
        type=Path,
        default=PROJECT_ROOT / "outputs/generation_analysis/per_query_metrics.jsonl",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/low_grounding_cases",
    )
    p.add_argument("--n-cases", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--passage-snippet-chars",
        type=int,
        default=320,
        help="Truncate each top-3 passage to this many chars in the markdown.",
    )
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def predictions_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        out[str(r["query_id"])] = {
            "query": r.get("query") or "",
            "prediction": r.get("prediction") or "",
            "references": list(r.get("references") or []),
            "passages": list(r.get("passages") or []),
        }
    return out


def metrics_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r["query_id"]): r for r in rows}


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def classify(
    *,
    prediction: str,
    lex_rerank: float,
    ngram_rerank: float,
) -> str:
    pred_toks = tokens(prediction)
    if len(pred_toks) <= SHORT_PREDICTION_TOKENS:
        return "prediction_too_short"
    if lex_rerank >= HIGH_THRESHOLD and ngram_rerank < HIGH_THRESHOLD:
        return "paraphrase_reorder"
    if lex_rerank < LOW_LEX_PARAMETRIC:
        return "parametric_or_external"
    if lex_rerank < HIGH_THRESHOLD:
        return "partial_external"
    return "residual"


def truncate(text: str, n: int) -> str:
    s = (text or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def render_markdown(
    *,
    pool_size: int,
    n_cases: int,
    seed: int,
    counts: Counter,
    rows: list[dict[str, Any]],
    passage_snippet_chars: int,
) -> str:
    lines: list[str] = []
    lines.append("# Low-grounding case study")
    lines.append("")
    lines.append(
        f"Sampled **{n_cases}** of the **{pool_size}** rerank-arm queries "
        f"with `lex_rerank < {HIGH_THRESHOLD}` OR `ngram_rerank < {HIGH_THRESHOLD}` "
        f"(seed = {seed}), sorted ascending by `min(lex, ngram)`. The "
        "rule-cascade label (see ``classify()``) is a coarse failure-mode "
        "triage; the value is in the distribution + the per-case rows."
    )
    lines.append("")
    lines.append("## 1. Aggregate counts")
    lines.append("")
    lines.append("| label | n | share |")
    lines.append("|---|---:|---:|")
    for cat in CATEGORIES:
        n = counts.get(cat, 0)
        share = n / max(n_cases, 1)
        lines.append(f"| `{cat}` | {n} | {share:.1%} |")
    lines.append(f"| **total** | **{n_cases}** | 100 % |")
    lines.append("")

    lines.append("## 2. Per-case detail (sorted by min(lex, ngram) ascending)")
    lines.append("")
    for i, row in enumerate(rows, 1):
        refs_str = " · ".join(f"*{truncate(r, 160)}*" for r in row["references"][:3]) or "—"
        lines.append(
            f"### {i}. qid={row['query_id']}  ·  `{row['label']}`  ·  "
            f"lex={row['lex_rerank']:.3f}  ·  ngram={row['ngram_rerank']:.3f}  ·  "
            f"token-F1 {row['rerank_token_f1']:.3f}"
        )
        lines.append("")
        lines.append(f"- **Query** ({row['query_type'] or '—'}): {row['query']}")
        lines.append(f"- **Reference(s)**: {refs_str}")
        lines.append(f"- **BM25 pred**: *{truncate(row['bm25_prediction'], 220)}*")
        lines.append(f"- **Rerank pred**: *{truncate(row['rerank_prediction'], 220)}*")
        for j, psg in enumerate(row["rerank_top3_passages"][:3], 1):
            lines.append(
                f"- **Rerank top-{j}**: {truncate(psg, passage_snippet_chars)}"
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
    grounding = {str(r["query_id"]): r for r in load_jsonl(args.per_query_grounding)}

    logger.info("Loading rerank predictions ...")
    rerank_idx = predictions_index(load_jsonl(args.rerank_predictions))
    logger.info("Loading BM25 predictions (for side-by-side display) ...")
    bm25_idx = predictions_index(load_jsonl(args.bm25_predictions))
    logger.info("Loading generation-analysis per-query metrics ...")
    metrics = metrics_index(load_jsonl(args.per_query_metrics))

    # Pool: rerank-arm queries with either grounding metric < 0.9.
    pool: list[dict[str, Any]] = []
    for qid, g in grounding.items():
        lex_r = float(g["lex_rerank"])
        ng_r = float(g["ngram_rerank"])
        if lex_r < HIGH_THRESHOLD or ng_r < HIGH_THRESHOLD:
            pool.append({"query_id": qid, "lex_rerank": lex_r, "ngram_rerank": ng_r})
    logger.info(
        "Pool: %d rerank-arm queries with lex<%.1f OR ngram<%.1f.",
        len(pool), HIGH_THRESHOLD, HIGH_THRESHOLD,
    )
    if not pool:
        raise SystemExit("Empty low-grounding pool.")

    rng = random.Random(args.seed)
    n_cases = min(args.n_cases, len(pool))
    sampled = pool if n_cases >= len(pool) else rng.sample(pool, n_cases)
    sampled.sort(key=lambda r: min(r["lex_rerank"], r["ngram_rerank"]))
    logger.info("Sampled %d cases (seed = %d).", n_cases, args.seed)

    rows_out: list[dict[str, Any]] = []
    label_counts: Counter = Counter()
    for s in sampled:
        qid = s["query_id"]
        r_pred = rerank_idx.get(qid, {})
        b_pred = bm25_idx.get(qid, {})
        m = metrics.get(qid, {})
        label = classify(
            prediction=r_pred.get("prediction", ""),
            lex_rerank=s["lex_rerank"],
            ngram_rerank=s["ngram_rerank"],
        )
        label_counts[label] += 1
        rows_out.append(
            {
                "query_id": qid,
                "query": r_pred.get("query") or m.get("query"),
                "query_type": m.get("query_type"),
                "label": label,
                "lex_rerank": s["lex_rerank"],
                "ngram_rerank": s["ngram_rerank"],
                "lex_bm25": float(grounding[qid]["lex_bm25"]),
                "ngram_bm25": float(grounding[qid]["ngram_bm25"]),
                "rerank_prediction": r_pred.get("prediction", ""),
                "bm25_prediction": b_pred.get("prediction", ""),
                "references": list(r_pred.get("references") or m.get("references") or []),
                "rerank_top3_passages": list((r_pred.get("passages") or [])[:3]),
                "bm25_top3_passages": list((b_pred.get("passages") or [])[:3]),
                "rerank_token_f1": float(m.get("rerank_token_f1", 0.0)),
                "bm25_token_f1": float(m.get("bm25_token_f1", 0.0)),
                "delta_token_f1": float(m.get("delta_token_f1", 0.0)),
                "bucket": m.get("bucket"),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_jsonl = args.output_dir / "cases.jsonl"
    with open(cases_jsonl, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    logger.info("Wrote %s (%d rows)", cases_jsonl, len(rows_out))

    cases_md = args.output_dir / "cases.md"
    cases_md.write_text(
        render_markdown(
            pool_size=len(pool),
            n_cases=len(rows_out),
            seed=args.seed,
            counts=label_counts,
            rows=rows_out,
            passage_snippet_chars=args.passage_snippet_chars,
        )
    )
    logger.info("Wrote %s", cases_md)

    summary = {
        "task": "low_grounding_case_study",
        "pool_size": len(pool),
        "n_cases": len(rows_out),
        "seed": args.seed,
        "high_threshold": HIGH_THRESHOLD,
        "categories": list(CATEGORIES),
        "counts": dict(label_counts),
        "thresholds": {
            "short_prediction_tokens": SHORT_PREDICTION_TOKENS,
            "low_lex_parametric": LOW_LEX_PARAMETRIC,
        },
        "notes": (
            "Pool: rerank-arm queries with lex_rerank < 0.9 OR "
            "ngram_rerank < 0.9. Cases are sorted ascending by "
            "min(lex_rerank, ngram_rerank). Rule cascade is "
            "first-match-wins."
        ),
    }
    summary_json = args.output_dir / "summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", summary_json)

    # ---- console summary ----
    print()
    print(f"=== Low-grounding case study (n={len(rows_out)}/{len(pool)}, seed={args.seed}) ===")
    print(f"  {'label':28s}  {'n':>3s}  {'share':>6s}")
    for cat in CATEGORIES:
        n = label_counts.get(cat, 0)
        print(f"  {cat:28s}  {n:>3d}  {n/max(len(rows_out),1):>6.1%}")


if __name__ == "__main__":
    main()
