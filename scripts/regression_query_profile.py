"""Profile W3 regression queries vs the rest on simple structural features.

A *regression* (per ``scripts/analyze_generation_rerank.py``) is a query
where the cross-encoder brought a qrel-relevant passage into top-3 yet
the generator produced a *worse* token-F1 answer than under BM25 — 233
queries out of 6,980 paired qids. The W6 regression taxonomy already
attributed most of these to generator-side truncation; this script
asks a complementary question:

> Do regression queries *look different* from non-regression queries
> on simple, model-agnostic structural features?

Features computed per qid (all deterministic, CPU-cheap):

- ``query_length_tokens``                       — len(query.split())
- ``query_length_chars``                        — len(query)
- ``n_qrels``                                   — count of dev/small qrels
                                                  for this qid (from W6
                                                  per_query metrics).
- ``rerank_top3_avg_passage_length_tokens``     — mean over the rerank
                                                  top-3 passages T5
                                                  actually saw at prompt
                                                  time (from W3 reranked
                                                  predictions.jsonl).
- ``bm25_top3_avg_passage_length_tokens``       — same, for the BM25 arm
                                                  (for completeness — the
                                                  regression bucket is
                                                  defined on the rerank
                                                  arm so the rerank
                                                  column is the load-
                                                  bearing one).

For each feature we report median / IQR / mean on the two groups, plus
a two-sided Mann-Whitney U test and a rank-biserial effect size. Three
boxplot PNGs go under ``figures/`` (tracked in git); the per-qid
features, summary, and markdown table go under
``outputs/regression_features/`` (gitignored).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("regression_query_profile")


FEATURES: tuple[tuple[str, str], ...] = (
    ("query_length_tokens", "Query length (tokens)"),
    ("query_length_chars", "Query length (chars)"),
    ("n_qrels", "Qrels per query"),
    ("rerank_top3_avg_passage_length_tokens", "Rerank top-3 mean passage length (tokens)"),
    ("bm25_top3_avg_passage_length_tokens", "BM25 top-3 mean passage length (tokens)"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--per-query-metrics",
        type=Path,
        default=PROJECT_ROOT / "outputs/generation_analysis/per_query_metrics.jsonl",
    )
    p.add_argument(
        "--bm25-predictions",
        type=Path,
        default=PROJECT_ROOT / "outputs/generation_bm25_full/predictions.jsonl",
    )
    p.add_argument(
        "--rerank-predictions",
        type=Path,
        default=PROJECT_ROOT / "outputs/generation_reranked_full/predictions.jsonl",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/regression_features",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "figures",
    )
    p.add_argument("--top-k", type=int, default=3, help="Top-K passages to average over.")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def passages_index(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {str(r["query_id"]): list(r.get("passages") or []) for r in rows}


def mean_passage_tokens(passages: Sequence[str], k: int) -> float:
    pool = passages[:k]
    if not pool:
        return 0.0
    lengths = [len((p or "").split()) for p in pool]
    return sum(lengths) / len(lengths)


def summarise(values: Sequence[float]) -> dict[str, float]:
    import statistics

    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "min": 0.0, "max": 0.0}
    sorted_vals = sorted(values)
    return {
        "n": n,
        "mean": float(statistics.fmean(sorted_vals)),
        "median": float(statistics.median(sorted_vals)),
        "p25": float(sorted_vals[n // 4]),
        "p75": float(sorted_vals[(3 * n) // 4]),
        "min": float(sorted_vals[0]),
        "max": float(sorted_vals[-1]),
    }


def mann_whitney(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """Two-sided Mann-Whitney U + rank-biserial effect size.

    scipy returns ``U`` for the first argument (``a``). The rank-biserial
    correlation r = 2U / (n_a * n_b) - 1 lives in [-1, 1]; positive
    means group ``a`` tends to be larger than ``b`` (equivalently
    P(a > b) > 0.5).
    """
    from scipy.stats import mannwhitneyu

    n_a = len(a)
    n_b = len(b)
    if n_a == 0 or n_b == 0:
        return {"U": 0.0, "p_two_sided": 1.0, "effect_size_r": 0.0}
    res = mannwhitneyu(a, b, alternative="two-sided")
    U = float(res.statistic)
    p = float(res.pvalue)
    r = 2.0 * U / (n_a * n_b) - 1.0
    return {"U": U, "p_two_sided": p, "effect_size_r": float(r)}


def render_markdown(
    *,
    n_total: int,
    n_regression: int,
    n_other: int,
    summary: dict[str, Any],
    figures_rel: dict[str, str],
) -> str:
    lines: list[str] = []
    lines.append("# Regression vs non-regression query features — W6-C")
    lines.append("")
    lines.append(
        f"Compared the **{n_regression}** regression-bucket queries against "
        f"the **{n_other}** non-regression paired qids "
        f"({n_total} total) on five structural features. "
        "Two-sided Mann-Whitney U test per feature; rank-biserial *r* in "
        "[-1, 1] gives the direction and magnitude of the shift "
        "(*r* > 0 ⇒ regression group tends to be larger on the feature)."
    )
    lines.append("")
    lines.append("## 1. Per-feature comparison")
    lines.append("")
    lines.append(
        "| feature | regr median (IQR) | other median (IQR) | regr mean | other mean | U | p | r |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for key, label in FEATURES:
        s = summary[key]
        r_s = s["regression"]
        o_s = s["other"]
        mw = s["mannwhitney"]
        lines.append(
            f"| {label} | "
            f"{r_s['median']:.1f} ({r_s['p25']:.1f}–{r_s['p75']:.1f}) | "
            f"{o_s['median']:.1f} ({o_s['p25']:.1f}–{o_s['p75']:.1f}) | "
            f"{r_s['mean']:.2f} | {o_s['mean']:.2f} | "
            f"{mw['U']:.0f} | {mw['p_two_sided']:.3g} | "
            f"{mw['effect_size_r']:+.3f} |"
        )
    lines.append("")

    # Headline read: features with p < 0.05 + |r| >= 0.1
    notable = []
    for key, label in FEATURES:
        mw = summary[key]["mannwhitney"]
        if mw["p_two_sided"] < 0.05 and abs(mw["effect_size_r"]) >= 0.10:
            notable.append((label, mw))
    lines.append("**Headline read.**")
    lines.append("")
    if notable:
        for label, mw in notable:
            direction = "higher" if mw["effect_size_r"] > 0 else "lower"
            lines.append(
                f"- *{label}* is significantly **{direction}** on regressions "
                f"(p = {mw['p_two_sided']:.3g}, rank-biserial r = "
                f"{mw['effect_size_r']:+.3f})."
            )
    else:
        lines.append(
            "- No feature crosses the (p < 0.05, |r| ≥ 0.10) threshold. "
            "Regression queries do not differ from the rest on any of these "
            "five structural features at meaningful effect size — consistent "
            "with the W6 taxonomy finding that the failure is generator-side, "
            "not driven by query / retrieval shape."
        )
    lines.append("")

    if figures_rel:
        lines.append("## 2. Distribution plots")
        lines.append("")
        for label, rel in figures_rel.items():
            lines.append(f"- **{label}** — `{rel}`")
        lines.append("")
    return "\n".join(lines)


def plot_boxes(
    *,
    features_by_qid: list[dict[str, Any]],
    figures_dir: Path,
) -> dict[str, str]:
    """Three side-by-side boxplots (regression vs other) for the three
    primary features. Two secondary features (chars, bm25 passage length)
    are reported in the table only.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)

    primary: list[tuple[str, str, str]] = [
        ("query_length_tokens", "Query length (tokens)", "w6c_regression_vs_other_query_length.png"),
        ("n_qrels", "Qrels per query", "w6c_regression_vs_other_n_qrels.png"),
        (
            "rerank_top3_avg_passage_length_tokens",
            "Rerank top-3 mean passage length (tokens)",
            "w6c_regression_vs_other_top3_passage_length.png",
        ),
    ]
    out: dict[str, str] = {}
    for key, label, filename in primary:
        regr = [r[key] for r in features_by_qid if r["is_regression"]]
        other = [r[key] for r in features_by_qid if not r["is_regression"]]
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        ax.boxplot(
            [regr, other],
            labels=[f"regression\n(n={len(regr)})", f"other\n(n={len(other)})"],
            showfliers=False,
        )
        ax.set_ylabel(label)
        ax.set_title(f"{label}\nregression vs non-regression")
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()
        out_path = figures_dir / filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Wrote figure %s", out_path)
        out[label] = f"figures/{filename}"
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    logger.info("Loading W6 per-query metrics from %s ...", args.per_query_metrics)
    per_query = load_jsonl(args.per_query_metrics)
    logger.info("Loaded %d rows.", len(per_query))

    logger.info("Loading W3 BM25 + reranked predictions ...")
    bm25_psgs = passages_index(load_jsonl(args.bm25_predictions))
    rerank_psgs = passages_index(load_jsonl(args.rerank_predictions))

    rows_out: list[dict[str, Any]] = []
    for r in per_query:
        qid = str(r["query_id"])
        query = r.get("query") or ""
        rows_out.append(
            {
                "query_id": qid,
                "query": query,
                "bucket": r.get("bucket"),
                "is_regression": r.get("bucket") == "regression",
                "query_length_tokens": len(query.split()),
                "query_length_chars": len(query),
                "n_qrels": int(r.get("n_qrels") or 0),
                "rerank_top3_avg_passage_length_tokens": mean_passage_tokens(
                    rerank_psgs.get(qid, []), args.top_k
                ),
                "bm25_top3_avg_passage_length_tokens": mean_passage_tokens(
                    bm25_psgs.get(qid, []), args.top_k
                ),
            }
        )

    n_reg = sum(1 for r in rows_out if r["is_regression"])
    n_other = len(rows_out) - n_reg
    logger.info("Split: %d regression / %d other (total %d).", n_reg, n_other, len(rows_out))

    summary: dict[str, Any] = {}
    for key, _label in FEATURES:
        regr = [r[key] for r in rows_out if r["is_regression"]]
        other = [r[key] for r in rows_out if not r["is_regression"]]
        summary[key] = {
            "regression": summarise(regr),
            "other": summarise(other),
            "mannwhitney": mann_whitney(regr, other),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_query_path = args.output_dir / "per_query_features.jsonl"
    with open(per_query_path, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    logger.info("Wrote %s (%d rows)", per_query_path, len(rows_out))

    figures_rel = plot_boxes(features_by_qid=rows_out, figures_dir=args.figures_dir)

    summary_payload = {
        "task": "regression_query_profile",
        "n_total": len(rows_out),
        "n_regression": n_reg,
        "n_other": n_other,
        "top_k": args.top_k,
        "features": [k for k, _ in FEATURES],
        "by_feature": summary,
        "figures": figures_rel,
        "notes": (
            "Two-sided Mann-Whitney U on each feature; rank-biserial r "
            "in [-1, 1] (positive ⇒ regression group tends to be larger). "
            "Boxplots in figures/ use the rank-stable IQR / median (no "
            "outlier marks) for the three primary features; the two "
            "secondary features (chars, BM25 passage length) are reported "
            "in the summary table only."
        ),
    }
    summary_json = args.output_dir / "summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary_payload, f, indent=2)
    logger.info("Wrote %s", summary_json)

    md = render_markdown(
        n_total=len(rows_out),
        n_regression=n_reg,
        n_other=n_other,
        summary=summary,
        figures_rel=figures_rel,
    )
    md_path = args.output_dir / "summary.md"
    md_path.write_text(md)
    logger.info("Wrote %s", md_path)

    # ---- console summary ----
    print()
    print(f"=== Regression vs other (n_reg={n_reg}, n_other={n_other}) ===")
    print(f"  {'feature':45s}  {'regr med':>9s}  {'other med':>9s}  {'p':>8s}  {'r':>7s}")
    for key, label in FEATURES:
        s = summary[key]
        print(
            f"  {label:45s}  "
            f"{s['regression']['median']:>9.1f}  "
            f"{s['other']['median']:>9.1f}  "
            f"{s['mannwhitney']['p_two_sided']:>8.3g}  "
            f"{s['mannwhitney']['effect_size_r']:>+7.3f}"
        )


if __name__ == "__main__":
    main()
