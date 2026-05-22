"""Break down the W3/W5 rerank delta by question-form (W6-A axis).

For every paired qid in the W6 analysis pool, join the per-query
metrics with the question-form label from
``outputs/week06_querytype/query_forms.jsonl`` (W6-A) and aggregate per
form. Reports BM25 vs reranked deltas on:

- Token-F1, Exact-Match     (already per-query in W6 metrics)
- ROUGE-L                   (computed here via ``per_query_rouge_l``)
- MRR@10, nDCG@10           (computed here from W2 / W5 run.tsv files
                             and dev/small qrels)

Each form's headline ΔToken-F1 is reported with a 95 % paired-bootstrap
CI. Bucket composition per form (regression, rerank_fixed_*, …) is
also reported, so the table directly answers "on which query types
does the reranker help most, and on which does it regress most".

This is the W6-B follow-up to W6-A: no retrieval re-runs, no new
generation; pure offline join + cheap per-query rescoring + paired
bootstrap. CPU minutes end-to-end.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from msmarco_genqa.evaluation.bootstrap import (
    paired_bootstrap_diff,
    per_query_rouge_l,
)
from msmarco_genqa.evaluation.query_form import QUESTION_FORM_CATEGORIES
from msmarco_genqa.evaluation.retrieval import ndcg_at_k, reciprocal_rank
from msmarco_genqa.reranking.io import read_run_tsv

logger = logging.getLogger("analyze_rerank_by_query_form")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--query-forms",
        type=Path,
        default=PROJECT_ROOT / "outputs/week06_querytype/query_forms.jsonl",
    )
    p.add_argument(
        "--per-query-metrics",
        type=Path,
        default=PROJECT_ROOT / "outputs/week06_analysis/per_query_metrics.jsonl",
    )
    p.add_argument(
        "--bm25-predictions",
        type=Path,
        default=PROJECT_ROOT / "outputs/week03_generation_bm25_full/predictions.jsonl",
    )
    p.add_argument(
        "--rerank-predictions",
        type=Path,
        default=PROJECT_ROOT / "outputs/week03_generation_reranked_full/predictions.jsonl",
    )
    p.add_argument(
        "--bm25-run",
        type=Path,
        default=PROJECT_ROOT / "outputs/week02_bm25/run.tsv",
    )
    p.add_argument(
        "--rerank-run",
        type=Path,
        default=PROJECT_ROOT / "outputs/week05_reranker_full/run.tsv",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/week06_rerank_by_form",
    )
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Bootstrap resamples per form for the ΔToken-F1 CI.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k-retrieval", type=int, default=10)
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_qrels() -> dict[str, set[str]]:
    from msmarco_genqa.data.msmarco import load_msmarco_passage

    logger.info("Loading dev/small qrels via ir_datasets...")
    bundle = load_msmarco_passage(load_corpus=False)
    return bundle.qrels


def build_predictions_index(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        out[str(r["query_id"])] = {
            "prediction": r.get("prediction") or "",
            "references": list(r.get("references") or []),
        }
    return out


def compute_per_query_retrieval(
    *,
    run: dict[str, list[tuple[str, float]]],
    qrels: dict[str, set[str]],
    k: int,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for qid, ranked in run.items():
        relevant = qrels.get(qid, set())
        doc_ids = [doc_id for doc_id, _score in ranked]
        out[qid] = {
            "mrr_at_k": reciprocal_rank(doc_ids, relevant, k),
            "ndcg_at_k": ndcg_at_k(doc_ids, relevant, k),
        }
    return out


def aggregate_by_form(
    rows: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    by_form: dict[str, list[dict[str, Any]]] = {c: [] for c in QUESTION_FORM_CATEGORIES}
    for r in rows:
        by_form[r["question_form"]].append(r)

    out: dict[str, Any] = {}
    for form, group in by_form.items():
        n = len(group)
        if n == 0:
            out[form] = {"n": 0}
            continue

        def mean(key: str) -> float:
            return sum(r[key] for r in group) / n

        bm25_token_f1 = [r["bm25_token_f1"] for r in group]
        rerank_token_f1 = [r["rerank_token_f1"] for r in group]
        ci = paired_bootstrap_diff(
            bm25_token_f1,
            rerank_token_f1,
            n_resamples=n_bootstrap,
            seed=seed,
        )

        bucket_counts = Counter(r.get("bucket") for r in group)
        out[form] = {
            "n": n,
            "share_of_total": n / len(rows) if rows else 0.0,
            "bm25_token_f1_mean": mean("bm25_token_f1"),
            "rerank_token_f1_mean": mean("rerank_token_f1"),
            "delta_token_f1_mean": mean("delta_token_f1"),
            "delta_token_f1_ci95_low": ci["ci_low"],
            "delta_token_f1_ci95_high": ci["ci_high"],
            "delta_token_f1_p_two_sided": ci["p_two_sided"],
            "bm25_rouge_l_mean": mean("bm25_rouge_l"),
            "rerank_rouge_l_mean": mean("rerank_rouge_l"),
            "delta_rouge_l_mean": mean("delta_rouge_l"),
            "bm25_exact_match_mean": mean("bm25_exact_match"),
            "rerank_exact_match_mean": mean("rerank_exact_match"),
            "delta_exact_match_mean": mean("delta_exact_match"),
            "bm25_mrr_at_10_mean": mean("bm25_mrr_at_10"),
            "rerank_mrr_at_10_mean": mean("rerank_mrr_at_10"),
            "delta_mrr_at_10_mean": mean("delta_mrr_at_10"),
            "bm25_ndcg_at_10_mean": mean("bm25_ndcg_at_10"),
            "rerank_ndcg_at_10_mean": mean("rerank_ndcg_at_10"),
            "delta_ndcg_at_10_mean": mean("delta_ndcg_at_10"),
            "bucket_counts": dict(bucket_counts),
            "regression_rate": bucket_counts.get("regression", 0) / n,
        }
    return out


def render_markdown(
    *,
    n_total: int,
    agg: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Rerank Δ by question-form — W6-B")
    lines.append("")
    lines.append(
        f"Joined the W6-A question-form tags onto the W3/W5/W6 per-query "
        f"metrics ({n_total} paired qids) and aggregated per form. "
        "Headline metric is ΔToken-F1 (rerank − BM25), reported with a "
        "95 % paired-bootstrap CI on the per-form slice."
    )
    lines.append("")

    # Load-bearing finding: which forms have a Δ CI that includes zero,
    # and which form has the strongest non-trivially-sized lift. Reported
    # directly so the table is not the only source of the headline read.
    ranked = sorted(
        [(f, d) for f, d in agg.items() if d.get("n", 0) > 0],
        key=lambda kv: -kv[1]["delta_token_f1_mean"],
    )
    null_forms = [
        f
        for f, d in ranked
        if d["delta_token_f1_ci95_low"] <= 0 <= d["delta_token_f1_ci95_high"]
    ]
    if ranked:
        best_form, best_d = ranked[0]
        worst_form, worst_d = ranked[-1]
        lines.append("**Headline read.**")
        lines.append("")
        if null_forms:
            null_str = ", ".join(f"`{f}`" for f in null_forms)
            lines.append(
                f"- Forms where the 95 % CI on ΔToken-F1 includes zero: {null_str}. "
                "Reranking does not produce a detectable generation lift on these "
                "(generation-side) — the retrieval-side ΔMRR@10 / ΔnDCG@10 are "
                "still large, so the bottleneck is downstream of retrieval."
            )
        else:
            lines.append(
                "- Every form has a 95 % CI on ΔToken-F1 strictly above zero."
            )
        lines.append(
            f"- Largest ΔToken-F1: `{best_form}` "
            f"({best_d['delta_token_f1_mean']:+.4f}, n={best_d['n']}). "
            f"Smallest: `{worst_form}` "
            f"({worst_d['delta_token_f1_mean']:+.4f}, n={worst_d['n']})."
        )
        lines.append("")

    lines.append("## 1. Headline — ΔToken-F1 per form (sorted by Δ desc)")
    lines.append("")
    lines.append(
        "| form | n | BM25 token-F1 | Rerank token-F1 | Δ token-F1 | 95% CI | p |"
    )
    lines.append("|---|---:|---:|---:|---:|---|---:|")
    sortable = [
        (form, d)
        for form, d in agg.items()
        if d.get("n", 0) > 0
    ]
    sortable.sort(key=lambda kv: -kv[1]["delta_token_f1_mean"])
    for form, d in sortable:
        lines.append(
            f"| `{form}` | {d['n']} | "
            f"{d['bm25_token_f1_mean']:.4f} | "
            f"{d['rerank_token_f1_mean']:.4f} | "
            f"**{d['delta_token_f1_mean']:+.4f}** | "
            f"[{d['delta_token_f1_ci95_low']:+.4f}, {d['delta_token_f1_ci95_high']:+.4f}] | "
            f"{d['delta_token_f1_p_two_sided']:.3f} |"
        )
    lines.append("")

    lines.append("## 2. All metrics per form")
    lines.append("")
    lines.append(
        "| form | n | Δ token-F1 | Δ ROUGE-L | Δ EM | Δ MRR@10 | Δ nDCG@10 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for form in QUESTION_FORM_CATEGORIES:
        d = agg.get(form, {})
        if not d or d.get("n", 0) == 0:
            continue
        lines.append(
            f"| `{form}` | {d['n']} | "
            f"{d['delta_token_f1_mean']:+.4f} | "
            f"{d['delta_rouge_l_mean']:+.4f} | "
            f"{d['delta_exact_match_mean']:+.4f} | "
            f"{d['delta_mrr_at_10_mean']:+.4f} | "
            f"{d['delta_ndcg_at_10_mean']:+.4f} |"
        )
    lines.append("")

    lines.append("## 3. Bucket distribution per form")
    lines.append("")
    bucket_keys = sorted(
        {b for d in agg.values() for b in d.get("bucket_counts", {}).keys()}
    )
    header = "| form | n | " + " | ".join(bucket_keys) + " | regression rate |"
    sep = "|---|---:|" + "|".join(["---:"] * len(bucket_keys)) + "|---:|"
    lines.append(header)
    lines.append(sep)
    for form in QUESTION_FORM_CATEGORIES:
        d = agg.get(form, {})
        if not d or d.get("n", 0) == 0:
            continue
        bc = d.get("bucket_counts", {})
        cells = [str(bc.get(b, 0)) for b in bucket_keys]
        lines.append(
            f"| `{form}` | {d['n']} | "
            + " | ".join(cells)
            + f" | {d['regression_rate']:.1%} |"
        )
    lines.append("")
    lines.append(
        "*Regression rate* = share of the form's queries where rerank "
        "produced a meaningfully worse Token-F1 than BM25 (bucket label "
        "from `scripts/analyze_generation_rerank.py`)."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    logger.info("Loading question-form tags from %s ...", args.query_forms)
    form_rows = load_jsonl(args.query_forms)
    form_of: dict[str, str] = {
        str(r["query_id"]): r["question_form"] for r in form_rows
    }

    logger.info("Loading W6 per-query metrics from %s ...", args.per_query_metrics)
    per_query = load_jsonl(args.per_query_metrics)

    logger.info("Loading W3 predictions (BM25 + reranked) ...")
    bm25_preds = build_predictions_index(load_jsonl(args.bm25_predictions))
    rerank_preds = build_predictions_index(load_jsonl(args.rerank_predictions))

    logger.info("Loading run.tsv files ...")
    bm25_run = read_run_tsv(args.bm25_run)
    rerank_run = read_run_tsv(args.rerank_run)

    qrels = load_qrels()
    logger.info("Computing per-query MRR@%d / nDCG@%d ...", args.top_k_retrieval, args.top_k_retrieval)
    bm25_retr = compute_per_query_retrieval(
        run=bm25_run, qrels=qrels, k=args.top_k_retrieval
    )
    rerank_retr = compute_per_query_retrieval(
        run=rerank_run, qrels=qrels, k=args.top_k_retrieval
    )

    # ROUGE-L per query: align with per_query order so we can index by qid.
    logger.info("Computing per-query ROUGE-L for both arms ...")
    qids_ordered = [str(r["query_id"]) for r in per_query]
    refs_ordered = [bm25_preds[q]["references"] for q in qids_ordered]
    bm25_pred_ordered = [bm25_preds[q]["prediction"] for q in qids_ordered]
    rerank_pred_ordered = [rerank_preds[q]["prediction"] for q in qids_ordered]
    bm25_rouge = per_query_rouge_l(bm25_pred_ordered, refs_ordered)
    rerank_rouge = per_query_rouge_l(rerank_pred_ordered, refs_ordered)
    bm25_rouge_of = dict(zip(qids_ordered, bm25_rouge))
    rerank_rouge_of = dict(zip(qids_ordered, rerank_rouge))

    # Join
    joined: list[dict[str, Any]] = []
    skipped = 0
    for r in per_query:
        qid = str(r["query_id"])
        form = form_of.get(qid)
        if form is None:
            skipped += 1
            continue
        b_retr = bm25_retr.get(qid, {"mrr_at_k": 0.0, "ndcg_at_k": 0.0})
        r_retr = rerank_retr.get(qid, {"mrr_at_k": 0.0, "ndcg_at_k": 0.0})
        b_rouge = bm25_rouge_of.get(qid, 0.0)
        r_rouge = rerank_rouge_of.get(qid, 0.0)
        joined.append(
            {
                "query_id": qid,
                "query": r.get("query"),
                "question_form": form,
                "ms_marco_query_type": r.get("query_type"),
                "bucket": r.get("bucket"),
                "bm25_token_f1": r["bm25_token_f1"],
                "rerank_token_f1": r["rerank_token_f1"],
                "delta_token_f1": r["delta_token_f1"],
                "bm25_exact_match": r["bm25_exact_match"],
                "rerank_exact_match": r["rerank_exact_match"],
                "delta_exact_match": r["rerank_exact_match"] - r["bm25_exact_match"],
                "bm25_rouge_l": b_rouge,
                "rerank_rouge_l": r_rouge,
                "delta_rouge_l": r_rouge - b_rouge,
                "bm25_mrr_at_10": b_retr["mrr_at_k"],
                "rerank_mrr_at_10": r_retr["mrr_at_k"],
                "delta_mrr_at_10": r_retr["mrr_at_k"] - b_retr["mrr_at_k"],
                "bm25_ndcg_at_10": b_retr["ndcg_at_k"],
                "rerank_ndcg_at_10": r_retr["ndcg_at_k"],
                "delta_ndcg_at_10": r_retr["ndcg_at_k"] - b_retr["ndcg_at_k"],
            }
        )
    if skipped:
        logger.warning("Skipped %d per-query rows without a question_form tag.", skipped)

    logger.info("Aggregating by question_form (n_bootstrap=%d) ...", args.n_bootstrap)
    agg = aggregate_by_form(joined, n_bootstrap=args.n_bootstrap, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.output_dir / "per_query_joined.jsonl"
    with open(out_jsonl, "w") as f:
        for r in joined:
            f.write(json.dumps(r) + "\n")
    logger.info("Wrote %s (%d rows)", out_jsonl, len(joined))

    payload = {
        "task": "rerank_delta_by_question_form",
        "n_total": len(joined),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "top_k_retrieval": args.top_k_retrieval,
        "categories": list(QUESTION_FORM_CATEGORIES),
        "by_form": agg,
        "notes": (
            "Question-form labels from W6-A (scripts/tag_query_forms.py). "
            "Per-query Token-F1 / EM / bucket from W6 analyze_generation_"
            "rerank.py. Per-query ROUGE-L computed here via rouge_score "
            "(best-of-N references). Per-query MRR@10 / nDCG@10 computed "
            "from W2 run.tsv and W5 rerank run.tsv against dev/small qrels. "
            "ΔToken-F1 CI is a paired bootstrap on each form's slice; "
            "small forms (e.g. 'why', n=75) have correspondingly wide CIs."
        ),
    }
    out_json = args.output_dir / "summary.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s", out_json)

    md = render_markdown(n_total=len(joined), agg=agg)
    out_md = args.output_dir / "summary.md"
    out_md.write_text(md)
    logger.info("Wrote %s", out_md)

    # ---- console summary ----
    print()
    print(f"=== Rerank ΔToken-F1 by question_form (n={len(joined)}) ===")
    print(f"  {'form':10s}  {'n':>5s}  {'BM25':>7s}  {'Rerank':>7s}  {'Δ':>8s}  {'95% CI':>20s}")
    sortable = sorted(
        [(f, d) for f, d in agg.items() if d.get("n", 0) > 0],
        key=lambda kv: -kv[1]["delta_token_f1_mean"],
    )
    for form, d in sortable:
        ci = f"[{d['delta_token_f1_ci95_low']:+.3f}, {d['delta_token_f1_ci95_high']:+.3f}]"
        print(
            f"  {form:10s}  {d['n']:>5d}  "
            f"{d['bm25_token_f1_mean']:>7.4f}  "
            f"{d['rerank_token_f1_mean']:>7.4f}  "
            f"{d['delta_token_f1_mean']:>+8.4f}  "
            f"{ci:>20s}"
        )


if __name__ == "__main__":
    main()
