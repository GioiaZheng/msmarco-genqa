"""Query-level retrieval lift analysis for two TREC-format run files.

The aggregate W5 result says whether reranking improves MRR. This script
answers the follow-up question: *where did the lift come from?* It compares
two ranked runs against qrels and writes bucketed diagnostics for promoted,
demoted, newly recovered, and lost queries.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from msmarco_genqa.evaluation.retrieval import compare_retrieval_runs_per_query
from msmarco_genqa.reranking.io import read_run_tsv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--before-run",
        type=Path,
        default=PROJECT_ROOT / "outputs/week04_dense/run.tsv",
        help="Baseline TREC run.tsv, e.g. dense or BM25 first-stage output.",
    )
    p.add_argument(
        "--after-run",
        type=Path,
        default=PROJECT_ROOT / "outputs/week05_reranker_full/run.tsv",
        help="Comparison TREC run.tsv, e.g. cross-encoder reranked output.",
    )
    p.add_argument(
        "--qrels",
        type=Path,
        default=None,
        help=(
            "Optional qrels TSV file. Supports TREC 4-column qrels "
            "(qid, iter, docid, rel) or compact 3-column (qid, docid, rel). "
            "If omitted, dev/small qrels are loaded through ir_datasets."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/week05_retrieval_lift_analysis",
    )
    p.add_argument("--k-rank", type=int, default=10)
    p.add_argument("--k-recall", type=int, default=100)
    p.add_argument("--examples-per-bucket", type=int, default=8)
    return p.parse_args()


def load_qrels_tsv(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                qid, doc_id, rel = parts[0], parts[2], parts[3]
            elif len(parts) == 3:
                qid, doc_id, rel = parts
            elif len(parts) == 2:
                qid, doc_id = parts
                rel = "1"
            else:
                continue
            try:
                is_relevant = float(rel) > 0
            except ValueError:
                is_relevant = False
            if is_relevant:
                qrels[qid].add(doc_id)
    return dict(qrels)


def load_qrels(path: Path | None) -> dict[str, set[str]]:
    if path is not None:
        return load_qrels_tsv(path)

    from msmarco_genqa.data.msmarco import load_msmarco_passage

    bundle = load_msmarco_passage(load_corpus=False)
    return bundle.qrels


def doc_ids_only(
    run: dict[str, list[tuple[str, float]]],
) -> dict[str, list[str]]:
    return {qid: [doc_id for doc_id, _score in docs] for qid, docs in run.items()}


def average(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return mean(vals) if vals else 0.0


def summarise(rows: list[dict[str, Any]], *, k_rank: int, k_recall: int) -> dict[str, Any]:
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[str(row["bucket"])].append(row)

    bucket_rows: list[dict[str, Any]] = []
    total = len(rows)
    rr_key = f"rr_delta@{k_rank}"
    recall_key = f"recall_delta@{k_recall}"
    for bucket in sorted(by_bucket):
        items = by_bucket[bucket]
        bucket_rows.append(
            {
                "bucket": bucket,
                "n_queries": len(items),
                "share": len(items) / total if total else 0.0,
                f"mean_{rr_key}": average(items, rr_key),
                f"mean_{recall_key}": average(items, recall_key),
                "mean_rank_movement": average(items, "rank_movement"),
            }
        )

    counts = Counter(str(r["bucket"]) for r in rows)
    return {
        "n_queries": total,
        "buckets": dict(counts),
        f"mean_rr_delta@{k_rank}": average(rows, rr_key),
        f"mean_recall_delta@{k_recall}": average(rows, recall_key),
        "bucket_summary": bucket_rows,
    }


def top_examples(
    rows: list[dict[str, Any]],
    *,
    k_rank: int,
    per_bucket: int,
) -> list[dict[str, Any]]:
    rr_key = f"rr_delta@{k_rank}"
    out: list[dict[str, Any]] = []
    buckets = sorted({str(r["bucket"]) for r in rows})
    for bucket in buckets:
        items = [r for r in rows if r["bucket"] == bucket]
        items.sort(key=lambda r: abs(float(r[rr_key])), reverse=True)
        out.extend(items[:per_bucket])
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_bucket_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(
    summary: dict[str, Any],
    examples: list[dict[str, Any]],
    *,
    before_run: Path,
    after_run: Path,
    qrels: Path | None,
    k_rank: int,
    k_recall: int,
) -> str:
    rr_key = f"rr_delta@{k_rank}"
    recall_key = f"recall_delta@{k_recall}"
    lines = [
        "# Retrieval lift analysis",
        "",
        f"- Before run: `{before_run}`",
        f"- After run: `{after_run}`",
        f"- Qrels: `{qrels}`" if qrels else "- Qrels: MS MARCO dev/small via `ir_datasets`",
        f"- Evaluable queries: **{summary['n_queries']}**",
        f"- Mean RR delta @{k_rank}: **{summary[f'mean_rr_delta@{k_rank}']:+.4f}**",
        f"- Mean recall delta @{k_recall}: **{summary[f'mean_recall_delta@{k_recall}']:+.4f}**",
        "",
        "## Bucket summary",
        "",
        "| bucket | n queries | share | mean RR delta | mean recall delta | mean rank movement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["bucket_summary"]:
        lines.append(
            f"| {row['bucket']} | {row['n_queries']} | {row['share']:.1%} | "
            f"{row[f'mean_{rr_key}']:+.4f} | {row[f'mean_{recall_key}']:+.4f} | "
            f"{row['mean_rank_movement']:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Diagnostic examples",
            "",
            "| qid | bucket | before rank | after rank | rank movement | RR delta |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in examples[: min(len(examples), 40)]:
        before_rank = row["before_first_relevant_rank"] or "-"
        after_rank = row["after_first_relevant_rank"] or "-"
        movement = row["rank_movement"] if row["rank_movement"] is not None else "-"
        lines.append(
            f"| {row['qid']} | {row['bucket']} | {before_rank} | {after_rank} | "
            f"{movement} | {float(row[rr_key]):+.4f} |"
        )

    lines.extend(
        [
            "",
            "Positive rank movement means the first relevant document moved upward.",
            "The JSONL artifact contains the full per-query table for deeper slicing.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    before = doc_ids_only(read_run_tsv(args.before_run))
    after = doc_ids_only(read_run_tsv(args.after_run))
    qrels = load_qrels(args.qrels)

    rows = compare_retrieval_runs_per_query(
        before,
        after,
        qrels,
        k_rank=args.k_rank,
        k_recall=args.k_recall,
    )
    summary = summarise(rows, k_rank=args.k_rank, k_recall=args.k_recall)
    examples = top_examples(
        rows,
        k_rank=args.k_rank,
        per_bucket=args.examples_per_bucket,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "retrieval_lift.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_bucket_csv(args.output_dir / "retrieval_lift_by_bucket.csv", summary["bucket_summary"])
    write_jsonl(args.output_dir / "retrieval_lift_examples.jsonl", examples)
    (args.output_dir / "retrieval_lift.md").write_text(
        render_markdown(
            summary,
            examples,
            before_run=args.before_run,
            after_run=args.after_run,
            qrels=args.qrels,
            k_rank=args.k_rank,
            k_recall=args.k_recall,
        )
    )


if __name__ == "__main__":
    main()
