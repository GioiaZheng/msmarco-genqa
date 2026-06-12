"""Retrieval quality reporting helpers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from msmarco_genqa.evaluation.retrieval import (
    compare_retrieval_runs_per_query,
    evaluate_retrieval,
)
from msmarco_genqa.reranking.io import read_run_tsv


RunDocs = Mapping[str, Sequence[str]]
Qrels = Mapping[str, set[str]]


def load_qrels_tsv(path: Path | str) -> dict[str, set[str]]:
    """Load positive qrels from a compact or TREC-format TSV/space file."""
    qrels: dict[str, set[str]] = {}
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 3:
                qid, doc_id, rel_text = parts
            elif len(parts) >= 4:
                qid, _iter, doc_id, rel_text = parts[:4]
            else:
                raise ValueError(
                    f"{p}:{line_number}: expected 3 or 4+ qrels columns, got {len(parts)}"
                )
            try:
                rel = float(rel_text)
            except ValueError as exc:
                raise ValueError(
                    f"{p}:{line_number}: relevance is not numeric: {rel_text!r}"
                ) from exc
            if rel > 0:
                qrels.setdefault(qid, set()).add(doc_id)
            else:
                qrels.setdefault(qid, set())
    return qrels


def read_run_doc_ids(path: Path | str) -> dict[str, list[str]]:
    """Read a standard run.tsv file as ``{qid: [doc_id, ...]}``."""
    return {
        qid: [doc_id for doc_id, _score in docs]
        for qid, docs in read_run_tsv(path).items()
    }


def qrels_with_positive_labels(qrels: Qrels) -> set[str]:
    return {qid for qid, doc_ids in qrels.items() if doc_ids}


def evaluate_run_report(
    runs: RunDocs,
    qrels: Qrels,
    *,
    run_name: str,
    ks_mrr: Sequence[int] = (10,),
    ks_ndcg: Sequence[int] = (10,),
    ks_recall: Sequence[int] = (100, 1000),
) -> dict[str, Any]:
    """Evaluate one run and attach coverage/skipped-qid diagnostics."""
    run_qids = set(runs)
    positive_qrels = qrels_with_positive_labels(qrels)
    metrics = evaluate_retrieval(
        {qid: list(doc_ids) for qid, doc_ids in runs.items()},
        dict(qrels),
        ks_mrr=ks_mrr,
        ks_ndcg=ks_ndcg,
        ks_recall=ks_recall,
    )
    n_evaluable = int(metrics.pop("n_queries", 0))
    missing_qrels = sorted(run_qids - set(qrels))
    empty_qrels = sorted(qid for qid in run_qids & set(qrels) if not qrels.get(qid))
    qrels_only = sorted(positive_qrels - run_qids)
    return {
        "run_name": run_name,
        "settings": {
            "ks_mrr": list(ks_mrr),
            "ks_ndcg": list(ks_ndcg),
            "ks_recall": list(ks_recall),
        },
        "coverage": {
            "n_run_qids": len(run_qids),
            "n_positive_qrels_qids": len(positive_qrels),
            "n_evaluable_qids": n_evaluable,
            "n_skipped_missing_qrels": len(missing_qrels),
            "n_skipped_empty_qrels": len(empty_qrels),
            "n_qrels_only": len(qrels_only),
        },
        "metrics": metrics,
    }


def compare_runs_report(
    baseline_runs: RunDocs,
    candidate_runs: RunDocs,
    qrels: Qrels,
    *,
    baseline_name: str,
    candidate_name: str,
    k_rank: int = 10,
    k_recall: int = 100,
    ks_mrr: Sequence[int] = (10,),
    ks_ndcg: Sequence[int] = (10,),
    ks_recall: Sequence[int] = (100, 1000),
) -> dict[str, Any]:
    """Evaluate two runs on matched qids and summarize deltas."""
    baseline_qids = set(baseline_runs)
    candidate_qids = set(candidate_runs)
    shared_qids = baseline_qids & candidate_qids
    positive_qrels = qrels_with_positive_labels(qrels)
    matched_qids = sorted(shared_qids & positive_qrels)

    baseline_matched = {qid: list(baseline_runs[qid]) for qid in matched_qids}
    candidate_matched = {qid: list(candidate_runs[qid]) for qid in matched_qids}
    matched_qrels = {qid: set(qrels[qid]) for qid in matched_qids}

    baseline_summary = evaluate_run_report(
        baseline_matched,
        matched_qrels,
        run_name=baseline_name,
        ks_mrr=ks_mrr,
        ks_ndcg=ks_ndcg,
        ks_recall=ks_recall,
    )
    candidate_summary = evaluate_run_report(
        candidate_matched,
        matched_qrels,
        run_name=candidate_name,
        ks_mrr=ks_mrr,
        ks_ndcg=ks_ndcg,
        ks_recall=ks_recall,
    )
    baseline_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    deltas = {
        key: candidate_metrics[key] - baseline_metrics[key]
        for key in sorted(baseline_metrics.keys() & candidate_metrics.keys())
    }

    per_query = compare_retrieval_runs_per_query(
        baseline_matched,
        candidate_matched,
        matched_qrels,
        k_rank=k_rank,
        k_recall=k_recall,
    )
    bucket_counts = Counter(str(row["bucket"]) for row in per_query)
    rr_key = f"rr_delta@{k_rank}"
    recall_key = f"recall_delta@{k_recall}"

    return {
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "deltas": deltas,
        "coverage": {
            "n_baseline_qids": len(baseline_qids),
            "n_candidate_qids": len(candidate_qids),
            "n_shared_qids": len(shared_qids),
            "n_matched_evaluable_qids": len(matched_qids),
            "n_baseline_only_qids": len(baseline_qids - candidate_qids),
            "n_candidate_only_qids": len(candidate_qids - baseline_qids),
            "n_shared_without_positive_qrels": len(shared_qids - positive_qrels),
        },
        "diagnostics": {
            "k_rank": k_rank,
            "k_recall": k_recall,
            "buckets": dict(sorted(bucket_counts.items())),
            f"mean_{rr_key}": _mean([row[rr_key] for row in per_query]),
            f"mean_{recall_key}": _mean([row[recall_key] for row in per_query]),
        },
        "per_query": per_query,
    }


def compare_run_matrix_report(
    named_runs: Mapping[str, RunDocs],
    qrels: Qrels,
    *,
    baseline_name: str | None = None,
    k_rank: int = 10,
    k_recall: int = 100,
    ks_mrr: Sequence[int] = (10,),
    ks_ndcg: Sequence[int] = (10,),
    ks_recall: Sequence[int] = (100, 1000),
) -> dict[str, Any]:
    """Evaluate two or more runs on one shared qid set.

    The matrix view is intended for BM25 / dense / RRF / reranked comparisons:
    every metric is computed on qids that appear in every run and have a
    positive qrel, so table rows are directly comparable.
    """
    if len(named_runs) < 2:
        raise ValueError("matrix report requires at least two runs")
    names = list(named_runs)
    if len(set(names)) != len(names):
        raise ValueError("run names must be unique")
    baseline = baseline_name or names[0]
    if baseline not in named_runs:
        raise ValueError(f"baseline run {baseline!r} is not one of: {names}")

    run_qids = {name: set(runs) for name, runs in named_runs.items()}
    shared_qids = set.intersection(*run_qids.values())
    positive_qrels = qrels_with_positive_labels(qrels)
    matched_qids = sorted(shared_qids & positive_qrels)
    matched_qrels = {qid: set(qrels[qid]) for qid in matched_qids}

    run_summaries: dict[str, Any] = {}
    for name, runs in named_runs.items():
        matched_run = {qid: list(runs[qid]) for qid in matched_qids}
        run_summaries[name] = evaluate_run_report(
            matched_run,
            matched_qrels,
            run_name=name,
            ks_mrr=ks_mrr,
            ks_ndcg=ks_ndcg,
            ks_recall=ks_recall,
        )

    metric_keys = sorted(
        set.intersection(*(set(summary["metrics"]) for summary in run_summaries.values()))
    )
    baseline_metrics = run_summaries[baseline]["metrics"]
    deltas_vs_baseline = {
        name: {
            key: run_summaries[name]["metrics"][key] - baseline_metrics[key]
            for key in metric_keys
        }
        for name in names
        if name != baseline
    }

    pairwise_rows = []
    diagnostics_vs_baseline: dict[str, Any] = {}
    baseline_docs = {qid: list(named_runs[baseline][qid]) for qid in matched_qids}
    rr_key = f"rr_delta@{k_rank}"
    recall_key = f"recall_delta@{k_recall}"
    for name in names:
        if name == baseline:
            continue
        candidate_docs = {qid: list(named_runs[name][qid]) for qid in matched_qids}
        deltas = {
            key: run_summaries[name]["metrics"][key] - baseline_metrics[key]
            for key in metric_keys
        }
        per_query = compare_retrieval_runs_per_query(
            baseline_docs,
            candidate_docs,
            matched_qrels,
            k_rank=k_rank,
            k_recall=k_recall,
        )
        bucket_counts = Counter(str(row["bucket"]) for row in per_query)
        diagnostics_vs_baseline[name] = {
            "buckets": dict(sorted(bucket_counts.items())),
            f"mean_{rr_key}": _mean([row[rr_key] for row in per_query]),
            f"mean_{recall_key}": _mean([row[recall_key] for row in per_query]),
        }
        pairwise_rows.append(
            {
                "baseline": baseline,
                "candidate": name,
                "deltas": deltas,
                "diagnostics": diagnostics_vs_baseline[name],
            }
        )

    best_by_metric: dict[str, Any] = {}
    for key in metric_keys:
        best_name = max(names, key=lambda name: float(run_summaries[name]["metrics"][key]))
        best_by_metric[key] = {
            "run_name": best_name,
            "value": run_summaries[best_name]["metrics"][key],
        }

    return {
        "baseline": baseline,
        "run_order": names,
        "runs": run_summaries,
        "metric_keys": metric_keys,
        "deltas_vs_baseline": deltas_vs_baseline,
        "diagnostics_vs_baseline": diagnostics_vs_baseline,
        "pairwise_rows": pairwise_rows,
        "best_by_metric": best_by_metric,
        "coverage": {
            "n_input_runs": len(names),
            "n_positive_qrels_qids": len(positive_qrels),
            "n_shared_qids": len(shared_qids),
            "n_matched_evaluable_qids": len(matched_qids),
            "n_qids_not_shared_by_run": {
                name: len(run_qids[name] - shared_qids) for name in names
            },
            "n_shared_without_positive_qrels": len(shared_qids - positive_qrels),
        },
        "settings": {
            "k_rank": k_rank,
            "k_recall": k_recall,
            "ks_mrr": list(ks_mrr),
            "ks_ndcg": list(ks_ndcg),
            "ks_recall": list(ks_recall),
        },
    }


def _mean(values: Sequence[float | int]) -> float:
    return mean([float(value) for value in values]) if values else 0.0


def render_single_run_markdown(report: Mapping[str, Any], *, run_path: str, qrels_path: str) -> str:
    lines = [
        "# Retrieval quality report",
        "",
        f"- Run: `{run_path}`",
        f"- Qrels: `{qrels_path}`",
        f"- Run name: `{report['run_name']}`",
        f"- Evaluable qids: **{report['coverage']['n_evaluable_qids']}**",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in sorted(report["metrics"].items()):
        lines.append(f"| {key} | {float(value):.4f} |")
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| field | value |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(report["coverage"].items()):
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines) + "\n"


def render_comparison_markdown(report: Mapping[str, Any], *, qrels_path: str) -> str:
    baseline = report["baseline"]
    candidate = report["candidate"]
    lines = [
        "# Retrieval comparison report",
        "",
        f"- Baseline: `{baseline['run_name']}`",
        f"- Candidate: `{candidate['run_name']}`",
        f"- Qrels: `{qrels_path}`",
        f"- Matched evaluable qids: **{report['coverage']['n_matched_evaluable_qids']}**",
        "",
        "| metric | baseline | candidate | delta |",
        "|---|---:|---:|---:|",
    ]
    for key in sorted(report["deltas"]):
        b = float(baseline["metrics"][key])
        c = float(candidate["metrics"][key])
        d = float(report["deltas"][key])
        lines.append(f"| {key} | {b:.4f} | {c:.4f} | {d:+.4f} |")
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| field | value |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(report["coverage"].items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Movement buckets",
            "",
            "| bucket | qids |",
            "|---|---:|",
        ]
    )
    for bucket, count in sorted(report["diagnostics"]["buckets"].items()):
        lines.append(f"| {bucket} | {count} |")
    lines.append("")
    lines.append("Metric deltas are candidate minus baseline on the matched qid set.")
    return "\n".join(lines) + "\n"


def render_matrix_markdown(report: Mapping[str, Any], *, qrels_path: str) -> str:
    metric_keys = list(report["metric_keys"])
    lines = [
        "# Retrieval matrix report",
        "",
        f"- Baseline for deltas: `{report['baseline']}`",
        f"- Qrels: `{qrels_path}`",
        f"- Matched evaluable qids: **{report['coverage']['n_matched_evaluable_qids']}**",
        "",
        "## Metrics",
        "",
        "| run | " + " | ".join(metric_keys) + " |",
        "|---|" + "|".join("---:" for _ in metric_keys) + "|",
    ]
    for name in report["run_order"]:
        metrics = report["runs"][name]["metrics"]
        values = " | ".join(f"{float(metrics[key]):.4f}" for key in metric_keys)
        lines.append(f"| {name} | {values} |")

    lines.extend(
        [
            "",
            "## Delta vs baseline",
            "",
            "| candidate | " + " | ".join(metric_keys) + " |",
            "|---|" + "|".join("---:" for _ in metric_keys) + "|",
        ]
    )
    for name in report["run_order"]:
        if name == report["baseline"]:
            continue
        deltas = report["deltas_vs_baseline"][name]
        values = " | ".join(f"{float(deltas[key]):+.4f}" for key in metric_keys)
        lines.append(f"| {name} | {values} |")

    lines.extend(
        [
            "",
            "## Best run by metric",
            "",
            "| metric | run | value |",
            "|---|---|---:|",
        ]
    )
    for key in metric_keys:
        best = report["best_by_metric"][key]
        lines.append(f"| {key} | {best['run_name']} | {float(best['value']):.4f} |")

    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| field | value |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(report["coverage"].items()):
        if isinstance(value, Mapping):
            continue
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append(
        "All metric rows use the qids shared by every input run and positive qrels; "
        "candidate deltas are candidate minus baseline on that same qid set."
    )
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
