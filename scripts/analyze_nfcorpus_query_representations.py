#!/usr/bin/env python3
"""Validate and compare the three frozen NFCorpus video query representations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msmarco_genqa.data.benchmark import load_benchmark_queries
from msmarco_genqa.evaluation.bootstrap import paired_bootstrap_diff
from msmarco_genqa.evaluation.retrieval import (
    first_relevant_rank,
    recall_at_k,
    reciprocal_rank,
)
from msmarco_genqa.evaluation.trec import (
    compare_metric_sets,
    evaluate_ir_measures,
    evaluate_trec_retrieval,
    graded_ndcg_at_k,
)
from msmarco_genqa.reranking.io import read_run_tsv


REPRESENTATIONS = ("title", "description", "title_plus_description")
METRICS = ("mrr@10", "ndcg@10", "recall@100", "recall@1000")
PER_QUERY_METRICS = ("rr@10", "ndcg@10", "recall@100", "recall@1000")


@dataclass(frozen=True)
class Condition:
    name: str
    directory: Path
    run: dict[str, list[tuple[str, float]]]
    metrics_payload: dict[str, object]
    manifest: dict[str, object]
    query_summary: dict[str, object]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/beir_nfcorpus_video",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/analysis/nfcorpus_video_query_representation",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON artifact {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def load_condition(input_root: Path, representation: str) -> Condition:
    directory = input_root / representation / "bm25"
    return Condition(
        name=representation,
        directory=directory,
        run=read_run_tsv(directory / "run.tsv"),
        metrics_payload=_read_json(directory / "metrics.json"),
        manifest=_read_json(directory / "manifest.json"),
        query_summary=_read_json(directory / "query_representation/summary.json"),
    )


def validate_shared_contract(
    conditions: Mapping[str, Condition],
    *,
    expected_queries: int = 102,
    expected_depth: int = 1000,
) -> dict[str, object]:
    if set(conditions) != set(REPRESENTATIONS):
        raise ValueError(f"expected conditions {REPRESENTATIONS}")

    qid_sets = {name: set(condition.run) for name, condition in conditions.items()}
    reference_qids = qid_sets["title"]
    if len(reference_qids) != expected_queries:
        raise ValueError(
            f"title condition has {len(reference_qids)} qids; expected {expected_queries}"
        )
    if any(qids != reference_qids for qids in qid_sets.values()):
        raise ValueError("query-id sets differ across representation conditions")

    commits: set[str] = set()
    index_hashes: set[str] = set()
    cohort_hashes: set[str] = set()
    record_hashes: set[str] = set()
    for name, condition in conditions.items():
        if condition.query_summary.get("representation") != name:
            raise ValueError(f"{name}: query summary names another representation")
        if any(len(rows) != expected_depth for rows in condition.run.values()):
            raise ValueError(f"{name}: not every run block has depth {expected_depth}")
        git = condition.manifest.get("git")
        if not isinstance(git, dict) or git.get("dirty") is not False:
            raise ValueError(f"{name}: canonical manifest must record a clean git tree")
        commit = git.get("commit")
        if not isinstance(commit, str) or not commit:
            raise ValueError(f"{name}: manifest git commit is missing")
        commits.add(commit)

        fingerprint = condition.query_summary.get("index_fingerprint")
        if not isinstance(fingerprint, dict):
            raise ValueError(f"{name}: index fingerprint is missing")
        index_hash = fingerprint.get("sha256")
        if not isinstance(index_hash, str) or len(index_hash) != 64:
            raise ValueError(f"{name}: index SHA-256 is invalid")
        index_hashes.add(index_hash)
        cohort_hashes.add(str(condition.query_summary.get("qid_sha256")))
        record_hashes.add(
            str(condition.query_summary.get("official_query_records_sha256"))
        )

    if len(commits) != 1:
        raise ValueError("conditions were produced from different commits")
    if len(index_hashes) != 1:
        raise ValueError("conditions were produced from different BM25 indexes")
    if len(cohort_hashes) != 1 or len(record_hashes) != 1:
        raise ValueError("conditions do not share the same query cohort/source records")

    return {
        "query_count": len(reference_qids),
        "run_depth": expected_depth,
        "git_commit": next(iter(commits)),
        "index_sha256": next(iter(index_hashes)),
        "qid_sha256": next(iter(cohort_hashes)),
        "official_query_records_sha256": next(iter(record_hashes)),
        "run_structure_valid": True,
    }


def build_per_query_rows(
    runs: Mapping[str, dict[str, list[tuple[str, float]]]],
    graded_qrels: Mapping[str, dict[str, int]],
) -> list[dict[str, object]]:
    qids = sorted(graded_qrels)
    if any(set(run) != set(qids) for run in runs.values()):
        raise ValueError("run qids must exactly match the evaluation qids")

    rows: list[dict[str, object]] = []
    for qid in qids:
        judgments = graded_qrels[qid]
        relevant = {
            doc_id for doc_id, relevance in judgments.items() if relevance >= 1
        }
        row: dict[str, object] = {
            "query_id": qid,
            "n_relevant": len(relevant),
            "conditions": {},
        }
        condition_metrics: dict[str, object] = {}
        for name, run in runs.items():
            ranked = [doc_id for doc_id, _score in run[qid]]
            condition_metrics[name] = {
                "rr@10": reciprocal_rank(ranked, relevant, 10),
                "ndcg@10": graded_ndcg_at_k(ranked, judgments, k=10),
                "recall@100": recall_at_k(ranked, relevant, 100),
                "recall@1000": recall_at_k(ranked, relevant, 1000),
                "first_relevant_rank@100": first_relevant_rank(
                    ranked,
                    relevant,
                    100,
                ),
                "first_relevant_rank@1000": first_relevant_rank(
                    ranked,
                    relevant,
                    1000,
                ),
            }
        row["conditions"] = condition_metrics
        rows.append(row)
    return rows


def build_paired_comparisons(
    rows: Sequence[Mapping[str, object]],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, object]:
    pairs = (
        ("description_vs_title", "title", "description"),
        (
            "title_plus_description_vs_title",
            "title",
            "title_plus_description",
        ),
        (
            "title_plus_description_vs_description",
            "description",
            "title_plus_description",
        ),
    )
    comparisons: dict[str, object] = {}
    for label, baseline, treatment in pairs:
        metrics: dict[str, object] = {}
        for metric in PER_QUERY_METRICS:
            baseline_scores = [
                float(row["conditions"][baseline][metric])  # type: ignore[index]
                for row in rows
            ]
            treatment_scores = [
                float(row["conditions"][treatment][metric])  # type: ignore[index]
                for row in rows
            ]
            bootstrap = paired_bootstrap_diff(
                baseline_scores,
                treatment_scores,
                n_resamples=n_resamples,
                seed=seed,
            )
            deltas = [
                treatment_score - baseline_score
                for baseline_score, treatment_score in zip(
                    baseline_scores,
                    treatment_scores,
                )
            ]
            bootstrap["wins"] = sum(delta > 0 for delta in deltas)
            bootstrap["ties"] = sum(delta == 0 for delta in deltas)
            bootstrap["losses"] = sum(delta < 0 for delta in deltas)
            metrics[metric] = bootstrap

        no_hit: dict[str, object] = {}
        for cutoff in (100, 1000):
            key = f"first_relevant_rank@{cutoff}"
            baseline_miss = [
                row["conditions"][baseline][key] is None  # type: ignore[index]
                for row in rows
            ]
            treatment_miss = [
                row["conditions"][treatment][key] is None  # type: ignore[index]
                for row in rows
            ]
            no_hit[f"@{cutoff}"] = {
                "baseline": sum(baseline_miss),
                "treatment": sum(treatment_miss),
                "recovered": sum(
                    before and not after
                    for before, after in zip(baseline_miss, treatment_miss)
                ),
                "lost": sum(
                    not before and after
                    for before, after in zip(baseline_miss, treatment_miss)
                ),
            }
        comparisons[label] = {
            "baseline": baseline,
            "treatment": treatment,
            "metrics": metrics,
            "no_hit_queries": no_hit,
        }
    return comparisons


def _condition_metrics(
    conditions: Mapping[str, Condition],
    graded_qrels: dict[str, dict[str, int]],
    *,
    tolerance: float,
) -> tuple[dict[str, object], dict[str, object]]:
    aggregate: dict[str, object] = {}
    cross_checks: dict[str, object] = {}
    for name, condition in conditions.items():
        ranked = {
            qid: [doc_id for doc_id, _score in rows]
            for qid, rows in condition.run.items()
        }
        internal = evaluate_trec_retrieval(
            ranked,
            graded_qrels,
            rel_threshold=1,
            ks_mrr=(10,),
            ks_ndcg=(10,),
            ks_recall=(100, 1000),
        )
        external = evaluate_ir_measures(
            condition.run,
            graded_qrels,
            rel_threshold=1,
        )
        deltas = compare_metric_sets(
            internal,
            external,
            tolerance=tolerance,
        )
        reported = condition.metrics_payload.get("metrics")
        if not isinstance(reported, dict):
            raise ValueError(f"{name}: metrics.json is missing metrics")
        report_deltas = compare_metric_sets(
            internal,
            {metric: float(reported[metric]) for metric in METRICS},
            tolerance=tolerance,
        )
        aggregate[name] = {metric: float(internal[metric]) for metric in METRICS}
        cross_checks[name] = {
            "backend": "ir-measures",
            "status": "passed",
            "external_metrics": external,
            "external_absolute_deltas": deltas,
            "reported_absolute_deltas": report_deltas,
        }
    return aggregate, cross_checks


def _format_p(value: float) -> str:
    return "< 0.0002" if value == 0.0 else f"= {value:.4f}"


def render_report(summary: Mapping[str, object]) -> str:
    aggregate = summary["aggregate_metrics"]
    comparisons = summary["paired_comparisons"]
    lines = [
        "# NFCorpus Video Query-Representation Analysis",
        "",
        "## Result",
        "",
        "| Representation | MRR@10 | nDCG@10 | Recall@100 | Recall@1000 |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "title": "Title",
        "description": "Description",
        "title_plus_description": "Title + description",
    }
    for name in REPRESENTATIONS:
        metrics = aggregate[name]
        lines.append(
            f"| {labels[name]} | {metrics['mrr@10']:.6f} | "
            f"{metrics['ndcg@10']:.6f} | {metrics['recall@100']:.6f} | "
            f"{metrics['recall@1000']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Paired comparison against title",
            "",
            "| Treatment | Metric | Delta | 95% CI | p (two-sided) | W/T/L |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for comparison_name in (
        "description_vs_title",
        "title_plus_description_vs_title",
    ):
        comparison = comparisons[comparison_name]
        treatment = labels[comparison["treatment"]]
        for metric in PER_QUERY_METRICS:
            result = comparison["metrics"][metric]
            lines.append(
                f"| {treatment} | {metric} | {result['mean_delta']:+.6f} | "
                f"[{result['ci_low']:+.6f}, {result['ci_high']:+.6f}] | "
                f"{_format_p(result['p_two_sided'])} | "
                f"{result['wins']}/{result['ties']}/{result['losses']} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The comparison changes only the query representation on the official "
            "102-query video subset. It does not establish generation quality, "
            "cross-dataset transfer, or an architecture improvement. Official video "
            "descriptions are source-page context, not independently authored user "
            "queries.",
            "",
            "All three runs use the same qids, qrels, corpus index, BM25 parameters, "
            "deterministic tie rule, code commit, and clean-tree manifest. Aggregate "
            "metrics were independently cross-checked with ir-measures.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.bootstrap_resamples < 1:
        raise SystemExit("--bootstrap-resamples must be positive")
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        raise SystemExit("--tolerance must be a finite non-negative number")

    input_root = (
        args.input_root
        if args.input_root.is_absolute()
        else PROJECT_ROOT / args.input_root
    )
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else PROJECT_ROOT / args.output_dir
    )
    conditions = {
        representation: load_condition(input_root, representation)
        for representation in REPRESENTATIONS
    }
    contract = validate_shared_contract(conditions)

    benchmark = load_benchmark_queries(
        "beir/nfcorpus/test",
        cache_dir=PROJECT_ROOT / "data/raw",
    )
    qids = set(conditions["title"].run)
    graded_qrels = {
        qid: dict(benchmark.graded_qrels[qid])
        for qid in sorted(qids)
    }
    aggregate, cross_checks = _condition_metrics(
        conditions,
        graded_qrels,
        tolerance=args.tolerance,
    )
    rows = build_per_query_rows(
        {name: condition.run for name, condition in conditions.items()},
        graded_qrels,
    )
    comparisons = build_paired_comparisons(
        rows,
        n_resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )

    summary: dict[str, object] = {
        "schema": "msmarco-genqa.nfcorpus-video-query-analysis.v1",
        "dataset": "beir/nfcorpus/test",
        "subset": "official_test_video",
        "contract": contract,
        "aggregate_metrics": aggregate,
        "independent_cross_checks": cross_checks,
        "paired_comparisons": comparisons,
        "bootstrap": {
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "confidence_level": 0.95,
        },
        "interpretation_boundary": (
            "Retrieval-only comparison of richer source representations on the "
            "official 102-query NFCorpus video subset."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "per_query.jsonl").open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "report.md").write_text(
        render_report(summary),
        encoding="utf-8",
        newline="\n",
    )

    print("NFCorpus video query-representation analysis complete")
    for name in REPRESENTATIONS:
        metrics = aggregate[name]
        print(
            f"  {name:24s} "
            f"MRR@10={metrics['mrr@10']:.6f} "
            f"nDCG@10={metrics['ndcg@10']:.6f} "
            f"Recall@100={metrics['recall@100']:.6f}"
        )
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()
