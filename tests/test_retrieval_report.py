"""Tests for retrieval quality reporting."""

from __future__ import annotations

import json

import pytest

from msmarco_genqa.cli import retrieval_report as cli
from msmarco_genqa.evaluation.retrieval_report import (
    compare_runs_report,
    evaluate_run_report,
    load_qrels_tsv,
    read_run_doc_ids,
)
from msmarco_genqa.reranking.io import RunTsvFormatError


def test_load_qrels_tsv_keeps_empty_qrel_queries(tmp_path):
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text(
        "# qid iter doc rel\n"
        "q1 0 d1 1\n"
        "q2 0 d2 0\n"
        "q3 d3 1\n",
        encoding="utf-8",
    )

    loaded = load_qrels_tsv(qrels)

    assert loaded == {"q1": {"d1"}, "q2": set(), "q3": {"d3"}}


def test_evaluate_run_report_records_missing_and_empty_qrels():
    report = evaluate_run_report(
        {
            "q1": ["d1", "d2"],
            "q2": ["d2"],
            "q_missing": ["d9"],
        },
        {
            "q1": {"d1"},
            "q2": set(),
            "q_qrels_only": {"d7"},
        },
        run_name="bm25",
        ks_recall=(2,),
    )

    assert report["coverage"]["n_run_qids"] == 3
    assert report["coverage"]["n_evaluable_qids"] == 1
    assert report["coverage"]["n_skipped_missing_qrels"] == 1
    assert report["coverage"]["n_skipped_empty_qrels"] == 1
    assert report["coverage"]["n_qrels_only"] == 1
    assert report["metrics"]["mrr@10"] == pytest.approx(1.0)
    assert report["metrics"]["recall@2"] == pytest.approx(1.0)


def test_compare_runs_report_uses_matched_qids_and_records_coverage():
    report = compare_runs_report(
        baseline_runs={
            "q1": ["d0", "d1"],
            "q2": ["d2"],
            "q_baseline_only": ["d3"],
            "q_empty": ["d4"],
        },
        candidate_runs={
            "q1": ["d1", "d0"],
            "q2": ["d9"],
            "q_candidate_only": ["d5"],
            "q_empty": ["d4"],
        },
        qrels={
            "q1": {"d1"},
            "q2": {"d2"},
            "q_empty": set(),
            "q_baseline_only": {"d3"},
            "q_candidate_only": {"d5"},
        },
        baseline_name="dense",
        candidate_name="rrf",
        k_recall=2,
        ks_recall=(2,),
    )

    assert report["coverage"]["n_baseline_qids"] == 4
    assert report["coverage"]["n_candidate_qids"] == 4
    assert report["coverage"]["n_shared_qids"] == 3
    assert report["coverage"]["n_matched_evaluable_qids"] == 2
    assert report["coverage"]["n_baseline_only_qids"] == 1
    assert report["coverage"]["n_candidate_only_qids"] == 1
    assert report["coverage"]["n_shared_without_positive_qrels"] == 1
    assert report["deltas"]["mrr@10"] == pytest.approx((1.0 + 0.0) / 2 - (0.5 + 1.0) / 2)
    assert report["diagnostics"]["buckets"] == {"lost_hit": 1, "promoted": 1}
    assert len(report["per_query"]) == 2


def test_read_run_doc_ids_rejects_duplicate_doc_ids(tmp_path):
    run = tmp_path / "run.tsv"
    run.write_text(
        "q1\tQ0\td1\t1\t1.0\tbm25\n"
        "q1\tQ0\td1\t2\t0.5\tbm25\n",
        encoding="utf-8",
    )

    with pytest.raises(RunTsvFormatError, match="duplicate document id"):
        read_run_doc_ids(run)


def test_cli_evaluate_writes_metrics_and_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    run = tmp_path / "run.tsv"
    run.write_text(
        "q1\tQ0\td1\t1\t2.0\tbm25\n"
        "q2\tQ0\td9\t1\t1.0\tbm25\n",
        encoding="utf-8",
    )
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text("q1 0 d1 1\nq2 0 d2 1\n", encoding="utf-8")
    output_dir = tmp_path / "outputs" / "eval"

    cli.main(
        [
            "evaluate",
            "--run",
            str(run),
            "--qrels",
            str(qrels),
            "--run-name",
            "bm25",
            "--output-dir",
            str(output_dir),
            "--ks-recall",
            "1",
        ]
    )

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["run_name"] == "bm25"
    assert metrics["inputs"] == {"qrels": "qrels.tsv", "run": "run.tsv"}
    assert metrics["coverage"]["n_evaluable_qids"] == 2
    assert metrics["metrics"]["mrr@10"] == pytest.approx(0.5)
    assert "Retrieval quality report" in (output_dir / "report.md").read_text(encoding="utf-8")


def test_cli_compare_writes_comparison_and_per_query(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    baseline = tmp_path / "baseline.tsv"
    baseline.write_text(
        "q1\tQ0\td0\t1\t2.0\tdense\n"
        "q1\tQ0\td1\t2\t1.0\tdense\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.tsv"
    candidate.write_text(
        "q1\tQ0\td1\t1\t2.0\trrf\n"
        "q1\tQ0\td0\t2\t1.0\trrf\n",
        encoding="utf-8",
    )
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text("q1 0 d1 1\n", encoding="utf-8")
    output_dir = tmp_path / "outputs" / "compare"

    cli.main(
        [
            "compare",
            "--baseline-run",
            str(baseline),
            "--candidate-run",
            str(candidate),
            "--qrels",
            str(qrels),
            "--baseline-name",
            "dense",
            "--candidate-name",
            "rrf",
            "--output-dir",
            str(output_dir),
        ]
    )

    comparison = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["inputs"] == {
        "baseline_run": "baseline.tsv",
        "candidate_run": "candidate.tsv",
        "qrels": "qrels.tsv",
    }
    assert comparison["deltas"]["mrr@10"] == pytest.approx(0.5)
    assert comparison["diagnostics"]["buckets"] == {"promoted": 1}
    assert "per_query" not in comparison
    per_query_lines = (output_dir / "per_query.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(per_query_lines) == 1
    assert "Retrieval comparison report" in (output_dir / "report.md").read_text(
        encoding="utf-8"
    )
