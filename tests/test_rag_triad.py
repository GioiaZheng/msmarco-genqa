from __future__ import annotations

import json
from pathlib import Path

import pytest

from msmarco_genqa.cli import rag_triad as cli
from msmarco_genqa.evaluation.rag_triad import (
    PredictionPairingError,
    UnsupportedEvaluatorError,
    build_triad_report,
    score_prediction_row,
    validate_evaluator,
)


def _row(
    qid: str,
    *,
    query: str = "what is the capital of australia",
    top_doc_ids: list[str] | None = None,
    passages: list[str] | None = None,
    prediction: str = "Canberra",
    references: list[str] | None = None,
) -> dict:
    return {
        "query_id": qid,
        "query": query,
        "top_doc_ids": top_doc_ids if top_doc_ids is not None else ["d1"],
        "passages": passages if passages is not None else ["Canberra is the capital of Australia."],
        "prediction": prediction,
        "references": references if references is not None else ["Canberra"],
    }


def test_build_triad_report_scores_two_aligned_configs():
    report = build_triad_report(
        {
            "bm25": [_row("q1", top_doc_ids=["d2"], prediction="Sydney")],
            "reranked": [_row("q1", top_doc_ids=["d1"], prediction="Canberra")],
        },
        qrels={"q1": {"d1"}},
        baseline_config="bm25",
    )

    summary = report["summary"]
    assert summary["configs"]["bm25"]["metrics"]["mean_context_relevance"] == 0.0
    assert summary["configs"]["reranked"]["metrics"]["mean_context_relevance"] == 1.0
    assert summary["configs"]["reranked"]["metrics"]["mean_answer_relevance"] == 1.0
    reranked_row = [r for r in report["per_query"] if r["config"] == "reranked"][0]
    assert reranked_row["movement"]["bucket"] == "new_hit"
    assert reranked_row["query_form"] == "what"


def test_validate_evaluator_rejects_unsupported_config():
    with pytest.raises(UnsupportedEvaluatorError, match="unsupported triad evaluator"):
        validate_evaluator("model-assisted")


def test_empty_context_scores_context_and_grounding_as_zero():
    scored = score_prediction_row(
        _row("q1", top_doc_ids=[], passages=[], prediction="Canberra"),
        config_name="bm25",
        qrels={"q1": {"d1"}},
    )

    assert scored["scores"]["context_relevance"] == 0.0
    assert scored["scores"]["groundedness"] == 0.0
    assert scored["scores"]["answer_relevance"] == 1.0
    assert scored["flags"]["empty_context"] is True


def test_unsupported_answer_is_visible_in_low_dimensions():
    scored = score_prediction_row(
        _row("q1", prediction="Sydney", references=["Canberra"]),
        config_name="reranked",
        qrels={"q1": {"d1"}},
    )

    assert scored["scores"]["context_relevance"] == 1.0
    assert scored["scores"]["groundedness"] == 0.0
    assert scored["scores"]["answer_relevance"] == 0.0
    assert scored["flags"]["low_dimensions"] == ["groundedness", "answer_relevance"]


def test_mismatched_prediction_rows_are_rejected():
    with pytest.raises(PredictionPairingError, match="different order"):
        build_triad_report(
            {
                "bm25": [_row("q1"), _row("q2")],
                "reranked": [_row("q2"), _row("q1")],
            }
        )


def test_cli_writes_triad_outputs(tmp_path: Path):
    bm25_path = tmp_path / "bm25.jsonl"
    reranked_path = tmp_path / "reranked.jsonl"
    qrels_path = tmp_path / "qrels.tsv"
    bm25_path.write_text(json.dumps(_row("q1", top_doc_ids=["d2"], prediction="Sydney")) + "\n")
    reranked_path.write_text(json.dumps(_row("q1", top_doc_ids=["d1"], prediction="Canberra")) + "\n")
    qrels_path.write_text("q1 0 d1 1\n")
    output_dir = tmp_path / "out"

    cli.main(
        [
            "--predictions",
            f"bm25={bm25_path}",
            "--predictions",
            f"reranked={reranked_path}",
            "--qrels",
            str(qrels_path),
            "--baseline-config",
            "bm25",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "per_query_triad.jsonl").exists()
    assert (output_dir / "low_score_cases.jsonl").exists()
    assert "RAG triad evaluation" in (output_dir / "report.md").read_text(encoding="utf-8")
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["inputs"]["predictions"]["bm25"] == str(bm25_path)
    assert metrics["inputs"]["qrels"] == str(qrels_path)
